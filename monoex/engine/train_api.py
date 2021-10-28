import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from monoex.engine.infer_api import inference
from monoex.utils import comm
from monoex.utils.logger import MetricLogger


def reduce_loss_dict(loss_dict):
    world_size = comm.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_eval(cfg, model, data_loader, iteration):
    eval_types = ("detection", )
    dataset_name = cfg.DATASETS.TEST[0]
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, dataset_name, "inference_{}".format(iteration))
        os.makedirs(output_folder, exist_ok=True)

    eval_metric, eval_result, dist_ious = inference(
        model,
        data_loader,
        dataset_name=dataset_name,
        eval_types=eval_types,
        device=cfg.MODEL.DEVICE,
        output_folder=output_folder,
    )
    comm.synchronize()
    return eval_metric, eval_result, dist_ious


def do_train(
    cfg,
    distributed,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    warmup_scheduler,
    checkpointer,
    device,
    arguments,
):
    logger = logging.getLogger(__name__)
    logger.info("Start training")

    meters = MetricLogger(delimiter=" ", )
    max_iter = cfg.SOLVER.MAX_ITERATION
    start_iter = arguments["iteration"]

    # enable warmup
    if cfg.SOLVER.LR_WARMUP:
        assert warmup_scheduler is not None
        warmup_iters = cfg.SOLVER.WARMUP_STEPS
    else:
        warmup_iters = -1

    model.train()
    start_training_time = time.time()
    end = time.time()

    default_depth_method = cfg.MODEL.HEAD.OUTPUT_DEPTH
    grad_norm_clip = cfg.SOLVER.GRAD_NORM_CLIP

    if comm.get_local_rank() == 0:
        writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'writer/{}/'.format(cfg.START_TIME)))
        best_mAP = 0
        best_result_str = None
        best_iteration = 0
        eval_iteration = 0
        record_metrics = ['Car_bev_', 'Car_3d_']

    for data, iteration in zip(data_loader, range(start_iter, max_iter)):
        data_time = time.time() - end

        images = data["images"].to(device)
        targets = [target.to(device) for target in data["targets"]]

        loss_dict, log_loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        log_losses_reduced = sum(loss for key, loss in log_loss_dict.items() if key.find('loss') >= 0)
        meters.update(loss=log_losses_reduced, **log_loss_dict)

        optimizer.zero_grad()
        losses.backward()
        if grad_norm_clip > 0:
            clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()
        if iteration < warmup_iters:
            warmup_scheduler.step()  # del iteration in step()
        else:
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        iteration += 1
        arguments["iteration"] = iteration

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if comm.get_rank() == 0:
            # save learning rate
            writer.add_scalar('stat/lr', optimizer.param_groups[0]["lr"], iteration)
            depth_errors_dict = {key: meters.meters[key].value for key in meters.meters.keys() if key.find('MAE') >= 0}
            writer.add_scalars('train_metric/depth_err', depth_errors_dict, iteration)

            for name, meter in meters.meters.items():
                if name.find('MAE') >= 0:
                    continue
                if name in ['time', 'data']:
                    writer.add_scalar("stat/{}".format(name), meter.value, iteration)
                else:
                    writer.add_scalar("train_metric/{}".format(name), meter.value, iteration)

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.8f}",
                        "max men: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                )
            )

        if iteration % cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL == 0:
            logger.info(f'iteration = {iteration}, saving checkpoint ...')
            if comm.get_rank() == 0:
                checkpointer.save("model_checkpoint", **arguments)

        if iteration == max_iter and comm.get_rank() == 0:
            checkpointer.save("model_final", **arguments)

        if iteration % cfg.SOLVER.EVAL_INTERVAL == 0:
            if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
                cur_epoch = iteration // arguments["iter_per_epoch"]
                logger.info(f'epoch = {cur_epoch}, evaluate on validation set')
            else:
                logger.info(f'iteration = {iteration}, evaluate on validation set')

            result_dict, result_str, dis_ious = do_eval(cfg, model, data_loader_val, iteration)

            if comm.get_rank() == 0:
                # only record more accurate R40 results
                result_dict = result_dict[0]
                if len(result_dict) > 0:
                    for key, value in result_dict.items():
                        for metric in record_metrics:
                            if key.find(metric) >= 0:
                                threshold = key[len(metric):len(metric) + 4]
                                writer.add_scalar(
                                    "eval_{}_{}/{}".format(default_depth_method, threshold, key), float(value),
                                    eval_iteration + 1
                                )

                for key, value in dis_ious.items():
                    writer.add_scalar("ious_{}/{}".format(key, default_depth_method), value, eval_iteration + 1)

                # record the best model according to the AP_3D, Car, Moderate, IoU=0.7
                important_key = '{}_3d_{:.2f}/moderate'.format('Car', 0.7)
                eval_mAP = float(result_dict[important_key])
                if eval_mAP >= best_mAP:
                    # save best mAP and corresponding iterations
                    best_mAP = eval_mAP
                    best_iteration = iteration
                    best_result_str = result_str
                    checkpointer.save("model_moderate_best_{}".format(default_depth_method), **arguments)

                    if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
                        logger.info(
                            'epoch = {}, best_mAP = {:.2f}, update best ckpt for depth {} \n'.format(
                                cur_epoch, eval_mAP, default_depth_method
                            )
                        )
                    else:
                        logger.info(
                            'iteration = {}, best_mAP = {:.2f}, update best ckpt for depth {} \n'.format(
                                iteration, eval_mAP, default_depth_method
                            )
                        )

                eval_iteration += 1

            model.train()
            comm.synchronize()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))

    if comm.get_rank() == 0:
        logger.info(
            "Total training time: {} ({:.4f} s / it), best model is achieved at iteration = {}".format(
                total_time_str,
                total_training_time / (max_iter),
                best_iteration,
            )
        )
        logger.info('The best performance details:')
        logger.info('\n' + best_result_str)
