import torch
from monoex.modeling.backbones.dlanet import DLACompose

if __name__ == '__main__':
    CONV_BODY = "dla34"
    PRETRAIN = False
    DOWN_RATIO = 4
    NORM = "BN"

    model = DLACompose(
        base_name=CONV_BODY,
        pretrained=PRETRAIN,
        down_ratio=DOWN_RATIO,
        last_level=5,
        use_dcn=True,
        norm=NORM,
    ).cuda()

    print(model)

    dummy_input = torch.rand(1, 3, 384, 1280).cuda()
    print("input size: ", dummy_input.size())

    dummy_output = model(dummy_input)
    print("output size: ", dummy_output.size())
