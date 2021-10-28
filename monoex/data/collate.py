from monoex.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset, returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return dict(images=images, targets=targets, img_ids=img_ids)