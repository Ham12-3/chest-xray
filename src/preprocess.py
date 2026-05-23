"""Preprocessing and augmentation transforms.

Train transforms add light augmentation. Eval transforms are deterministic.
Both resize to the configured size, convert to a tensor, and normalize with
ImageNet statistics, because we fine-tune ImageNet pretrained weights.

Note on horizontal flip: it swaps left and right. The heart sits on the left,
so a flip can confuse Cardiomegaly and laterality. It is OFF by default. Turn
it on only on purpose.
"""

from torchvision import transforms


def build_transforms(image_cfg, train):
    """Return the transform pipeline for train or eval."""
    size = image_cfg["size"]
    mean = image_cfg["mean"]
    std = image_cfg["std"]

    if train:
        return transforms.Compose(
            [
                # Mild zoom and crop. Keeps most of the lung field.
                transforms.RandomResizedCrop(size, scale=(0.9, 1.0)),
                transforms.RandomRotation(7),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
