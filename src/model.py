"""Model definition: DenseNet-121 adapted for 14-output multi-label.

We take DenseNet-121 pretrained on ImageNet and replace its 1000-class
classifier with a 14-unit linear layer, one unit per finding.

The model outputs raw logits, NOT probabilities. BCEWithLogitsLoss applies
the sigmoid internally during training, which is numerically safer. At
inference time, apply torch.sigmoid to the logits to get per-finding
probabilities. There is no softmax anywhere (hard rule 2).
"""

import torch.nn as nn
from torchvision import models


def build_model(model_cfg):
    """Build the model from the config dict."""
    name = model_cfg["name"]
    if name != "densenet121":
        raise ValueError(f"unsupported model: {name}")

    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if model_cfg["pretrained"] else None
    net = models.densenet121(weights=weights)

    in_features = net.classifier.in_features
    net.classifier = nn.Linear(in_features, model_cfg["num_classes"])
    return net
