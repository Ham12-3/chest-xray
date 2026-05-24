"""Grad-CAM heatmaps for DenseNet-121.

Grad-CAM (Gradient-weighted Class Activation Mapping) shows which regions of
the image the model focused on when making a prediction.

For DenseNet-121 the target layer is features.norm5, the batch normalisation
after the last dense block. Its output is 1024 channels at 7x7 spatial
resolution for a 224x224 input. The heatmap is upsampled back to the input
size and overlaid on the original image.

Usage:
    cam = GradCAM(model)
    heatmap = cam.generate(image_tensor, class_index)   # (H, W) numpy array 0-1
    overlay = cam.overlay(pil_image, heatmap)           # PIL Image with red overlay
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM for DenseNet-121.

    Hooks the last dense block (features.norm5) to capture activations and
    gradients needed to build the heatmap.
    """

    def __init__(self, model):
        self.model = model
        self._activations = None
        self._gradients = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        target = self.model.features.norm5

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(target.register_forward_hook(forward_hook))
        self._hooks.append(target.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(self, image_tensor, class_index):
        """Return a (H, W) heatmap array, values in [0, 1].

        image_tensor: (1, 3, H, W) float tensor, already normalised.
        class_index: integer 0-13, which of the 14 findings to explain.
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(False)

        self.model.zero_grad()
        logits = self.model(image_tensor)

        # Backprop the score for this single finding.
        score = logits[0, class_index]
        score.backward()

        # Global average pool the gradients: (C,)
        weights = self._gradients[0].mean(dim=(1, 2))

        # Weight the activations: (C, H, W) -> (H, W)
        cam = (weights[:, None, None] * self._activations[0]).sum(dim=0)
        cam = F.relu(cam)

        # Normalise to [0, 1].
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()

    def overlay(self, pil_image, heatmap, alpha=0.4):
        """Blend a heatmap onto a PIL image and return a new PIL image.

        heatmap: (H, W) float array in [0, 1].
        alpha: how strongly to show the heatmap (0 = invisible, 1 = full).
        """
        # Resize heatmap to match the input image.
        h, w = pil_image.height, pil_image.width
        heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (w, h), Image.BILINEAR
        )

        # Apply a red colormap: high activation = red, low = transparent.
        heatmap_arr = np.array(heatmap_resized, dtype=np.float32) / 255.0
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = (heatmap_arr * 255).astype(np.uint8)       # red
        rgba[:, :, 3] = (heatmap_arr * alpha * 255).astype(np.uint8)  # alpha

        base = pil_image.convert("RGBA")
        overlay_img = Image.fromarray(rgba, mode="RGBA")
        blended = Image.alpha_composite(base, overlay_img)
        return blended.convert("RGB")
