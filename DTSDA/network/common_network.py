import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256):
        super(feat_classifier, self).__init__()
        self.fc = nn.Linear(bottleneck_dim, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x


class GradCAM:
    def __init__(self, model, target_layer_name):
        """
        Initialize Grad-CAM.
        :param model: PyTorch model
        :param target_layer_name: Name of the target convolutional layer in featurizer
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Hook into the featurizer's target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(self._forward_hook)
                module.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output  # Capture activations (feature maps)

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # Capture gradients w.r.t. the target layer

    def generate_heatmap(self):
        # Compute Grad-CAM heatmap
        gradients = self.gradients.cpu().detach().numpy()
        activations = self.activations.cpu().detach().numpy()
        weights = np.mean(gradients, axis=(2, 3))  # Global average pooling

        cam = np.zeros(activations.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i]

        # Apply ReLU and normalize the heatmap
        cam = np.maximum(cam, 0)
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

def visualize_gradcam(cam, input_signal, round):
    """
    Visualize the Grad-CAM heatmap overlayed on the time-series input signal.

    Args:
        cam (numpy.ndarray): The class activation map, shape (1, 69).
        input_signal (numpy.ndarray): The original time-series signal, shape (6, 1, 300).
                                      Assumes a single sample with 6 channels.

    Returns:
        None. Displays the plot with Grad-CAM overlay.
    """
    # Ensure the input signal has 6 channels
    if input_signal.shape[0] != 6 or input_signal.shape[1] != 1:
        raise ValueError("Input signal must have shape (6, 1, 300) for a single sample.")

    # Extract relevant dimensions
    cam_length = cam.shape[-1]  # Length of the CAM (69)
    input_length = input_signal.shape[-1]  # Length of the time-series input (300)

    # Convert CAM to a PyTorch tensor and reshape
    cam_tensor = torch.tensor(cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 69)

    # Try resizing CAM using PyTorch interpolation
    try:
        cam_resized = F.interpolate(cam_tensor, size=(input_length,), mode='linear', align_corners=True)
        cam_resized = cam_resized.squeeze().numpy()  # Reshape to (300,)
    except ValueError:
        # Fallback to NumPy interpolation if PyTorch fails
        original_indices = np.linspace(0, cam_length - 1, cam_length)
        new_indices = np.linspace(0, cam_length - 1, input_length)
        cam_resized = np.interp(new_indices, original_indices, cam[0])

    # Normalize CAM for better visualization (0 to 1 range)
    cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())

    # Plot each channel of the input signal with the CAM overlay
    plt.figure(figsize=(12, 10))
    for channel in range(input_signal.shape[0]):  # Iterate over 6 channels
        plt.subplot(input_signal.shape[0], 1, channel + 1)
        plt.plot(input_signal[channel, 0, :], label=f"Channel {channel + 1}", color='blue')
        plt.imshow(cam_resized[np.newaxis, :], cmap="jet", aspect="auto", alpha=0.5,
                   extent=(0, input_length, np.min(input_signal[channel, 0, :]),
                           np.max(input_signal[channel, 0, :])))
        plt.colorbar(label="Grad-CAM Intensity")
        plt.title(f"Channel {channel + 1}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Signal Amplitude")
        plt.legend()
    plt.tight_layout()

    # Save the figure
    save_path = f"gradcam_visualization_round_{round}.png"
    plt.savefig(save_path, dpi=300)  # Save at high resolution (300 DPI)
    print(f"Grad-CAM visualization saved as {save_path}")

    # Show the figure
    plt.show()
