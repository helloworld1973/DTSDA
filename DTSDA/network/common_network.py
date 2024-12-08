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
        """
        Generate Grad-CAM heatmap for the entire batch.

        Returns:
            numpy.ndarray: Grad-CAM heatmap for the batch, shape (N, H, W).
        """
        # Compute Grad-CAM heatmap for the batch
        gradients = self.gradients.cpu().detach().numpy()  # Shape: (N, C, H, W)
        activations = self.activations.cpu().detach().numpy()  # Shape: (N, C, H, W)
        weights = np.mean(gradients, axis=(2, 3))  # Global average pooling over H and W

        batch_size = activations.shape[0]
        cam_batch = []

        for b in range(batch_size):
            cam = np.zeros(activations.shape[2:], dtype=np.float32)  # Shape: (H, W)
            for i, w in enumerate(weights[b]):
                cam += w * activations[b, i]
            # Apply ReLU and normalize the heatmap
            cam = np.maximum(cam, 0)
            cam -= np.min(cam)
            cam /= np.max(cam) + 1e-8  # Avoid division by zero
            cam_batch.append(cam)

        return np.array(cam_batch)  # Shape: (N, H, W)


def visualize_gradcam(cam_batch, input_signal, round_number, input_labels):
    """
    Visualize the averaged Grad-CAM heatmaps overlayed on the averaged input signals per label.

    Args:
        cam_batch (numpy.ndarray): Batch of CAMs, shape (N, H, W).
        input_signal (numpy.ndarray): Input signals, shape (N, C, 1, T).
        round_number (int): Current training round or epoch number.
        input_labels (numpy.ndarray): Labels for each sample in the batch, shape (N,).

    Returns:
        None. Saves and displays the Grad-CAM visualization.
    """
    unique_labels = np.unique(input_labels)
    input_length = input_signal.shape[-1]  # Length of time-series data (T)
    cam_length = cam_batch.shape[-1]  # Length of CAMs (W)

    plt.figure(figsize=(15, 5 * len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        # Select samples corresponding to the current label
        label_indices = np.where(input_labels == label)[0]
        avg_cam = np.mean(cam_batch[label_indices], axis=0)  # Average CAM for the label
        avg_signal = np.mean(input_signal[label_indices], axis=0)  # Average signal for the label

        # Upsample CAM to match input signal length (T)
        original_indices = np.linspace(0, cam_length - 1, cam_length)
        new_indices = np.linspace(0, cam_length - 1, input_length)
        avg_cam_resized = np.interp(new_indices, original_indices, avg_cam.squeeze())

        # Normalize CAM for better visualization
        avg_cam_resized = (avg_cam_resized - avg_cam_resized.min()) / (avg_cam_resized.max() - avg_cam_resized.min())

        # Average signal across channels
        avg_signal_per_channel = np.mean(avg_signal, axis=0).squeeze()  # Shape: (T,)

        # Plotting
        plt.subplot(len(unique_labels), 1, idx + 1)
        plt.plot(avg_signal_per_channel, label=f"Label {label}: Avg Signal", color='blue')
        plt.imshow(avg_cam_resized[np.newaxis, :], cmap="jet", aspect="auto", alpha=0.5,
                   extent=(0, input_length, np.min(avg_signal_per_channel), np.max(avg_signal_per_channel)))
        plt.colorbar(label="Grad-CAM Intensity")
        plt.title(f"Label {label}: Averaged Grad-CAM and Signal")
        plt.xlabel("Time (samples)")
        plt.ylabel("Signal Amplitude")
        plt.legend()

    plt.tight_layout()
    save_path = f"gradcam_visualization_round_{round_number}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Grad-CAM visualization saved as {save_path}")
    plt.show()
