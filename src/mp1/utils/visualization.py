import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_first_prediction(images, binary_logits, instance_embeddings, binary_labels, instance_labels):
    """
    Visualize and return the combined visualization of the first image in the batch.
    """
    # Input image
    image = images[0].cpu().permute(1, 2, 0).squeeze().numpy()/255.  # Convert image to numpy

    # Binary ground truth and prediction
    binary_pred = torch.argmax(binary_logits[0], dim=0).cpu().numpy()  # Predicted segmentation
    binary_gt = binary_labels[0].cpu().numpy()  # Ground truth segmentation

    # Instance embedding visualization
    instance_emb = instance_embeddings[0].cpu()  # Instance embedding tensor
    instance_vis = instance_emb.squeeze()[0:3, :, :].argmax(axis=0).cpu().numpy()  # Argmax over first 3 channels

    # Normalize and prepare all images for consistent visualization
    image_rgb = np.repeat(image[:, :, None], 3, axis=2)  # Convert grayscale input to RGB
    binary_gt_rgb = np.stack((binary_gt,) * 3, axis=-1)  # Binary GT as grayscale
    binary_pred_rgb = np.stack((binary_pred,) * 3, axis=-1)  # Binary prediction as grayscale
    instance_vis_rgb = plt.cm.viridis(instance_vis / instance_vis.max())[:, :, :3]  # Colorize instance embedding

    # Combine the images into a single row
    combined_row = np.hstack((image_rgb, binary_gt_rgb, binary_pred_rgb, instance_vis_rgb))

    # # Display the combined row
    # plt.figure(figsize=(15, 5))
    # plt.imshow(combined_row)
    # plt.axis("off")
    # plt.title("Visualization: Input, GT, Prediction, Instance Embedding")
    # plt.show()

    return combined_row