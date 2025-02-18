import torch
import numpy as np
import cv2 as cv2
from sklearn.cluster import DBSCAN


class LaneDetector:
    DEFAULT_IMAGE_SIZE = (512, 256)

    def __init__(self, enet, device="cuda"):
        self._enet = enet
        self._device = device
        self._eps = 1.0

    def __call__(self, image):
        # Preprocess the image
        image = self._preprocess_image(image)
        
        # Get logits and embeddings
        binary_logits, instance_embeddings = self._enet(image)
        
        # Generate segmentation map and cluster instances
        segmentation_map = binary_logits.squeeze().argmax(dim=0)
        instances_map = self._cluster(segmentation_map, instance_embeddings)
        
        print(f"Detected {len(instances_map.unique()) - 1} lanes")
        return instances_map.cpu().numpy()

    def _cluster(self, segmentation_map, instance_embeddings):
        segmentation_map = segmentation_map.flatten()
        instance_embeddings = instance_embeddings.squeeze().permute(1, 2, 0).reshape(segmentation_map.shape[0], -1)
        assert segmentation_map.shape[0] == instance_embeddings.shape[0]

        mask_indices = segmentation_map.nonzero().flatten()
        cluster_data = instance_embeddings[mask_indices].detach().cpu()

        # Apply DBSCAN clustering
        clusterer = DBSCAN(eps=self._eps)
        labels = clusterer.fit_predict(cluster_data)
        labels = torch.tensor(labels, dtype=instance_embeddings.dtype, device=self._device)

        instances_map = torch.zeros(instance_embeddings.shape[0], dtype=instance_embeddings.dtype, device=self._device)
        instances_map[mask_indices] = labels
        instances_map = instances_map.reshape(self.DEFAULT_IMAGE_SIZE[::-1])

        return instances_map

    def _preprocess_image(self, image):
        image = cv2.resize(image, self.DEFAULT_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[..., None]
        image = torch.from_numpy(image).float().permute((2, 0, 1)).unsqueeze(dim=0).to(self._device)
        return image
