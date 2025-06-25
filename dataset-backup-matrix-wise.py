from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms
import torch

class SiTDataset(Dataset):
    def __init__(self, pth=None, transform=None, image_size=256):   
        self.pth = []
        self.images = []
        self.poses = []
        self.focal_length = []
        self.transform = transform

        self.image_size = image_size                      
        self.latent_size = image_size // 8                

        if pth is not None:
            self.pth = pth

            images_list = []
            poses_list = []

            for filename in os.listdir(pth):
                filepath = os.path.join(pth, filename)
                loaded = np.load(filepath)
                images_list.append(loaded["images"])
                poses_list.append(loaded["poses"])

            self.focal_length = loaded["focal"]
            self.images = np.concatenate(images_list, axis=0)
            self.poses = np.concatenate(poses_list, axis=0)

    def __getitem__(self, index):
        image = self.images[index]
        pose = self.poses[index]  # [4, 4]

        if self.transform:
            image = self._to_pil_image(image)
            image = self.transform(image)

        # === Add Gaussian noise to the 4x4 pose matrix (mean=0, std=1) ===
        noise = np.random.normal(0, 0.01, (4, 4))    # Only to the 4x4
        pose_noisy = pose + noise                 # [4, 4] noisy but structure preserved

        # === Now tile the noisy pose to match the expected latent input ===
        pose_vector = pose_noisy.reshape(1, 16)   # [1, 16]
        latent_size = self.latent_size
        pose_block = np.tile(pose_vector, (latent_size * latent_size // 16, 1)).reshape(latent_size, latent_size)
        pose_tiled = np.stack([pose_block]*4)     # [4, latent_size, latent_size]

        pose_tiled = torch.tensor(pose_tiled, dtype=torch.float32)

        return image, pose_tiled

    def _to_pil_image(self, image):
        return transforms.ToPILImage()(image)

    def __len__(self):
        if self.images is not None:
            return self.images.shape[0]
        return 0

    def save(self):
        np.savez_compressed(self.pth, images=self.images, poses=self.poses, focal=self.focal_length)
