import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import os
from models import SiT_models
from diffusers.models import AutoencoderKL
import numpy as np

# --------------- Configuration ----------------
device = "cuda:0"
model_name = "SiT-B/8"
image_size = 128
num_classes = 1
vae_type = "ema"
ckpt_path = "results/005-SiT-B-8-Linear-velocity-None/checkpoints/0100000.pt"
dataset_dir = "datasets/"
N = 10
output_dir = "pose_eval/005/0100000"
os.makedirs(output_dir, exist_ok=True)

# --------------- Load Dataset -----------------
from dataset import SiTDataset
dataset = SiTDataset(dataset_dir, transform=None, image_size=image_size)
latent_size = image_size // 8

# --------------- Load VAE --------------------
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
vae.eval()

# --------------- Load Model & PoseEncoder -----
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
ema_state_dict = ckpt.get("ema", ckpt.get("model", None))
pose_encoder_state_dict = ckpt.get("pose_encoder", None)
assert ema_state_dict is not None, "Checkpoint must contain 'ema' or 'model' weights"
assert pose_encoder_state_dict is not None, "Checkpoint must contain 'pose_encoder' weights"

from pose_encoder import PoseEncoder
pose_encoder = PoseEncoder(latent_size=latent_size, channels=4).to(device)
pose_encoder.load_state_dict(pose_encoder_state_dict)
pose_encoder.eval()

model = SiT_models[model_name](
    input_size=latent_size,
    num_classes=num_classes,
).to(device)
model.load_state_dict(ema_state_dict)
model.eval()

# --------------- Inference Loop --------------
with torch.no_grad():
    for i in range(N):
        # ---- Get original image ----
        image, _ = dataset[i]

        # ---- Encode pose with PoseEncoder ----
        pose_raw = dataset.poses[i]   # shape [4, 4] or [16]
        if pose_raw.shape == (4, 4):
            pose_input = torch.tensor(pose_raw, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 4, 4]
        else:
            pose_input = torch.tensor(pose_raw, dtype=torch.float32).reshape(1, 4, 4).to(device)

        pose_encoded = pose_encoder(pose_input)  # [1, 4, latent_size, latent_size]

        # ---- Model input ----
        t = torch.zeros(1, device=device)    # [1]
        y = torch.zeros(1, dtype=torch.long, device=device)  # [1], adjust if needed

        latent_pred = model(pose_encoded, t, y=y)
        img_pred = vae.decode(latent_pred / 0.18215).sample    # [1, 3, H, W]

        # ---- Handle original image (as before) ----
        if isinstance(image, torch.Tensor):
            img_orig = image.unsqueeze(0).to(device)  # [1, 3, H, W]
        else:
            from torchvision import transforms
            img_orig = transforms.ToTensor()(image).unsqueeze(0).to(device)

        if img_orig.shape[2:] != img_pred.shape[2:]:
            img_orig = F.interpolate(img_orig, size=img_pred.shape[2:], mode='bilinear', align_corners=False)

        comparison = torch.cat([img_pred.cpu(), img_orig.cpu()], dim=0)
        save_image(comparison, os.path.join(output_dir, f"compare_{i}.png"),
                   nrow=2, normalize=True, value_range=(-1,1))

        print(f"Saved {output_dir}/compare_{i}.png: [0]=Generated (pose encoder, *no noise*), [1]=Original")

print("All comparisons saved in", output_dir)
