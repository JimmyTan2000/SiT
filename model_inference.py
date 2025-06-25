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
ckpt_path = "results/004-SiT-B-8-Linear-velocity-None/ema_sampling_checkpoints/ema_sampling_0036000.pt"
dataset_dir = "datasets/"
N = 10
output_dir = "pose_eval/004/0036000"
os.makedirs(output_dir, exist_ok=True)

# --------------- Load Dataset -----------------
from dataset import SiTDataset
# Load once to get the data structure (but we'll bypass __getitem__ for pose)
dataset = SiTDataset(dataset_dir, transform=None, image_size=image_size)
latent_size = image_size // 8

# --------------- Load VAE --------------------
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
vae.eval()

# --------------- Load Model ------------------
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
ema_state_dict = ckpt.get("ema", ckpt.get("model", None))
assert ema_state_dict is not None, "Checkpoint must contain 'ema' or 'model' weights"
model = SiT_models[model_name](
    input_size=image_size // 8,
    num_classes=num_classes,
).to(device)
model.load_state_dict(ema_state_dict)
model.eval()

# --------------- Inference Loop --------------
with torch.no_grad():
    for i in range(N):
        # ---- Get original image (from __getitem__, so you get transform etc) ----
        image, _ = dataset[i]

        # ---- Use *clean* pose: extract raw pose, reshape/tile to latent shape ----
        pose_raw = dataset.poses[i]           # shape [16] or [1, 16]
        pose = pose_raw.reshape(1, 16)
        # Tile to (latent_size*latent_size, 1) then reshape
        pose_block = np.tile(pose, (latent_size * latent_size // 16, 1)).reshape(latent_size, latent_size)
        pose_clean = np.stack([pose_block]*4)  # [4, latent_size, latent_size]
        pose_tensor = torch.tensor(pose_clean, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 4, latent_size, latent_size]

        # ---- Model input ----
        t = torch.zeros(1, device=device)    # [1]
        y = torch.zeros(1, dtype=torch.long, device=device)  # [1], adjust if needed

        latent_pred = model(pose_tensor, t, y=y)
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

        print(f"Saved {output_dir}/compare_{i}.png: [0]=Generated (pose *no noise*), [1]=Original")

print("All comparisons saved in", output_dir)
