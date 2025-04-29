import os
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from peft import get_peft_model, LoraConfig

# ----- CONFIG -----
MODEL_ID = "runwayml/stable-diffusion-v1-5"
INSTANCE_DIR = "./face_training/ronaldo"
INSTANCE_TOKEN = "<ronaldo>"
OUTPUT_DIR = "./lora-output-boy"
BATCH_SIZE = 1
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 512

# ----- Dataset Loader -----
class FaceDataset(Dataset):
    def __init__(self, image_dir, prompt_token):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]
        self.prompt = f"a photo of {prompt_token}"
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "prompt": self.prompt
        }

from huggingface_hub import login
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers import UNet2DConditionModel

# Paste your Hugging Face token here


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Load Stable Diffusion Base
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

# ----- Load pipeline -----
tokenizer = pipe.tokenizer
unet = pipe.unet
text_encoder = pipe.text_encoder
scheduler = DDPMScheduler.from_config(pipe.scheduler.config)


# ✅ Collect trainable parameters
# trainable_params = [p for p in unet.parameters() if p.requires_grad]
# print("Got trainable params: ", trainable_params)

print("Got attn processor: ", unet.attn_processors)

# ----- Prepare data -----
dataset = FaceDataset(INSTANCE_DIR, INSTANCE_TOKEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ----- Optimizer -----
optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)

vae = pipe.vae
vae.requires_grad_(False)  # freeze VAE

# ----- Training -----
unet.train()
for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
        prompt = batch["prompt"]

        # Encode text
        input_ids = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer.model_max_length).input_ids.to(device)
        encoder_hidden_states = text_encoder(input_ids)[0]

        # Add noise
        # Encode image to latents
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215  # scale per SD-1.5

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix({"loss": loss.item()})

# ----- Save LoRA weights -----
unet.save_attn_procs(OUTPUT_DIR)
print("✅ LoRA training complete!")