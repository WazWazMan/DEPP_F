import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the base model requested by your assignment
pipe = StableDiffusionPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-base",
    torch_dtype=torch.float16
).to(device)

# Force the scheduler to be standard DDPM to match your math
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

unet = pipe.unet
vae = pipe.vae
scheduler = pipe.scheduler


def inpaint_ddpm(init_latents, mask_tensor, prompt=""):
    # Generate text embeddings for the UNet condition
    text_inputs = pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length)
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    # Initialize x_T ~ N(0, I)
    latents = torch.randn_like(init_latents)
    
    # Set number of inference steps (e.g., 1000 for standard DDPM, or fewer for testing)
    scheduler.set_timesteps(1000, device=device)

    with torch.no_grad():
        for t in scheduler.timesteps:
            # Step (a): Clamp known region at current noise level
            # Sample random noise epsilon ~ N(0, I)
            noise = torch.randn_like(init_latents)
            
            # Create x_t^{obs} = sqrt(alpha_bar_t)*X + sqrt(1-alpha_bar_t)*epsilon
            # The diffusers scheduler has a built-in function for this forward noising
            noisy_init_latents = scheduler.add_noise(init_latents, noise, t)
            
            # Apply Pi_t clamping: M * x_t^{obs} + (1 - M) * x_t
            latents = mask_tensor * noisy_init_latents + (1.0 - mask_tensor) * latents

            # Step (b): Predict noise epsilon_theta(x_t, t)
            noise_pred = unet(latents, t, encoder_hidden_states=text_embeddings).sample

            # Step (c): Reverse sample x_{t-1} = mu + sigma_t * z
            # The scheduler's step() function handles this exact math
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Finalize: Guarantee exact agreement on known pixels (decoded)
    # x_0 <- M * X + (1-M) * x_0
    final_latents = mask_tensor * init_latents + (1.0 - mask_tensor) * latents
    
    # Decode back to pixel space
    final_latents = 1 / vae.config.scaling_factor * final_latents
    image = vae.decode(final_latents).sample
    
    # Convert tensor to PIL Image
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    return Image.fromarray((image * 255).astype(np.uint8))

def prepare_latents_and_mask(image: Image.Image, mask: Image.Image):
    # 1. Convert image to tensor and scale to [-1, 1]
    img_tensor = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float16)
    
    # Encode image to X (init_latents)
    with torch.no_grad():
        init_latents = vae.encode(img_tensor).latent_dist.sample()
        init_latents = init_latents * vae.config.scaling_factor

    # 2. Prepare Mask M (1 = keep, 0 = inpaint)
    # Resize mask to match latent dimensions (usually 1/8th the size)
    latent_h, latent_w = init_latents.shape[2], init_latents.shape[3]
    mask_resized = mask.resize((latent_w, latent_h), Image.NEAREST)
    
    mask_tensor = torch.from_numpy(np.array(mask_resized)).float() / 255.0
    # Ensure mask has shape [1, 1, H, W] for broadcasting
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)
    
    # If your input mask is 1 for "inpaint" and 0 for "keep", invert it here to match your math:
    # mask_tensor = 1.0 - mask_tensor 

    return init_latents, mask_tensor

from PIL import Image

# 1. Define your target size (must be divisible by 8)
target_size = (512, 512)

# 2. Load and prepare the Source Image
# .convert("RGB") ensures it has exactly 3 color channels (removes alpha/transparency if it's a PNG)
source_img = Image.open("photo.jpeg").convert("RGB")
source_img = source_img.resize(target_size)

# 3. Load and prepare the Mask Image
# .convert("L") converts it to a single-channel grayscale image
mask_img = Image.open("mask.png").convert("L")
mask_img = mask_img.resize(target_size)

# Assuming you have a PIL Image 'source_img' and 'mask_img' loaded
# source_img size must be divisible by 8 (e.g., 512x512)
# mask_img should be grayscale, where white (255) means KEEP, black (0) means INPAINT.

X_latents, M_tensor = prepare_latents_and_mask(source_img, mask_img)

# Run the algorithm
result_image = inpaint_ddpm(X_latents, M_tensor, prompt="a cozy bedroom")
result_image.show()