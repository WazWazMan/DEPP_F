from PIL import Image
import torch
import numpy as np


class VanilaImpainting:
    def __init__(self,pipe):
        self.device = device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = pipe
        self.model = pipe.unet
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler


    def impaint(self,image: Image.Image, mask: Image.Image, prompt=""):
        # convert img to tensor
        img_tensor = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float16)

        # encode image to latent space
        with torch.no_grad():
            init_latents = self.vae.encode(img_tensor).latent_dist.sample()
            init_latents = init_latents * self.vae.config.scaling_factor

        # create tensor mask from image
        latent_h, latent_w = init_latents.shape[2], init_latents.shape[3]
        mask_resized = mask.resize((latent_w, latent_h), Image.NEAREST)

        # mask is either 255 of 0 due to it bing black and white mask
        mask_tensor = torch.from_numpy(np.array(mask_resized)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float16)

        # Generate text embeddings for the UNet condition
        text_inputs = self.pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=self.pipe.tokenizer.model_max_length)
        text_embeddings = self.pipe.text_encoder(text_inputs.input_ids.to(self.device))[0]

        # Initialize x_T ~ N(0, I)
        latents = torch.randn_like(init_latents)

        with torch.no_grad():
            for t in self.scheduler.timesteps:
                # Step (a): Clamp known region at current noise level
                # Sample random noise epsilon ~ N(0, I)
                noise = torch.randn_like(init_latents)
                
                # Create x_t^{obs} = sqrt(alpha_bar_t)*X + sqrt(1-alpha_bar_t)*epsilon
                # The diffusers scheduler has a built-in function for this forward noising
                noisy_init_latents = self.scheduler.add_noise(init_latents, noise, t)
                
                # Apply Pi_t clamping: M * x_t^{obs} + (1 - M) * x_t
                latents = mask_tensor * noisy_init_latents + (1.0 - mask_tensor) * latents

                # Step (b): Predict noise epsilon_theta(x_t, t)
                noise_pred = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample

                # Step (c): Reverse sample x_{t-1} = mu + sigma_t * z
                # The scheduler's step() function handles this exact math
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Finalize: Guarantee exact agreement on known pixels (decoded)
        # x_0 <- M * X + (1-M) * x_0
        final_latents = mask_tensor * init_latents + (1.0 - mask_tensor) * latents
        
        # Decode back to pixel space
        final_latents = 1 / self.vae.config.scaling_factor * final_latents
        image = self.vae.decode(final_latents).sample
        
        # Convert tensor to PIL Image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        return Image.fromarray((image * 255).astype(np.uint8))