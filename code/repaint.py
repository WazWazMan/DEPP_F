from PIL import Image
import torch
import numpy as np
from IPython.display import clear_output


class RePaint:
    def __init__(self, pipe):
        self.device = pipe.device
        self.pipe = pipe
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler

    def impaint(self, image: Image.Image, mask: Image.Image, prompt="", repaint_count=1, jump_length=0, timestamps=-1):
        # convert img to tensor
        img_tensor = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(
            0).to(self.device, dtype=torch.float16)

        # encode image to latent space
        with torch.no_grad():
            init_latents = self.vae.encode(img_tensor).latent_dist.sample()
            init_latents = init_latents * self.vae.config.scaling_factor

        # create tensor mask from image
        latent_h, latent_w = init_latents.shape[2], init_latents.shape[3]
        mask_resized = mask.resize((latent_w, latent_h), Image.NEAREST)

        # mask is either 255 of 0 due to it bing black and white mask
        mask_tensor = torch.from_numpy(np.array(mask_resized)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(
            0).to(self.device, dtype=torch.float16)

        # Generate text embeddings for the UNet condition
        text_inputs = self.pipe.tokenizer(prompt, return_tensors="pt", padding="max_length",
                                          truncation=True, max_length=self.pipe.tokenizer.model_max_length)
        text_embeddings = self.pipe.text_encoder(
            text_inputs.input_ids.to(self.device))[0]

        # Initialize x_T ~ N(0, I)
        latents = torch.randn_like(init_latents)

        # repaint stuff
        if (timestamps != -1):
            self.scheduler.set_timesteps(timestamps, device=self.device)
        times = self.scheduler.timesteps
        time = self.scheduler.timesteps.size()[0]
        r_count = repaint_count
        j_count = jump_length
        rs = 0

        curr_t = 0
        last_t = j_count

        with torch.no_grad():
            while curr_t < time:
                if curr_t % 50 == 0:
                    print(" ", flush=True)
                    print(curr_t, last_t, time, flush=True)

                t = times[curr_t]
            # for t in self.scheduler.timesteps:
                # Step (a): Clamp known region at current noise level
                # Sample random noise epsilon ~ N(0, I)
                noise = torch.randn_like(init_latents)

                # Create x_t^{obs} = sqrt(alpha_bar_t)*X + sqrt(1-alpha_bar_t)*epsilon
                # The diffusers scheduler has a built-in function for this forward noising
                noisy_init_latents = self.scheduler.add_noise(
                    init_latents, noise, t)

                # Apply Pi_t clamping: M * x_t^{obs} + (1 - M) * x_t
                latents = mask_tensor * noisy_init_latents + \
                    (1.0 - mask_tensor) * latents

                # Step (b): Predict noise epsilon_theta(x_t, t)
                noise_pred = self.unet(
                    latents, t, encoder_hidden_states=text_embeddings).sample

                # Step (c): Reverse sample x_{t-1} = mu + sigma_t * z
                # The scheduler's step() function handles this exact math
                latents = self.scheduler.step(
                    noise_pred, t, latents).prev_sample

                curr_t += 1
                if (curr_t == last_t):

                    rs += 1
                    if rs == r_count:
                        last_t += j_count
                        if (last_t > time):
                            j_count = time - j_count
                            last_t = time
                        rs = 0
                    else:
                        curr_t -= j_count
                        target_t_idx = curr_t - j_count
                        print(curr_t, target_t_idx)
                        assert(1 == 0)

        
                        # 2. Iteratively add noise back (Transition from x_{t-1} to x_t)
                        # We loop backwards through the scheduler's timesteps
                        for i in range(curr_t - 1, target_t_idx - 1, -1):
                            t_sub = times[i]
                            beta = self.scheduler.betas[t_sub]
                            
                            noise = torch.randn_like(latents)
                            # Standard RePaint transition: x_t = sqrt(1-beta)*x_{t-1} + sqrt(beta)*epsilon
                            latents = (1 - beta).sqrt() * latents + beta.sqrt() * noise
                        
                        # 3. Update the current index to the new "noisier" position
                        curr_t = target_t_idx

        # Finalize: Guarantee exact agreement on known pixels (decoded)
        # x_0 <- M * X + (1-M) * x_0
        final_latents = mask_tensor * init_latents + \
            (1.0 - mask_tensor) * latents

        # Decode back to pixel space
        final_latents = 1 / self.vae.config.scaling_factor * final_latents
        image = self.vae.decode(final_latents).sample

        # Convert tensor to PIL Image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
        return Image.fromarray((image * 255).astype(np.uint8))
