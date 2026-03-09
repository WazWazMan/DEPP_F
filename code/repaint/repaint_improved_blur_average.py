from PIL import Image
import torch
from tqdm import tqdm
from .repaint_improved_blur import RePaintImprovedBlur
import numpy as np

class RePaintImprovedBlueAverage(RePaintImprovedBlur):
    def __init__(self, pipe,avg_count=3):
        super().__init__(pipe)
        self.avg_count = avg_count

    @torch.no_grad()
    def _single_reverse_step(self, x: torch.Tensor, t: int, text_embeddings) -> torch.Tensor:
        mean_pred = torch.zeros_like(x)
        for _ in range(self.avg_count):
            jitter = torch.randn_like(x) * 0.01
            jittered_sample = x + jitter
            noise_pred = self.pipe.unet(
                jittered_sample, t, encoder_hidden_states=text_embeddings).sample
            
            mean_pred += noise_pred / self.avg_count

        mean = self.sqrt_one_over_alphas[t] * (x - self.betas[t] * mean_pred / self.sqrt_one_minus_alphas_cumprod[t])
        if t == 0:
            return mean
        else:
            noise = torch.randn_like(
                x) * torch.sqrt(self.posterior_variance[t])
            return mean + noise
