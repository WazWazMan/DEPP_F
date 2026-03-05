from .repaint_improve_jumps import RePaintImproved
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class RePaintImprovedBlur(RePaintImproved):
    def __init__(self, pipe):
        super().__init__(pipe)
    
    def impaint(self, image: Image.Image, mask: Image.Image, prompt="",
     j:int=10, r:int = 5, timestamps=-1,jumps_every=50 ):
        jumps = self._get_jumps(jumps_every=jumps_every,r=r)
        original_tensor = self._image_to_tensor(image)
        mask_tensor = self._mask_to_tensor(mask,original_tensor.shape)
        og_mask = mask_tensor

        text_embeddings= self._embed_test(prompt)
        mask_tensor = TF.to_tensor(image.convert("L")).unsqueeze(0).to(self.device, dtype=torch.float16)
        soft_mask = F.interpolate(mask_tensor, size=(original_tensor.shape[2], original_tensor.shape[3]), mode='bilinear', align_corners=False)

        mask_tensor = og_mask * soft_mask

        sample = torch.randn_like(original_tensor).to(self.device)
        
        print("beginning impainting")
        for t in tqdm(self.scheduler.timesteps):
            while len(jumps) > 0 and jumps[0] == t:
                jumps = jumps[1:]
                sample = self._forward_j_steps(sample, t, j)

                for override_t in range(t + j, t, -1):
                    x_known_inner = self._zero_to_t(original_tensor, override_t)
                    x_unknown_inner = self._single_reverse_step(sample, override_t, text_embeddings)
                    sample = mask_tensor * x_known_inner + (1 - mask_tensor) * x_unknown_inner


            x_known = self._zero_to_t(original_tensor, t)
            x_unknown = self._single_reverse_step(sample, t, text_embeddings)
            sample = mask_tensor * x_known + (1-mask_tensor) * x_unknown
        
        sample = og_mask * x_known + (1-og_mask) * x_unknown

        return self._tensor_to_image(sample)
