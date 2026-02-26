from PIL import Image
import torch
from tqdm import tqdm
from repaint_base import RePaintBase

class RePaintImproved(RePaintBase):
    def __init__(self, pipe):
        super().__init__(pipe)

    def impaint(self, image: Image.Image, mask: Image.Image, prompt="", j:int=10, r:int = 5, timestamps=-1,jumps_every=50 ):
        jumps = self.__get_jumps(jumps_every=jumps_every,r=r)
        original_tensor = self.__image_to_tensor(image)
        mask_tensor = self.__mask_to_tensor(mask,original_tensor.shape)
        text_embeddings= self.__embed_test(prompt)

        sample = torch.randn_like(original_tensor).to(self.device)
        
        print("beginning impainting")
        for t in tqdm(self.scheduler.timesteps):
            while len(jumps) > 0 and jumps[0] == t:
                jumps = jumps[1:]
                sample = self.__forward_j_steps(sample, t, j)

                for override_t in range(t + j, t, -1):
                    x_known_inner = self.__zero_to_t(original_tensor, override_t)
                    x_unknown_inner = self.__single_reverse_step(sample, override_t, text_embeddings)
                    sample = mask_tensor * x_known_inner + (1 - mask_tensor) * x_unknown_inner


            x_known = self.__zero_to_t(original_tensor, t)
            x_unknown = self.__single_reverse_step(sample, t, text_embeddings)
            sample = mask_tensor * x_known + (1-mask_tensor) * x_unknown

        return self.__tensor_to_image(sample)
