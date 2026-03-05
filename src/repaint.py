from .repaint_base import RePaintBase
from .repaint_improve_jumps import RePaintImproved
from .repaint_improved_blur import RePaintImprovedBlur
from .repaint_improved_average import RePaintImprovedAverage
from .repaint_improved_blur_average import RePaintImprovedBlueAverage
from diffusers import StableDiffusionPipeline, DDPMScheduler
from PIL import Image

import torch

class Result:
    def __init__(self, image: Image.Image,text: Image.Image):
        self.image = image
        self.text = text


class RePaint:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "sd2-community/stable-diffusion-2-base"
        self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).to(device)

        
        self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)

    def set_seed(self, seed:int = None):
        if not seed:
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run_all(self,img: Image.Image,mask: Image.Image,prompt:str,j:int=10, r:int = 5,avg_count:int = 3,seed:int = None):
        images: list[Result] = []

        print("running basic ddpm")
        self.set_seed(seed)
        images.append(
            Result(
                self.run_ddpm_base(img,mask,prompt),
                "basic ddpm"
            )
        )

        # print("running basic repaint")
        # self.set_seed(seed)
        # images.append(
        #     Result(
        #         self.run_repaint_base(img,mask,prompt,j,r),
        #         "base repaint"
        #     )
        # )

        print("running improved repaint")
        self.set_seed(seed)
        images.append(
            Result(
                self.run_repaint_improved(img,mask,prompt,j,r),
                "improved repaint"
            )
        )

        print("running improved repaint with blur")
        self.set_seed(seed)
        images.append(
            Result(
                self.run_repaint_improved_blur(img,mask,prompt,j,r),
                "improved repaint with blur"
            )
        )

        print("running improved repaint with average over noise sampling")
        self.set_seed(seed)
        images.append(
            Result(
                self.run_repaint_improved_average(img,mask,prompt,j,r,avg_count),
                "improved repaint with average over noise sampling"
            )
        )

        print("running improved repaint with and blur average on noise sampling")
        self.set_seed(seed)
        images.append(
            Result(
                self.run_repaint_improved_average(img,mask,prompt,j,r,avg_count),
                "improved repaint with and blur average on noise sampling"
            )
        )

        return images

    def run_ddpm_base(self,img: Image.Image,mask: Image.Image,prompt:str):
        repaint: RePaintBase = RePaintBase(self.pipe)
        image = repaint.impaint(img,mask,prompt,j=1, r=1)
        return image

    def run_repaint_base(self,img: Image.Image,mask: Image.Image,prompt:str,j:int=10, r:int = 5):
        repaint: RePaintBase = RePaintBase(self.pipe)
        image = repaint.impaint(img,mask,prompt,j, r)
        return image

    def run_repaint_improved(self,img: Image.Image,mask: Image.Image,prompt:str,j:int=10, r:int = 5):
        repaint: RePaintImproved = RePaintImproved(self.pipe)
        image = repaint.impaint(img,mask,prompt,j, r)
        return image

    def run_repaint_improved_blur(self,img: Image.Image,mask: Image.Image,prompt:str,j:int=10, r:int = 5):
        repaint: RePaintImprovedBlur = RePaintImprovedBlur(self.pipe)
        image = repaint.impaint(img,mask,prompt,j, r)
        return image

    def run_repaint_improved_average(self,img: Image.Image,mask: Image.Image,prompt:str,j:int=10, r:int = 5,avg_count:int = 3):
        repaint: RePaintImprovedAverage = RePaintImprovedAverage(self.pipe,avg_count)
        image = repaint.impaint(img,mask,prompt,j, r)
        return image

    def run_repaint_improved_blur_average(self,img: Image.Image,mask: Image.Image,prompt:str,j:int=10, r:int = 5,avg_count:int = 3):
        repaint: RePaintImprovedBlueAverage = RePaintImprovedBlueAverage(self.pipe,avg_count)
        image = repaint.impaint(img,mask,prompt,j, r)
        return image