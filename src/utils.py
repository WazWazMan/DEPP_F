from PIL import Image, ImageOps
import torch
import matplotlib.pyplot as plt


def mask_image(image: Image.Image, mask: Image.Image):
    border_size = 5
    frame_color = (255, 215, 0) # Gold

    image.putalpha(mask)
    framed_img = ImageOps.expand(image, border=border_size, fill=frame_color)
    return framed_img
    image.show()


def open_image(path, target_size = (512, 512)) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(target_size)
    return image

def open_mask(path, target_size = (512, 512)) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image = image.resize(target_size)
    return image

def print_mask(self, mask_tensor: torch.Tensor, title: str = "Mask Visualization"):
    mask = mask_tensor.detach().cpu().float()
        
    # 2. Remove Batch and Channel dimensions [1, 1, H, W] -> [H, W]
    if mask.ndim == 4:
        mask = mask.squeeze(0).squeeze(0)
    elif mask.ndim == 3:
        mask = mask.squeeze(0)
    
    # 3. Plotting
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray', vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis('off')
    plt.show()