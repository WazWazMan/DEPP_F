from PIL import Image, ImageOps
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def mask_image(image: Image.Image, mask: Image.Image):
    border_size = 5
    frame_color = (255, 215, 0)

    image.putalpha(mask)
    framed_img = ImageOps.expand(image, border=border_size, fill=frame_color)
    return framed_img


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

def save_image(img: Image.Image, folder: str, name: str):
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    full_path = (folder_path / name).with_suffix('.png')

    img.save(full_path)