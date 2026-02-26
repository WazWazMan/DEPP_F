import PIL.Image
from pathlib import Path

TARGET_SIZE = (512, 512)

directory_path = Path('./full_size')
output_path = Path('./512x512')
output_path.mkdir(exist_ok=True)

files = [p for p in directory_path.iterdir() if p.is_file()]

for i, p in enumerate(files):
    if (not p.is_file):
        continue
    try:
        image = PIL.Image.open(p)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(TARGET_SIZE)
        image.save(output_path / f"p_{i}.png")
    except Exception as e:
        pass
