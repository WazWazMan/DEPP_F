from datasets import load_from_disk
from PIL import Image
from src import repaint,utils
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Run the model on a dataset.")
parser.add_argument("--start", type=int, default=0, help="Starting index for the dataset.")
parser.add_argument("--count", type=int, default=1, help="Number of images to process.")
args = parser.parse_args()

repaint = repaint.RePaint()
dataset = load_from_disk("coco_200_masks")

start_index = args.start
end_index = start_index + args.count

for i in range(start_index, end_index):
# for i, example in enumerate(dataset):
    print(f"Running example {i + 1}/200...")
    
    example = dataset[i]
    current_image = example["image"]
    current_mask = example["mask"]
    current_prompt = example["description"]

    result = repaint.run_all(
        img=current_image,
        mask=current_mask,
        prompt=current_prompt,
        j=10,
        r=10,
        seed=42
    )
    folderPath = f"./result_db/{i}"
    for res in result:
        utils.save_image(res.image,folderPath,res.text)