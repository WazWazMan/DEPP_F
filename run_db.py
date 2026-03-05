from datasets import load_from_disk
from PIL import Image
from src import repaint,utils
from pathlib import Path

repaint = repaint.RePaint()
dataset = load_from_disk("coco_200_masks")

start_index = 1
for i in range(start_index, len(dataset)):
# for i, example in enumerate(dataset):
    print(f"Running example {i + 1}/200...")
    
    example = dataset[i]
    current_image = example["image"]
    current_mask = example["mask"]
    current_prompt = example["description"]
    
    result = repaint.run_all(
        img=current_image,
        mask=current_mask,
        prompt=current_prompt
    )
    folderPath = f"./result_db/{i}"
    for res in result:
        utils.save_image(res.image,folderPath,res.text)