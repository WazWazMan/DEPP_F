from datasets import load_from_disk
from PIL import Image
from src import repaint,utils
from pathlib import Path

repaint = repaint.RePaint()
dataset = load_from_disk("coco_200_masks")

start_index = 0
for i in range(start_index, len(dataset)):
# for i, example in enumerate(dataset):
    print(f"Running example {i + 1}/200...")
    
    example = dataset[i]
    current_image = example["image"]
    current_mask = example["mask"]
    current_prompt = example["description"]
    
    # print("running improved repaint")
        # self.set_seed(seed)
        # images.append(
        #     Result(
        #         self.run_repaint_improved(img,mask,prompt,j,r),
        #         "improved repaint"
        #     )
        # )
    result = repaint.run_all(
        img=current_image,
        mask=current_mask,
        prompt=current_prompt,
        j=10,
        r=20
    )
    folderPath = f"./result_db/{i}"
    for res in result:
        utils.save_image(res.image,folderPath,res.text)