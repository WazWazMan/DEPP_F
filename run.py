from src import repaint,utils
from pathlib import Path

repaint = repaint.RePaint()

masks_path = Path('./images/general_masks')
masks = [p for p in masks_path.iterdir() if p.is_file()]

images_path = Path('./images/512x512')
images = [p for p in images_path.iterdir() if p.is_file()]

# for img in images:
for i in range(5,24,1):
    img_path = Path(f"./images/512x512/p_{i}.png")
    
    prompts_path = Path("./images/prompts") / img_path.stem
    prompts = [p for p in prompts_path.iterdir() if p.is_file()]

    for prompt_path in prompts:
        prompt_txt = ""
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt_txt = file.read()
            
        for mask_path in masks:

            folderPath = f"./result/{img_path.stem}/{prompt_path.stem}/{mask_path.stem}"
            print(folderPath)
            
            img = utils.open_image(img_path)
            mask = utils.open_mask(mask_path)
            result = repaint.run_all(img,mask,prompt_txt)
            for res in result:
                utils.save_image(res.image,folderPath,res.text)

            
