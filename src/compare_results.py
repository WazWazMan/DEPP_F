from lpips_2imgs import lpips_2imgs
from pathlib import Path


models = {"base repaint", "basic ddpm", "improved repaint with and blur average on noise sampling",
          "improved repaint with average over noise sampling", "improved repaint with blur", "improved repaint"}

scores = {}
count = 0

masks_path = Path('./images/general_masks')
masks = [p for p in masks_path.iterdir() if p.is_file()]

images_path = Path('./images/512x512')
images = [p for p in images_path.iterdir() if p.is_file()]

# for img in images:
for i in range(0, 13, 1):
    img_path = Path(f"./images/512x512/p_{i}.png")
    img_path_s = f"./images/512x512/p_{i}.png"

    prompts_path = Path("./images/prompts") / img_path.stem
    prompts = [p for p in prompts_path.iterdir() if p.is_file()]

    for prompt_path in prompts:
        prompt_txt = ""
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt_txt = file.read()

        for mask_path in masks:
            count += 1
            for model in models:
                filePath = f"./result/{img_path.stem}/{prompt_path.stem}/{mask_path.stem}/{model}.png"
                if model not in scores:
                    scores[model] = 0.0
                scores[model] += lpips_2imgs(img_path_s,filePath)

for model in models:
    scores[model] /= count

for model in models:
    print(model,scores[model])
                