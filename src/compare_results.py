from ssim import ssim_2imgs
from lpips_2imgs import lpips_2imgs
from datasets import load_from_disk
from tqdm import tqdm
import argparse
import lpips
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def compare_dataset_clip(dataset, models, device="cuda" if torch.cuda.is_available() else "cpu"):
    scores = {}
    count = 0
    
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    
    for i, example in tqdm(enumerate(dataset)):
        prompt = example["description"] 

        count += 1
        for model_name in models:
            img_path = f"./result_db/{i}/{model_name}.png"
            
            img_to_compare = Image.open(img_path).convert("RGB")

            inputs = processor(text=[prompt], images=img_to_compare, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                clip_score = outputs.logits_per_image.item() 

            scores[model_name] += clip_score

    for model_name in models:
        scores[model_name] /= count

    return scores

def compare_dataset_ssim(dataset,models):
    scores = {}
    count = 0

    # for i in tqdm(range(40,42)):
    for i, example in tqdm(enumerate(dataset)):
        # example = dataset[i]
        og_image = example["image"]

        count += 1
        for model in models:
            img_to_compare = f"./result_db/{i}/{model}.png"
            if model not in scores:
                scores[model] = 0.0
            scores[model] += ssim_2imgs(og_image,img_to_compare)

    for model in models:
        scores[model] /= count

    return scores

    for model in models:
        print(model,scores[model])

def compare_dataset_lpips(dataset,models,use_gpu = True):
    loss_fn = lpips.LPIPS(net='alex',version=0.1)
    scores = {}
    count = 0

    # for i in tqdm(range(40,42)):
    for i, example in tqdm(enumerate(dataset)):
        # example = dataset[i]
        og_image = example["image"]

        count += 1
        for model in models:
            img_to_compare = f"./result_db/{i}/{model}.png"
            if model not in scores:
                scores[model] = 0.0
            scores[model] += lpips_2imgs(og_image,img_to_compare,use_gpu,loss_fn=loss_fn)

    for model in models:
        scores[model] /= count

    return scores
    for model in models:
        print(model,scores[model])

def compare_dataset(dataset,models,use_gpu):
    clip_scores = compare_dataset_clip(dataset,models)
    lpips_scores = compare_dataset_lpips(dataset,models,use_gpu)
    ssim_scores = compare_dataset_ssim(dataset,models)

    print("lpips results:")
    for model in models:
        print(model,lpips_scores[model].item())

    print()

    print("ssim results:")
    for model in models:
        print(model,ssim_scores[model].item())

    print("clip results:")
    for model in models:
        print(model,clip_scores[model].item())


models = {"basic ddpm", "improved repaint with and blur average on noise sampling",
          "improved repaint with average over noise sampling", "improved repaint with blur", "improved repaint"}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model on a dataset.")
    parser.add_argument("--dataset", type=str, default="coco_200_masks", help="Name of dataset")
    parser.add_argument("--no-gpu",action='store_true', help="Use gpu")
    args = parser.parse_args()

    if(args.no_gpu):
        print("not using gpu")

    dataset = load_from_disk(args.dataset)
    compare_dataset(dataset,models,not args.no_gpu)


# dataset = load_from_disk("coco_200_masks")
# models = {"basic ddpm", "improved repaint with and blur average on noise sampling",
#           "improved repaint with average over noise sampling", "improved repaint with blur", "improved repaint"}
# scores = {}
# count = 0

# # for i, example in tqdm(enumerate(dataset)):
# for i in tqdm(range(40,41)):
#     example = dataset[i]
#     og_image = example["image"]
    
#     for model in models:
#         count += 1
#         img_to_compare = f"./result_db/{i}/{model}.png"
#         if model not in scores:
#             scores[model] = 0.0
#         scores[model] += lpips_2imgs(og_image,img_to_compare,use_gpu=False)

# for model in models:
#     scores[model] /= count

# for model in models:
#     print(model,scores[model].item())

