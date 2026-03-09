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
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms

from torchmetrics.image.kid import KernelInceptionDistance

def compare_dataset_kid(dataset, models, device="cuda" if torch.cuda.is_available() else "cpu"):
    kid_scores = {}
    
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte())
    ])

    for model_name in models:
        # subset_size קובע כמה תמונות נדגמות בכל פעם לחישוב ה-Kernel
        kid = KernelInceptionDistance(subset_size=50).to(device)
        
        for i, example in enumerate(dataset):
            real_img = preprocess(example["image"].convert("RGB")).unsqueeze(0).to(device)
            
            gen_img_path = f"./result_db/{i}/{model_name}.png"
            gen_img = Image.open(gen_img_path).convert("RGB")
            gen_img_t = preprocess(gen_img).unsqueeze(0).to(device)

            kid.update(real_img, real=True)
            kid.update(gen_img_t, real=False)

        # KID מחזיר ממוצע וסטיית תקן, אנחנו ניקח את הממוצע (mean)
        mean_kid, std_kid = kid.compute()
        kid_scores[model_name] = mean_kid
        
    return kid_scores

def compare_dataset_fid(dataset, models, device="cuda" if torch.cuda.is_available() else "cpu"):
    fid_scores = {}
    
    # טרנספורמציה בסיסית: FID דורש גודל מסוים (לרוב 299x299) וערכים של uint8
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte())
    ])

    for model_name in models:
        # אתחול המטריקה עבור כל מודל
        fid = FrechetInceptionDistance(feature=2048).to(device)
        
        real_imgs = []
        gen_imgs = []

        for i, example in enumerate(dataset):
            # תמונה אמיתית מהדאטהסט
            real_img = example["image"].convert("RGB")
            # תמונה שנוצרה
            gen_img_path = f"./result_db/{i}/{model_name}.png"
            gen_img = Image.open(gen_img_path).convert("RGB")

            real_imgs.append(preprocess(real_img))
            gen_imgs.append(preprocess(gen_img))

        # המרה ל-Tensors ב-Batch אחד (אפשר לחלק ל-batches קטנים אם הזיכרון מוגבל)
        real_imgs = torch.stack(real_imgs).to(device)
        gen_imgs = torch.stack(gen_imgs).to(device)

        # עדכון המטריקה
        fid.update(real_imgs, real=True)
        fid.update(gen_imgs, real=False)

        fid_scores[model_name] = fid.compute()
        
    return fid_scores

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
                clip_score = outputs.logits_per_image
            if model_name not in scores:
                scores[model_name] = 0.0
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
    fid_scores = compare_dataset_kid(dataset, models)

    print("FID results (lower is better):")
    for model in models:
        print(f"{model}: {fid_scores[model].item():.4f}")
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

