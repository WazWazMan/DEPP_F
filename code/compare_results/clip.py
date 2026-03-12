from transformers import CLIPProcessor, CLIPModel
import torch
from tqdm import tqdm
from PIL import Image


def compare_dataset_clip(dataset, models, device):
    scores = {}
    count = 0
    
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    
    for i, example in tqdm(enumerate(dataset)):
        prompt = example["description"] 
        count += 1

        for model_name in models:
            img_to_compare = Image.open(f"./result_db/{i}/{model_name}.png").convert("RGB")

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