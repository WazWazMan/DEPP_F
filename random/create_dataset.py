# need to download annotation from:
# http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# and extract into root folder before running

import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from pycocotools.coco import COCO
from datasets import Dataset, Features, Image as HFImage, Value

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument("--start", type=int, default=0, help="Start index of first sample")
    parser.add_argument("--count", type=int, default=200, help="count of samples to use")
    parser.add_argument("--name", type=str, default="coco_200_masks2", help="name of dataset")
    args = parser.parse_args()


    # 1. Initialize COCO API for instance annotations and captions
    # Update these paths to where you extracted the annotations folder
    dataDir = '.'
    dataType = 'val2017'
    instances_annFile = f'{dataDir}/annotations/instances_{dataType}.json'
    captions_annFile = f'{dataDir}/annotations/captions_{dataType}.json'

    print("Loading annotations...")
    coco_instances = COCO(instances_annFile)
    coco_captions = COCO(captions_annFile)

    # 2. Get 200 image IDs that have both annotations and captions
    img_ids = coco_instances.getImgIds()
    start = args.start
    end = args.start + args.count
    selected_img_ids = img_ids[start:end]

    def generate_dataset():
        for img_id in selected_img_ids:
            # Get image metadata
            img_info = coco_instances.loadImgs(img_id)[0]
            
            # Fetch the actual image from the COCO URL
            response = requests.get(img_info['coco_url'])
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Get captions and pick the first one
            ann_ids_caps = coco_captions.getAnnIds(imgIds=img_id)
            anns_caps = coco_captions.loadAnns(ann_ids_caps)
            description = anns_caps[0]['caption'] if anns_caps else "No description available."
            
            # Get instance annotations to build the mask
            ann_ids_inst = coco_instances.getAnnIds(imgIds=img_id)
            anns_inst = coco_instances.loadAnns(ann_ids_inst)
            
            # Create a blank mask
            mask_array = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            
            # Merge all instance masks into one binary mask
            for ann in anns_inst:
                # pycocotools generates a binary mask for each annotation
                mask_array = np.maximum(mask_array, coco_instances.annToMask(ann))
            
            # Convert the mask array (0s and 1s) to a standard PIL image (0 and 255)
            mask_image = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")
            
            yield {
                "image": image,
                "mask": mask_image,
                "description": description
            }

    # 3. Define the Hugging Face Dataset features
    features = Features({
        "image": HFImage(),
        "mask": HFImage(),
        "description": Value("string")
    })

    # 4. Create the Hugging Face Dataset from the generator
    print("Building Hugging Face Dataset...")
    hf_dataset = Dataset.from_generator(generate_dataset, features=features)

    print(f"Successfully created dataset with {len(hf_dataset)} items!")
    print(hf_dataset[0]) 

    hf_dataset.save_to_disk(args.name)