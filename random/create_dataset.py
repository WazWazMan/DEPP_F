from datasets import load_dataset
from datasets import Dataset

# 1. Load both datasets in streaming mode (test split only)
pipe_stream = load_dataset('paint-by-inpaint/PIPE', split='test', streaming=True)
mask_stream = load_dataset('paint-by-inpaint/PIPE_Masks', split='test', streaming=True)

paired_data = []

# 2. Zip them together and verify the connection on-the-fly
for img_data, mask_data in zip(pipe_stream.take(300), mask_stream.take(300)):
    
    # 3. THE VERIFICATION STEP: 
    # If these IDs don't match, Python will immediately throw an error and stop.
    assert img_data['img_id'] == mask_data['img_id'], f"Mismatch on img_id!"
    assert img_data['ann_id'] == mask_data['ann_id'], f"Mismatch on ann_id!"
    
    # 4. If they match, safely combine them into your final list
    paired_data.append({
        "source_img": img_data["source_img"],
        "target_img": img_data["target_img"],
        "mask": mask_data["mask"],
        # Grabbing one of the instructions as an example
        "instruction": img_data["Instruction_Class"], 
        "img_id": img_data["img_id"],
        "ann_id": img_data["ann_id"]
    })

print(f"Successfully loaded and verified {len(paired_data)} connected pairs.")

my_local_dataset = Dataset.from_list(paired_data)

# 2. Save the entire dataset to a folder on your computer
my_local_dataset.save_to_disk("saved_pipe_300")

print("Dataset saved successfully!")