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

# Access your verified data like this:
# first_pair = paired_data[0]
# print(f"Image ID: {first_pair['img_id']}, Annotation ID: {first_pair['ann_id']}")

# import matplotlib.pyplot as plt

# # 1. Extract the images and instruction from your first_pair
# source_image = first_pair['source_img']
# mask_image = first_pair['mask']
# target_image = first_pair['target_img']
# instruction = first_pair['instruction']

# # 2. Set up a matplotlib figure with 3 subplots in a single row
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # 3. Plot the Source Image (The image before the object is added)
# axes[0].imshow(source_image)
# axes[0].set_title("Source Image\n(Object Removed)")
# axes[0].axis('off')

# # 4. Plot the Mask (Showing exactly where the object goes)
# # Using cmap='gray' because masks are typically single-channel grayscale
# axes[1].imshow(mask_image, cmap='gray')
# axes[1].set_title("Mask")
# axes[1].axis('off')

# # 5. Plot the Target Image (The ground truth with the object)
# axes[2].imshow(target_image)
# axes[2].set_title(f"Target Image\n({instruction})")
# axes[2].axis('off')

# # 6. Display everything cleanly
# plt.tight_layout()
# plt.show()


my_local_dataset = Dataset.from_list(paired_data)

# 2. Save the entire dataset to a folder on your computer
my_local_dataset.save_to_disk("saved_pipe_300")

print("Dataset saved successfully!")