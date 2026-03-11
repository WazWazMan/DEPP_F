
import torch
import matplotlib.pyplot as plt


def print_mask(mask_tensor: torch.Tensor, title: str = "Mask Visualization"):
    mask = mask_tensor.detach().cpu().float()
        
    # 2. Remove Batch and Channel dimensions [1, 1, H, W] -> [H, W]
    if mask.ndim == 4:
        mask = mask.squeeze(0).squeeze(0)
    elif mask.ndim == 3:
        mask = mask.squeeze(0)
    
    # 3. Plotting
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray', vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis('off')
    plt.show()