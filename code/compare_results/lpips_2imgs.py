import lpips
import cv2
import numpy as np
import torch

def lpips_2imgs(image:str,path1:str, use_gpu = True,version = 0.1,loss_fn = None):
    if not loss_fn:
        loss_fn = lpips.LPIPS(net='alex',version=version)

    if(use_gpu):
        loss_fn.cuda()

    # Load images
    # image0 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)[:,:,::-1]
    # image0 = cv2.resize(image0, (512,512))

    # we do this because we used it in the impainting proccess
    image0 = np.array(image.convert('RGB').resize((512, 512)))
    img0 = lpips.im2tensor(image0) # RGB image from [-1,1]
    img1 = lpips.im2tensor(lpips.load_image(path1))

    if(use_gpu):
        img0 = img0.cuda()
        img1 = img1.cuda()

    # Compute distance
    with torch.no_grad():
        dist01 = loss_fn.forward(img0, img1)
    return dist01
