def compare_dataset_kid(dataset, models, device):
    kid_scores = {}
    
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte())
    ])

    for model_name in models:
        kid = KernelInceptionDistance(subset_size=50).to(device)
        
        for i, example in enumerate(dataset):
            real_img = preprocess(example["image"].convert("RGB")).unsqueeze(0).to(device)
            
            gen_img_path = f"./result_db/{i}/{model_name}.png"
            gen_img = Image.open(gen_img_path).convert("RGB")
            gen_img_t = preprocess(gen_img).unsqueeze(0).to(device)

            kid.update(real_img, real=True)
            kid.update(gen_img_t, real=False)

        mean_kid, std_kid = kid.compute()
        kid_scores[model_name] = mean_kid
        
    return kid_scores