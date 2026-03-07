# Diffusion Inpainting Implementation and Improvement

## environment

In order to run the code you need to create a conda enviroment using the provided `environment.yml` file

using mamba:

```
conda env create -f environment.yml -n cs236781-hw
```

using conda:
```
$ conda env create -f environment.yml -n cs236781-hw
```

## dataset

we have provided the dataset we used inside the `coco_200_masks` folder

which can be created by downloading the [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) from coco website  and extracting them inside the root folder and running the following command:

`python ./code/dataset/create_dataset.py `

## running the impainting


## running the evaluation