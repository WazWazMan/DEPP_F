# Diffusion Inpainting Implementation and Improvement

## environment

In order to run the code you must first create a conda enviroment using the provided `environment.yml` file

using mamba:

```
conda env create -f ./code/environment.yml -n cs236781-hw
```

using conda:
```
$ conda env create -f ./code/environment.yml -n cs236781-hw
```

## dataset

To generate the dataset, you will need to download the COCO 2017 annotations.
1. download the [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) from coco website

2. Extract the zip into the project root directory

3. run the following command:

`python ./code/create_dataset.py `

this will create a new folder in the root directory which will contain the dataset

## running our implementation

In order to run our models on the dataset run the following command:

`python ./code/run_ds.py`

2 flags can be added:

`--start` - The starting index for the image processing(default 0)

`--count` - The total number of images to process(default 200)


## evaluating the results

To evaluate the model's performance and calculate the final metrics, run the comparison script:

`python ./code/compare_results/compare_results.py`

This script will calculate the relevant evaluation scores and output them directly to the terminal.