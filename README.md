# setup dataset
1. Goto [kaggle](https://www.kaggle.com/competitions/hotel-id-2021-fgvc8/data) and download the dataset zip
1. Extract it to dataset/hotel-id-2021-fgvc8. The directory structure should look like this:
    ```
    dataset/
    └──hotel-id-2021-fgvc8/
       └──test_images/
       └──train_images/
       └──sample_submission.csv
       └──train.csv
    ```
1. Convert the dataset into a usable format using the python script tools/hotel_dataset_converter.py
    ```bash
    python tools/hotel_dataset_converter.py dataset/hotel-id-2021-fgvc8/train.csv dataset/ --seed 0
    ```
1. The generated train, val and test files contain rows in the format `image-path class-idx` where class-idx is the index of the class-label which are listed in `hotel_classes_` file.

# Development

First setup the virtual conda environmnent with

```bash
conda env create -f environment.yaml
conda activate hotel-id-nns
```

Install the `hotel-id-nns` as editable install with pip to start development without any setup. This requires pip version 22.*

```bash
pip install -e . --config-settings editable_mode=strict

```