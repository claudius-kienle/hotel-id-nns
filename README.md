# setup dataset
1. Goto [kaggle](https://www.kaggle.com/competitions/hotel-id-2021-fgvc8/data) and download the dataset zip
1. Extract it to hotel-id-nns/dataset/hotel-id-2021-fgvc8. The directory structure should look like this:
    ```
    hotel-id-nns/
    └──dataset/
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
