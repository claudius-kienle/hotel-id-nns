import argparse
import pathlib
import os
import random
import math
import csv


def parse_args():
    parser = argparse.ArgumentParser(description='Converter to convert the hotel dataset from the kaggle competition')

    parser.add_argument('annotation_file', type=pathlib.Path, help='Path to the annotation file')
    parser.add_argument('output_dir', type=pathlib.Path, help='Directory where the output files should be generated')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--class-name', type=str, default="chain", help="Select the type of label for the images")

    return parser.parse_args()


class HotelDataset:
    def __init__(self):
        self.annotations = []
        self.classes = None

    @classmethod
    def from_annotation_file(cls, annotation_file: str, classes=None):
        dataset = HotelDataset()

        keys = []
        with open(annotation_file.absolute(), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

            keys = next(reader)[0].split(",")
            for row in reader:
                dataset.annotations.append({key: value for key, value in zip(keys, row[0].split(","))})

        if classes is None:
            classes = {key: dataset.get_column(key) for key in keys}
        dataset.classes = classes

        print(len(dataset.annotations), "annotations in total")
        print("Annotation examples:")
        print(*dataset.annotations[:5], sep="\n")

        return dataset

    @classmethod
    def from_rows(cls, rows: list, classes):
        dataset = HotelDataset()
        dataset.annotations = rows
        dataset.classes = classes
        return dataset

    def get_column(self, name, map_type=str):
        """
        returns a sorted list of unique values in the specified column
        """
        col = set()
        for row in self.annotations:
            col.add(map_type(row[name]))

        col_list = list(col)
        col_list.sort()
        return col_list

    def count_images_per_class(self, class_name: str):
        classes = self.get_column(class_name)
        image_counter = {c: 0 for c in classes}

        for row in self.annotations:
            image_counter[row[class_name]] += 1

        return image_counter

    def get_images_per_class(self, class_name: str):
        classes = self.get_column(class_name)
        image_paths = {c: [] for c in classes}

        for row in self.annotations:
            image_paths[row[class_name]].append(row["image"])

        return image_paths

    def split_images_by_ratio(self, ratio: float, class_name: str, shuffle=False, seed=0):
        """
        parameters:
         - ratio between 0.0 and 1.0
         - column identifying the column which contains the class of the images
        returns:
         - a split of the dataset. Dataset 1 contains (ratio * #images-per-<class_name>) images per class_name
           and dataset 2 contains ((1.0-ratio) * #images-per-<class_name>) images per class_name.
        """
        image_paths = self.get_images_per_class(class_name)

        classes = self.get_column(class_name)
        split_1 = {c: [] for c in classes}
        split_2 = {c: [] for c in classes}

        for c, images in image_paths.items():
            if shuffle:
                random.Random(seed).shuffle(images)

            split_index = math.ceil(len(images) * ratio)
            split_1[c] = images[:split_index]
            split_2[c] = images[split_index:]

        rows_1 = []
        rows_2 = []

        for c, images in split_1.items():
            for image in images:
                rows_1.append({"image": image, class_name: c})

        for c, images in split_2.items():
            for image in images:
                rows_2.append({"image": image, class_name: c})

        classes = {class_name: self.classes[class_name]}
        return HotelDataset.from_rows(rows_1, classes), HotelDataset.from_rows(rows_2, classes)

    def shuffle_rows(self, seed):
        random.Random(seed).shuffle(self.annotations)

    def save_annotations_to_file(self, output_dir, class_name, split_name):
        output_file = os.path.join(output_dir, "hotel_" + split_name + "_" + class_name + ".csv")
        print("Write annotation file to", output_file)

        if os.path.isfile(output_file):
            os.remove(output_file)

        name_to_index = {name: idx for idx, name in enumerate(self.classes[class_name])}

        # write rows to file in the following format:
        # <chain>/<image-name> <class-index>
        with open(output_file, "a") as outfile:
            for row in self.annotations:
                name = row[class_name]
                idx = name_to_index[name]
                image_path = os.path.join(row["chain"], row["image"])
                out = image_path + " " + str(idx) + "\n"
                outfile.write(out)

    def save_classes_to_file(self, output_dir, class_name):
        output_file = os.path.join(output_dir, "hotel_classes_" + class_name + ".txt")
        out = "classes=(" + (', '.join(tuple(self.classes[class_name]), )) + ")"

        with open(output_file, "w") as outfile:
            outfile.write(out)


if __name__ == "__main__":
    """
    mmclassification: 
     - data_prefix: <path-to-dataset>/hotel-id-2021-fgvc8/train_images
     - ann_file: e.g. <path-to-dataset>/val.txt (references all images in the validation split)
    """
    args = parse_args()

    if not os.path.isfile(args.annotation_file.absolute()):
        raise ValueError("The variable 'annotation_file' must point to a csv-file")

    output_dir = args.output_dir.absolute()
    if not os.path.isdir(output_dir):
        raise ValueError("The variable 'output_dir' must point to a directory")

    class_name = args.class_name

    seed = args.seed
    if seed is None:
        seed = random.random()

    dataset = HotelDataset.from_annotation_file(args.annotation_file)


    # TODO: parameterize
    train, set2 = dataset.split_images_by_ratio(0.9, class_name, seed=seed)
    val, test = set2.split_images_by_ratio(2/3, class_name, seed=seed)

    train.shuffle_rows(seed)
    train.save_annotations_to_file(output_dir, class_name, "train")

    val.shuffle_rows(seed)
    val.save_annotations_to_file(output_dir, class_name, "val")

    test.shuffle_rows(seed)
    test.save_annotations_to_file(output_dir, class_name, "test")

    dataset.save_classes_to_file(output_dir, class_name)

