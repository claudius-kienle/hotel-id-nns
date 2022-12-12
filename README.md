Development
===

0. Install Mamba

    Mamba is a way faster C++ implementation of the prominent package manager conda.
    It can be installed with
    ```bash
    > curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    > bash Mambaforge-$(uname)-$(uname -m).sh
    ```
    *Note: Go to https://mamba.readthedocs.io/en/latest/installation.html for further details*

1.  Setup conda environment 

    ```bash
    > mamba env create -f environment.yaml
    > mamba activate hotel-id-nns
    ```

2. Install `hotel-id-nns` package

    For development, this repo should be installed as pip editable install. This requires pip version 22.*

    ```bash
    > pip install -e . --config-settings editable_mode=compat --no-deps
    ```

    *Note: the argument `config-settings editable_mode=compat` enables the linter to work correctly.*


Setup Dataset
===

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

Get started with training
===

The loss functions used during training require weights for each class, since the dataset is unbalanced. These are located in data/dataset.

One can look at the .json configs in data/config\_files for inspiration how training works.

The training can be executed with 
```bash
python hotel_id_nns/scripts/train_chain_id.py data/configs/train_chain_id_ce.json
```

The optional parameter `--data-path` can be used to specify where the dataset files and folders are stored at.
If not given, the script asumes that the files are located under data/dataset.

Chain ID Prediction Roadmap
------------------------

- Use Binary Cross Entropy Loss with weights since the dataset is unbalanced
- Don't train chain_id == 0, since this class contains all samples where the chain_id isn't known
- -> Still, the network does not predict the chain_ids precisely, but always predicts the same chain_id,
    therefore, the next approach is to investigate the network and only use one sample for each chain_id.
    This should directly overfit, 
- Trained with one sample per class. Only converged with ResNet18, lr 1e-2, bs 32, val equals test set
- Trained on whole dataset, but val_dataset equals still train_dataset: Model overfits with f1 ~0.85 (checkpoint: 1670591014.047502)
- Also overfits, if val_dataset different, val_loss directly increases -> try adam with weight-decay 2e-5
  - -> therefore, now sure that model able to train on dataset. Only correct hp remain unknown to avoid overfitting
- started bayesian hyperparameter search with wandb sweeps to find best hyperparameters
  - best hp are: ResNet50, WeightDecay 1e-2, lr 5e-3, sgd, rather short lr-patience
    - results in {'accuracy': 0.359375, 'precision': 0.4362400472164154, 'recall': 0.3991617262363434, 'f1': 0.39577633142471313}
- traning one ResNet50 and one ResNet152 model with best hp: assumption that larger ResNet will generalize better