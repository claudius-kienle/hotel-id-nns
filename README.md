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
1. Extract it to data/dataset/hotel-id-2021-fgvc8. The directory structure should look like this:
    ```
    data/
    └──dataset/
        └──hotel-id-2021-fgvc8/
        └──test_images/
        └──train_images/
        └──sample_submission.csv
        └──train.csv
    ```
    **Note: all following scripts assume this file structure**


2. Convert the dataset into a usable format using the jupyter notebook at [hotel_id_nns/tools/train_val_test_split.ipynb](hotel_id_nns/tools/train_val_test_split.ipynb). Just execute each cell.


2. The generated train, val and test files (.csv) contain rows in the format `image-path,chain-idx,hotel-idx` where chain-idx is the index of the class-label which are listed in `chain_id_mapping.csv` file (Analog for the hotel-idx). 

3. To speed up training, use [hotel_id_nns/tools/csv_to_h5_converter.py](hotel_id_nns/tools/csv_to_h5_converter.py) to generate .h5 files for each dataset with the newly generated .csv files. 


Get started with training
===

The loss functions used during training require weights for each class, since the dataset is unbalanced. These are located in data/dataset.

One can look at the .json configs in data/config\_files for inspiration how training works.

The training can be executed with 

1. To train chain-id prediction network

    ```bash
    python hotel_id_nns/scripts/train_classification.py data/configs/train_chain_id.json
    ```

2. To train hotel-id prediction network

    ```bash
    python hotel_id_nns/scripts/train_classification.py data/configs/train_hotel_id.json
    ```

The optional parameter `--data-path` can be used to specify where the dataset files and folders are stored at.
If not given, the script assumes that the files are located under data/dataset.

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
- Did second sweep to check which model and if pretraining helps (on imagenet)
  - best net: pretrained on imagenet, ResNet50, no finetuning, lr ~ 0.0035
  - -> all non pretrained models performed badly, maybe lr to small

- Best Run on TorchVision (pretrained): difference
    - ResNet 50: peach-tree-201 (1673136565.3055131)
        {'accuracy': 0.4302244782447815, 'mAP@5': 0.5461959838867188, 'precision': 0.5035776495933533, 'recall': 0.4691525101661682, 'f1': 0.46258115768432617}
    - ResNet 101: atomic-cherry-203 (1673172280.4284968)
        {'accuracy': 0.41664355993270874, 'mAP@5': 0.525413453578949, 'precision': 0.4989142417907715, 'recall': 0.44573840498924255, 'f1': 0.44782912731170654}
- Best Run on Our Model (pretrained): 
    - ResNet 50 (Chr): glamorous-butterfly-205 (1673180942.4564664)
        - Test:  {'accuracy': 0.37638580799102783, 'mAP@5': 0.4941853880882263, 'precision': 0.45165151357650757, 'recall': 0.4202423095703125, 'f1': 0.4122817814350128}
        - Train: {'accuracy': 0.5951172113418579, 'mAP@5': 0.6947623491287231, 'precision': 0.7294825315475464, 'recall': 0.7660961151123047, 'f1': 0.7145957946777344}

- Triplet Net
  - # TODO


Hotel ID Prediction Roadmap
------------------------

1. Classical ResNet classification
    1. Without Regularization (weight_decay=0)
        - ResNet 50 (1673631392.28741)
            - TrainSet: {'accuracy': 0.948437511920929, 'mAP@5': 0.949999988079071, 'precision': 0.947555422782898, 'recall': 0.9483367204666138, 'f1': 0.9478158950805664}
            - TestSet: {'accuracy': 0.037278272211551666, 'mAP@5': 0.05137887969613075, 'precision': 0.03697093576192856, 'recall': 0.03735203295946121, 'f1': 0.03709796816110611}
            - Easily overfits
            - Sweep on weight decay and two models with wd 0.1 and 0.01
    2. sweep: did not train at all
    3. two models with 0.1, 0.01 also did not converge: assume wd still to high
        - new model with wd 2e-5 (torchvision default): did not converge, lr 0.1 just to high
    4. Use Adam + higher lr (not that sensitive to lr, converges faster?): converges faster but lr 0.1 to high
    5.  use Dropout Rate (0.1): nothing changed, equivalent to wd
    6.  Final Net Hotel ID 
        - wd 2e-5, lr 3.5e-3 (1674222514.1669555): 
          - Train set {'accuracy': 0.7364062666893005, 'mAP@5': 0.7716445326805115, 'precision': 0.74395751953125, 'recall': 0.7426989078521729, 'f1': 0.7425405979156494}
          - Test Set {'accuracy': 0.07147469371557236, 'mAP@5': 0.09485301375389099, 'precision': 0.07216004282236099, 'recall': 0.07179640978574753, 'f1': 0.07179736346006393}
        - wd 2e-5, lr 3.5e-3, patience 40 (1674233030.796025): 
          - Train set {'accuracy': 0.6115624904632568, 'mAP@5': 0.6524296402931213, 'precision': 0.6180585026741028, 'recall': 0.6180443167686462, 'f1': 0.6172311902046204}
          - Test set {'accuracy': 0.04306560009717941, 'mAP@5': 0.0652085542678833, 'precision': 0.042522773146629333, 'recall': 0.043607912957668304, 'f1': 0.04288448393344879}
        - wd 0.012, lr 3.5e-3 (1674181228.755098): did not converge

2.  Triplet Learning
    1.  First model had no unit circle map, no clamp, MSE Loss
        loss reduced quite a bit (1673706823.917135), but approach with mean of class has map@5=0
    2.  Two new models with MAP and cosine similarity, 'correct' loss and map to unit circle (normalization)
        - MSE loss net (1673797952.4643068) produces good looking results (~0.8 distance on unit circle)
            - Computing the mean for each hotel-id class results in metrics with cosine similarity for label probs: 
              - Train set {'accuracy': 0.09921874850988388, 'mAP@5': 0.1398177295923233, 'precision': 0.09918095171451569, 'recall': 0.09957157075405121, 'f1': 0.09931115806102753}
              - Test set {'accuracy': 0.026348039507865906, 'mAP@5': 0.0437028706073761, 'precision': 0.026427101343870163, 'recall': 0.026427101343870163, 'f1': 0.026427101343870163}
        - Cosine Similarity Loss (1673959543.7748818) also converges
            - Mean for each class:
              - Train set {'accuracy': 0.10000000149011612, 'mAP@5': 0.14813801646232605, 'precision': 0.10040322691202164, 'recall': 0.10040322691202164, 'f1': 0.10040322691202164}
              - Test set {'accuracy': 0.025876697152853012, 'mAP@5': 0.04259442910552025, 'precision': 0.02595576085150242, 'recall': 0.02595576085150242, 'f1': 0.02595576085150242}
    3. Add simple classification net on triplet-net backbone
        - MSE loss backbone
            - wd 2e-5, lr 0.1 (1674181285.901832)
              - Train set {'accuracy': 0.4006877839565277, 'mAP@5': 0.477467805147171, 'precision': 0.40475237369537354, 'recall': 0.405136376619339, 'f1': 0.4043208658695221}
              - Test set {'accuracy': 0.03125, 'mAP@5': 0.05229640007019043, 'precision': 0.03135787695646286, 'recall': 0.03099135123193264, 'f1': 0.030993277207016945}
              - -> large gap, assuming overfit, increase wd (0.012)
            - wd 0, lr 0.1 (1674181228.755045): did not converge
            - wd 2e-5, lr 3.5e-3 (1674228390.8946748): 
              - Train set {'accuracy': 0.775390625, 'mAP@5': 0.8120834231376648, 'precision': 0.782054603099823, 'recall': 0.7816636562347412, 'f1': 0.7810818552970886}
              - Test set {'accuracy': 0.041903410106897354, 'mAP@5': 0.06779119372367859, 'precision': 0.042263854295015335, 'recall': 0.041897378861904144, 'f1': 0.041900232434272766}
            - wd 2e-5, lr 3.5e-3, no finetune (1674229453.8456535):
              - Train set {'accuracy': 0.9417968988418579, 'mAP@5': 0.958759069442749, 'precision': 0.9445099830627441, 'recall': 0.9441522359848022, 'f1': 0.9437576532363892}
              - Test set {'accuracy': 0.034090910106897354, 'mAP@5': 0.054616477340459824, 'precision': 0.03422962874174118, 'recall': 0.03459326550364494, 'f1': 0.034350838512182236}
            - wd 0.012, lr 3.5e-3 (1674229525.3247921): did not converge
        - Cosine similarity loss backbone
            - wd 2e-5, lr 0.1 (1674181628.1866858)
                -   Train set {'accuracy': 0.5168589949607849, 'mAP@5': 0.5907645225524902, 'precision': 0.5221248865127563, 'recall': 0.521905779838562, 'f1': 0.5213735699653625}
                -   Test set {'accuracy': 0.04022468999028206, 'mAP@5': 0.062028661370277405, 'precision': 0.04003467410802841, 'recall': 0.040028903633356094, 'f1': 0.0397903248667717}
            - wd 0, lr 3.5e-3 (1674182383.6598442): did not converge
            - wd 2e-5, lr 3.5e-3 (1674228553.6910925):
                -   Train set {'accuracy': 0.9232812523841858, 'mAP@5': 0.9413606524467468, 'precision': 0.9284943342208862, 'recall': 0.9268926978111267, 'f1': 0.927031934261322}
                -   **Test set {'accuracy': 0.07192665338516235, 'mAP@5': 0.09925857186317444, 'precision': 0.07209797948598862, 'recall': 0.07173150032758713, 'f1': 0.07173436135053635}**
            - wd 0.012, lr 3.5e-3 (1674229865.491079): did not converge
            - wd 1e-4, lr 3.5e-3: # TODO

Run Sweep [https://docs.wandb.ai/guides/sweeps](https://docs.wandb.ai/guides/sweeps)
---

1. Initialize Sweep

```bash
wandb sweep --project ClassNet --entity hotel-id-nns data/configs/chain_id_sweep.yaml
```

2. Start Sweep Agents

    1. BwUniCluster

        Ensure the .sh file is set up correctly

        ```bash
            sbatch docs/slurm/train_sweep.sh
        ```

    2. CLI

        ```bash
        wandb agent hotel-id-nns/ClassNet/<<SWEEP-ID>>
        ```