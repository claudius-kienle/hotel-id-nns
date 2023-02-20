This folder contains all configuration files to train the different models:

Sweeps
---
- chain_id_sweep.yaml: contains sweep configuration to run a WanDB sweep on chain_id
- hotel_id_sweep.yaml: contains sweep configuration to run a WanDB sweep on hotel_id

Chain ID Classification
---
- train_chain_id_triplet_cosine.json: train feature extraction network with triplet learning for chain-id
- train_chain_id.json: train chain_id classification network


Hotel ID Classification
---

- train_hotel_id_triplet_cosine.json: train hotel id feature extraction network with triplet learning
- train_hotel_id-triplet-classification-cosine.json: train hotel-id classification network with feature extraction as backbone
- train_hotel_id.json: train hotel id classification network with standard classification model (e.g. ResNet50)