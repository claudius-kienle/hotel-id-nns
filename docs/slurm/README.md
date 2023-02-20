This directory contains all scripts to execute jobs on the [BWUniCluster 2.0](https://wiki.bwhpc.de/e/BwUniCluster2.0)

- alloc.sh: Interactive job for debuggin
- create_h5.sh: to create the `.h5` datasets

Train models
---

All scripts require two parameters: 
- the `.json` configuration file to use
- the model name to use (this overwrites `model_name` of the configuration file) e.g. ResNet50 or TripletNet

- train_chain_id_tripet.sh: train chain id feature extractor with triplet learning
- train_chain_id.sh: train chain id classification
- train_hotel_id_triplet.sh: train hotel id feature extractor with triplet learning
- train_hotel_id.sh: train hotel id classification

Run sweep
---

- train_sweep.sh: run sweep