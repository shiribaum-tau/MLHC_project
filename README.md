# Predicting Risk of Cancer From Disease Trajectories

This repository works with UK Biobank data to predict cancer risk from patient disease trajectories. It began as a fork of [CancerRiskNet](https://github.com/BrunakSanderLabs/CancerRiskNet), but introduces new model architectures and experiments, including the Multi-modal Transformer (MMTransformer) for handling both categorical and numerical features.

## Directory Overview

- **dataset/**  
  Data loading, preprocessing, and PyTorch dataset logic for patient trajectories.

- **models/**  
  Model architectures (MLP, Transformer, Multi-modal Transformer), pooling layers, training utilities, and evaluation code.

- **test_runs/**  
  Scripts for aggregating and renaming results from multiple experiment runs.

- **config_utils.py, build_config.py, consts_and_config.py**  
  Configuration management, argument parsing, and core constants/enums.

- **main.py**  
  Main entry point for training, evaluation, grid search, bulk validation, and bootstrapping.

