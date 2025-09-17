# Predicting Cancer Risk from Disease Trajectories

This repository analyzes UK Biobank data to predict cancer risk based on patient disease trajectories. Originally forked from [CancerRiskNet](https://github.com/BrunakSanderLabs/CancerRiskNet), it expands on the original work by introducing new model architectures and experiments, including the Multi-modal Transformer (MMTransformer) for integrating both categorical and numerical features.


## Directory Overview

- **dataset/**: Data loading, preprocessing, and PyTorch dataset logic for patient trajectories.
- **models/**: Model architectures (MLP, Transformer, Multi-modal Transformer), pooling layers, training utilities, and evaluation code.
- **test_runs/**: Scripts for aggregating and renaming results from multiple experiment runs.
- **config_utils.py, build_config.py, consts_and_config.py**: Configuration management, argument parsing, and core constants/enums.
- **main.py**: Main entry point for training, evaluation, grid search, bulk validation, and bootstrapping.


## Example Usage

To see all available command-line options, run:

```bash
python main.py --help
```

### Example Commands

#### 1. Grid Search
Run a grid search over model hyperparameters:
```bash
python main.py --model-type transformer --dataset-name <DATASET_NAME> --device-name gpu --grid-search
```

#### 2. Training and Validation
Train and validate a Transformer model:
```bash
python main.py --run-name train_and_validate_transformer --target-token 810 --num-epochs 100 --dropout 0 --weight-decay 0.001 --time-embed-dim 128 --hidden-dim 256 --num-layers 1 --num-heads 16 --model-type transformer --dataset-name <DATASET_NAME> --device-name gpu --train --val
```

#### 3. Bootstrap Test
Run bootstrapping on a trained model (requires a config.json file in the model directory):
```bash
python main.py --run-name run_bootstrap_on_trained_file --num-epochs 100 --target-token 514 --dropout 0 --weight-decay 0 --hidden-dim 128 --num-layers 8 --model-type mlp --dataset-name <DATASET_NAME> --device-name gpu --bootstrap-test --model-to-load-dir <PREVIOUS_TRAINING_OUT_DIR>/log --base-output-dir bootstrap_test_runs
```


### Data Format

Each dataset is a JSON object where each key is a patient ID, and the value contains patient metadata and a list of medical events. Below is an example entry:

```json
{
    "0": {
        "birth_date": "1988-07-15",
        "split_group": "train",
        "attendance_date": "2000-06-24",
        "death_date": null,
        "events": [
            {
                "type": "ICD10",
                "codes": "B94",
                "diagdate": "2000-06-24"
            },
            {
                "type": "ICD10",
                "codes": "I38",
                "diagdate": "2002-06-02"
            },
            {
                "type": "ICD10",
                "codes": "L400",
                "diagdate": "2003-12-02"
            },
            {
                "type": "ICD10",
                "codes": "F442",
                "diagdate": "2013-01-20"
            },
            {
                "type": "ICD10",
                "codes": "Q12",
                "diagdate": "2018-07-07"
            },
            {
                "type": "ICD10",
                "codes": "U490",
                "diagdate": "2020-07-24"
            }
        ]
    },
    "1": {
        // ...additional patients...
    }
}
```