import argparse
import json
from pathlib import Path
import random
import string
import datetime
import torch
from typing import Optional, Union
from consts_and_config import CATEGORICAL_TYPES, GROUP_SPLITS, SUPPORTED_MODELS, PAD_TOKEN, ROOT_DIR, UNK_TOKEN, Config


# config = Config(vocab, max_events_length=max_events_length, pad_size=max_events_length,
    # #                 time_embed_dim=12, hidden_dim=24, n_trajectories_per_patient_in_test=10,
    # #                 min_trajectory_length=3, num_workers=0)
def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for Config parameters.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """

    parser = argparse.ArgumentParser(description='Healthcare ML Configuration Parser')

    # What main steps to execute
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--val', action='store_true', default=False, help='Whether or not to run model on val set')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    parser.add_argument('--grid-search', action='store_true', default=False, help='Whether or not to run grid search')
    parser.add_argument('--bulk-val', action='store_true', default=False, help='Whether or not to run bulk validation')
    parser.add_argument('--bootstrap-test', action='store_true', default=False,
                        help='Enable bootstrap testing (default: False)')
    parser.add_argument('--subsample-ratio', type=float, default=None,
                        help='Subsample ratio for bootstrap testing (default: None)')

    # Device and basic configuration
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name of the run (default: None - set to a random 8 string)')
    parser.add_argument('--device-name', type=str, default='cpu',
                        help='Device name (default: cpu)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--base-output-dir', type=str, default=ROOT_DIR / "runs",
                        help='Base output directory (default: runs)')
    parser.add_argument('--data-dir', type=str, default=ROOT_DIR / "data",
                        help='Data directory (default: data)')
    parser.add_argument('--result-dir-for-val', type=str, default=ROOT_DIR / "runs",
                        help='Directory containing results for bulk validation (default: runs)')
    parser.add_argument('--model-to-load-dir', type=str, default=None,
                        help='Path to directory containing model to evaluate when train=False (default: None)')
    parser.add_argument('--model-to-load-name', type=str, default="val_auc_36_best_model.pt",
                        help='Filename of the model to load from the model directory (default: "val_auc_36_best_model.pt")')
    parser.add_argument('--grid-search-params', type=str, default=ROOT_DIR / "params.json",
                        help='Path to JSON containing grid search parameters (default: params.json)')
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='Dataset name (required)')
    parser.add_argument('--target-token', type=str, default='642',
                    help='Target token for prediction (default: "642")')
    parser.add_argument('--model-type', type=str, default=SUPPORTED_MODELS.MLP.value, choices=[m.value for m in SUPPORTED_MODELS], help=f"Model architecture to use (default: {SUPPORTED_MODELS.MLP.value})")

    # Data configuration
    parser.add_argument('--min-trajectory-length', type=int, default=5,
                        help='Minimum trajectory length (default: 5)')
    parser.add_argument('--min-followup-for-ctrls-mnths', type=int, default=24,
                        help='Minimum followup for controls in months (default: 24)')
    parser.add_argument('--exclusion-interval-mnths', type=int, default=0,
                        help='Exclusion interval in months (default: 0)')
    parser.add_argument('--trajectory-step-by-date', action='store_true', default=False,
                        help='Step trajectories by date (default: False)')
    
    # Model configuration
    parser.add_argument('--time-embed-dim', type=int, default=12,  # was 128
                        help='Time embedding dimension (default: 12)')
    parser.add_argument('--max-events-length', type=int, default=None,
                        help='Maximum events length (default: None - automatically set to longest trajectory in training dataset)')
    parser.add_argument('--pad-size', type=int, default=None,
                        help='Pad size for sequences (default: None - automatically set to max_events_length)')
    parser.add_argument('--hidden-dim', type=int, default=32,  # was 256
                        help='Hidden dimension (default: 32)')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (default: 0)')
    parser.add_argument('--pool-name', type=str, default='GlobalAvgPool',
                        help='Pooling layer name (default: GlobalAvgPool)')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of layers (default: 1)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads (default: 8, required for Transformer)')
    parser.add_argument('--threshold-method', type=str, default='f1', choices=['f1', 'rr', 'const'],
                        help="Method to use for classification threshold: 'f1', 'rr', or 'const'. If 'const', uses class_pred_threshold.")
    parser.add_argument('--class-pred-threshold', type=float, default=0.5,
                        help="Classification threshold to use if --threshold-method is 'const' (default: 0.5)")

    # Embedding configuration
    parser.add_argument('--no-time-embed', action='store_false', dest='use_time_embed', default=True,
                        help='Do not use time embedding in the transformer model (default: True)')
    parser.add_argument('--no-age-embed', action='store_false', dest='use_age_embed', default=True,
                        help='Do not use age embedding in the transformer model (default: True)')

    # Training configuration
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='Number of epochs (default: 20)')
    parser.add_argument('--train-batch-size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--eval-batch-size', type=int, default=16,
                        help='Evaluation batch size (default: 16)')
    parser.add_argument('--num-workers', type=int, default=0,  # was 8
                        help='Number of workers for data loader (default: 0)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay for optimizer (default: 0.0)')
    
    # Testing and evaluation configuration
    parser.add_argument('--n-trajectories-per-patient-in-test', type=int, default=10,
                        help='Number of trajectories per patient in test (default: 10)')
    parser.add_argument('--n-batches', type=int, default=1000,
                        help='Number of batches (default: 1000)')
    parser.add_argument('--n-train-batches-per-eval', type=int, default=100,
                        help='Number of training batches per evaluation (default: 100)')
    parser.add_argument('--n-batches-for-eval', type=int, default=200,
                        help='Number of batches from the validation set to use for evaluation (default: 200)')
    parser.add_argument('--resume-epoch', type=int, default=0,
                        help='Resume from epoch (default: 0)')
    
    return parser


def create_vocab(word_list):
    """
    Creates a vocabulary dictionary from a list of words.

    Args:
        word_list (list): List of words.

    Returns:
        dict: Mapping from word to index.
    """
    vocab_list = [PAD_TOKEN, UNK_TOKEN] + sorted(list(set(word_list)))
    return {v: k for k, v in enumerate(vocab_list)}


def get_config_from_data(data, req_split_group: GROUP_SPLITS=GROUP_SPLITS.TRAIN):
    """
    Extracts configuration parameters from the dataset.

    Args:
        data (dict): Patient data.
        req_split_group (GROUP_SPLITS): Required split group.

    Returns:
        tuple: (max_events_length, vocab, token_types)
    """
    event_lens = []
    vocab = []
    token_types = []
    for patient_data in data.values():
        patient_split_group = patient_data.get('split_group', GROUP_SPLITS.TEST.value)

        if req_split_group == GROUP_SPLITS.ALL or patient_split_group == req_split_group.value:
            event_lens.append(len(patient_data['events']))
            vocab.extend([event['codes'] for event in patient_data['events'] if event['type'] in CATEGORICAL_TYPES])
            token_types.extend([event['type'] for event in patient_data['events']])

    return max(event_lens) if event_lens else 0, create_vocab(vocab), create_vocab(token_types)


def get_device(device_name):
    """
    Returns the appropriate torch device based on device_name.

    Args:
        device_name (str): Name of the device.

    Returns:
        torch.device: Torch device object.
    """
    if device_name == 'gpu' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    elif device_name == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    return device



def get_data_and_config_from_cmdline() -> Config:
    """
    Main function to parse command line arguments and return a Config object.

    Returns:
        tuple: (data, Config)
    """
    # Validate num_heads for Transformer
    parser = create_argument_parser()

    args = parser.parse_args()

    # Error checking for threshold_method
    allowed_threshold_methods = {'f1', 'rr', 'const'}
    if args.threshold_method not in allowed_threshold_methods:
        parser.error(f"--threshold-method must be one of {allowed_threshold_methods} (got '{args.threshold_method}')")

    # Require model-to-load-dir if train is False
    if not (args.grid_search or args.bulk_val) and (not args.train and not args.model_to_load_dir):
        parser.error("If --train is False, --model-to-load-dir must be specified.")

    # Require result-dir-for-val if bulk-val is set
    if args.bulk_val and not args.result_dir_for_val:
        parser.error("If --bulk-val is set, --result-dir-for-val must be specified.")

    if args.model_type in [SUPPORTED_MODELS.TRANSFORMER.name, SUPPORTED_MODELS.MM_TRANSFORMER.name] \
            and args.num_heads is None:
        parser.error("--num-heads must be specified when using Transformer model.")

    with open(args.data_dir / f"{args.dataset_name}.json") as f:
        data = json.load(f)

    max_events_length, args.vocab, args.token_types = get_config_from_data(data)

    if args.max_events_length is None:
        args.max_events_length = max_events_length

    if args.pad_size is None:
        args.pad_size = args.max_events_length

    args.device = get_device(args.device_name)

    if args.run_name is None:
        args.run_name = args.dataset_name[:10] + ''.join(random.choices(string.ascii_letters + string.digits, k=4))

    args.run_dir = Path(args.base_output_dir) / args.run_name
    delattr(args, "base_output_dir")
    args.start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Force trajectory_step_by_date True if mm_transformer is selected
    if args.model_type == SUPPORTED_MODELS.MM_TRANSFORMER.value:
        args.trajectory_step_by_date = True

    # Handle bootstrap-test logic
    if args.bootstrap_test:
        args.test = True
        if args.subsample_ratio is None:
            args.subsample_ratio = 0.8

    return data, Config(**vars(args))
