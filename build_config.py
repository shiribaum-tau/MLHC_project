import argparse
import json
from pathlib import Path
import random
import string
import torch
from typing import Optional, Union
from consts_and_config import GROUP_SPLITS, PAD_TOKEN, ROOT_DIR, UNK_TOKEN, Config


# config = Config(vocab, max_events_length=max_events_length, pad_size=max_events_length,
    # #                 time_embed_dim=12, hidden_dim=24, n_trajectories_per_patient_in_test=10,
    # #                 min_trajectory_length=3, num_workers=0)
def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for Config parameters."""
    parser = argparse.ArgumentParser(description='Healthcare ML Configuration Parser')

    # Device and basic configuration
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name of the run (default: None - set to a random 8 string)')
    parser.add_argument('--device-name', type=str, default='cpu',
                        help='Device name (default: cpu)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--out-dir', type=str, default=ROOT_DIR / "out",
                        help='Output directory (default: out)')
    parser.add_argument('--log-dir', type=str, default=ROOT_DIR / "logs",
                        help='Log directory (default: log)')
    parser.add_argument('--data-dir', type=str, default=ROOT_DIR / "data",
                        help='Data directory (default: data)')
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='Dataset name (required)')
    parser.add_argument('--start-at-attendance', action='store_true', default=True,
                        help='Always include the attendance date in the trajectory (default: True)')
    parser.add_argument('--no-start-at-attendance', dest='start_at_attendance', action='store_false',
                        help='Do not include the attendance date in the trajectory')
    
    # Data configuration
    parser.add_argument('--min-trajectory-length', type=int, default=5,
                        help='Minimum trajectory length (default: 5)')
    parser.add_argument('--min-followup-for-ctrls-mnths', type=int, default=24,
                        help='Minimum followup for controls in months (default: 24)')
    parser.add_argument('--exclusion-interval-mnths', type=int, default=0,
                        help='Exclusion interval in months (default: 0)')
    
    # Model configuration
    parser.add_argument('--time-embed-dim', type=int, default=12,  # was 128
                        help='Time embedding dimension (default: 12)')
    parser.add_argument('--max-events-length', type=int, default=None,
                        help='Maximum events length (default: None - automatically set to longest trajectory in training dataset)')
    parser.add_argument('--pad-size', type=int, default=None,
                        help='Pad size for sequences (default: None - automatically set to max_events_length)')
    parser.add_argument('--hidden-dim', type=int, default=24,  # was 256
                        help='Hidden dimension (default: 24)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--pool-name', type=str, default='GlobalAvgPool',
                        help='Pooling layer name (default: GlobalAvgPool)')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of layers (default: 1)')
    
    # Training configuration
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='Number of epochs (default: 20)')
    parser.add_argument('--train-batch-size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--eval-batch-size', type=int, default=16,
                        help='Evaluation batch size (default: 16)')
    parser.add_argument('--num-workers', type=int, default=8,  # was 8
                        help='Number of workers for data loader (default: 8)')
    
    # Testing and evaluation configuration
    parser.add_argument('--n-trajectories-per-patient-in-test', type=int, default=10,
                        help='Number of trajectories per patient in test (default: 10)')
    parser.add_argument('--n-batches', type=int, default=1000,
                        help='Number of batches (default: 1000)')
    parser.add_argument('--n-batches-per-eval', type=int, default=3,
                        help='Number of batches per evaluation (default: 3)')
    parser.add_argument('--resume-epoch', type=int, default=0,
                        help='Resume from epoch (default: 0)')
    
    return parser


def get_config_from_data(data, req_split_group: GROUP_SPLITS=GROUP_SPLITS.TRAIN):
    event_lens = []
    vocab = []
    for patient_data in data.values():
        patient_split_group = patient_data.get('split_group', GROUP_SPLITS.TRAIN.value)

        if req_split_group == GROUP_SPLITS.ALL or patient_split_group == req_split_group.value:
            event_lens.append(len(patient_data['events']))
            vocab.extend([event['codes'] for event in patient_data['events']])

    vocab_list = [PAD_TOKEN, UNK_TOKEN] + sorted(list(set(vocab)))
    
    return max(event_lens) if event_lens else 0, {v: k for k, v in enumerate(vocab_list)}


def get_device(device_name):
    if device_name == 'gpu' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    elif device_name == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    return device



def get_data_and_config_from_cmdline() -> Config:
    """Main function to parse command line arguments and return a Config object."""
    parser = create_argument_parser()
    args = parser.parse_args()

    args.data_dir = Path(args.data_dir)
    args.log_dir = Path(args.log_dir)
    args.out_dir = Path(args.out_dir)

    with open(args.data_dir / f"{args.dataset_name}.json") as f:
        data = json.load(f)

    max_events_length, args.vocab = get_config_from_data(data)

    if args.max_events_length is None:
        args.max_events_length = max_events_length

    if args.pad_size is None:
        args.pad_size = args.max_events_length

    args.device = get_device(args.device_name)

    if args.run_name is None:
        args.run_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    return data, Config(**vars(args))
