import json

def add_basic_NN_arguments(parser):
    """
    Cmd line arguments that are common to several networks.
    """
    parser.add_argument('--model_root', required=True,
                      help='Full path of where to output the results of training.')
    # Data
    parser.add_argument(
        '--train_data_root', type=str, required=True,
        help='Where to find the tfrecord files (default: %(default)s)')
    parser.add_argument(
        '--val_data_root', type=str, required=True,
        help='Where to find the tfrecord files (default: %(default)s)')
    # Hyperparams
    parser.add_argument(
        '--lr', type=float, default=3e-4,
        help='Learning rate to use (default: %(default)s)')
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch Size')
    parser.add_argument(
        '--dropout', type=float, default=0.2,
        help='Dropout rate.')
    parser.add_argument(
        '--train_shift', type=json.loads, default=None,
        help='eg [1.5] or [1,2]')
    parser.add_argument(
        '--val_shift', type=json.loads, default=None,
        help='...')
    parser.add_argument(
        '--train_center', type=json.loads, default=None,
        help='...')
    parser.add_argument(
        '--val_center', type=json.loads, default=None,
        help='...')
    # Other
    parser.add_argument(
        '--max_steps', type=int, default=10000000,
        help='Number of epochs to run training for.')
    parser.add_argument(
        '--eval_period', type=int, default=5000,
        help='Eval after every N itrs.')
    parser.add_argument(
        '--eval_only', action='store_true',
        help='Do not train only eval.')
    return parser
    