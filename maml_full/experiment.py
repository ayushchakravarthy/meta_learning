"""
Reproducing Model-Agnostic Meta-Learning result(supervised)
"""


from torch.utils.data import DataLoader
from torch import nn
import argparse


from datasets import OmniglotDataset, MiniImageNet
from core import NShotTaskSampler, create_nshot_task_label, EvaluateFewShot
from maml import meta_gradient_step
from models import FewShotClassifier
from train import fit
from callbacks import *
from utils import setup_dirs
from config import PATH

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


# Parameters parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--n', default=1, type=int)
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--q', default=1, type=int)  # Number of examples per class
parser.add_argument('--inner-train-steps', default=1, type=int)
parser.add_argument('--innder-val-steps', default=3, type=int)
parser.add_argument('--inner-lr', default=0.4, type=float)
parser.add_argument('--meta-lr', default=0.01, type=float)
parser.add_argument('--meta-batch-size', default=32, type=int)
parser.add_argument('--order', default=1, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--epoch-len', default=100, type=int)
parser.add_argument('--eval-batches', default=20, type=int)

args = parser.parse_args()

if args.dataset == 'omniglot':
    dataset_class = OmniglotDataset
    fc_layer_size = 64
    num_input_channels = 1
elif args.dataset == 'miniImageNet':
    dataset_class = MiniImageNet
    fc_layer_size = 1600
    num_input_channels = 3
else:
    raise(ValueError('Unsupported Dataset'))

params_str = f'{args.dataset}_order={args.order}_n={args.n}_k={args.k}_metabatch={args.meta_batch_size}_' \
    f'
