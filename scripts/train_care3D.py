import sys
sys.path.insert(0, "/home/ashesh.ashesh/code/microSplit-reproducibility/")

from configs.data.care3D import get_train_data_configs
from configs.parameters.care3D import get_microsplit_parameters, get_musplit_parameters
from datasets.care3D import get_train_val_data   
from run import train_denoiSplit, train_muSplit
from utils.io import get_training_args_default
from functools import partial


def get_training_args():
    parser = get_training_args_default()
    parser.add_argument('--subdset_type', choices=['zebrafish','liver'], default='zebrafish')
    parser.add_argument('--depth3D', type=int, default=9)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_training_args()
    
    if args.algorithm == "denoiSplit":
        train_fn = train_denoiSplit
        param_fn = partial(get_microsplit_parameters, subdset_type=args.subdset_type, depth3D=args.depth3D)
    elif args.algorithm == "muSplit": 
        train_fn = train_muSplit
        param_fn = partial(get_musplit_parameters, depth3D=args.depth3D)    
    else:
        raise ValueError(f"Algorithm {args.algorithm} not recognized. Please choose between 'denoiSplit' and 'muSplit'.")
    
    train_fn(
        root_path=args.root_path,
        data_path=args.data_path,
        param_fn=param_fn,
        data_configs_fn=partial(get_train_data_configs, subdset_type=args.subdset_type, depth3D=args.depth3D),
        load_data_fn=get_train_val_data,
        wandb_project=args.wandb_project
    )