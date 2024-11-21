import sys
sys.path.insert(0, "/home/ashesh.ashesh/code/microSplit-reproducibility/")

from configs.data.puncta_removal import get_train_data_configs
from configs.parameters.puncta_removal import get_microsplit_parameters, get_musplit_parameters
from datasets.multicrop_dset_train_val_data import get_train_val_data   
from run import train_denoiSplit, train_muSplit
from utils.io import get_training_args


if __name__ == "__main__":
    args = get_training_args()
    
    if args.algorithm == "denoiSplit":
        train_fn = train_denoiSplit
        param_fn = get_microsplit_parameters
    elif args.algorithm == "muSplit": 
        train_fn = train_muSplit
        param_fn = get_musplit_parameters
    else:
        raise ValueError(f"Algorithm {args.algorithm} not recognized. Please choose between 'denoiSplit' and 'muSplit'.")
    
    train_fn(
        root_path=args.root_path,
        data_path=args.data_path,
        param_fn=param_fn,
        data_configs_fn=get_train_data_configs,
        load_data_fn=get_train_val_data,
        wandb_project=args.wandb_project
    )