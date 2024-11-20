import sys
sys.path.insert(0, "/home/federico.carrara/Documents/projects/microSplit-reproducibility/")

from configs.data.ht_iba1_ki64_2023 import get_eval_data_configs
from datasets.ht_iba1_ki64_2023 import get_train_val_data
from run import evaluate
from utils import fix_seeds
from utils.io import get_evaluation_args


if __name__ == "__main__":
    fix_seeds(0)
    args = get_evaluation_args()
    evaluate(
        ckpt_path=args.ckpt_path, 
        data_path=args.data_path, 
        mmse_count=args.mmse_count,
        subdset_type=args.subdset_type,
        data_configs_fn=get_eval_data_configs,
        load_data_fn=get_train_val_data,
        use_val_dset=False
    )