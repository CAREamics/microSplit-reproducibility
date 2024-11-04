import argparse
import sys

from careamics.lightning import VAEModule
from careamics.lvae_training.eval_utils import (
    get_dset_predictions, stitch_predictions
)
from careamics.utils.metrics import avg_range_inv_psnr

# TODO: sorry for this :(
sys.path.insert(0, "/home/federico.carrara/Documents/projects/microSplit-reproducibility/")

from configs import get_algorithm_config
from datasets import create_train_val_datasets
from utils import fix_seeds, get_ignored_pixels
from utils.io import load_checkpoint, load_config, log_experiment
# NOTE: the following imports are datasets and algorithm dependent
from configs.data.ht_iba1_ki64 import get_data_configs
from configs.parameters.ht_iba1_ki64 import get_musplit_parameters
from datasets.ht_iba1_ki64 import get_train_val_data
    
    
def evaluate(
    data_path: str,
    ckpt_path: str,
):
    params = get_musplit_parameters()
    
    # get datasets and dataloaders
    train_data_config, test_data_config = get_data_configs()
    train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
        datapath=data_path,
        train_config=train_data_config,
        val_config=test_data_config,
        test_config=test_data_config,
        load_data_func=get_train_val_data,
    )
    
    model_config = load_config(ckpt_path, "model")
    # TODO: all the previous configs can also be istantiated from a single function...
    algo_config = get_algorithm_config(
        algorithm=params["algorithm"],
        model_config=model_config,
    )
    
    # load checkpoint
    checkpoint = load_checkpoint(ckpt_path, best=True)
    
    # init lightning model
    lightning_model = VAEModule(algorithm_config=algo_config)
    lightning_model.load_state_dict(checkpoint['state_dict'], strict=True)
    lightning_model.eval()
    lightning_model.cuda()
    
    # get predictions
    pred_tiled = get_dset_predictions(
        model=lightning_model,
        dset=test_dset,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        mmse_count=params["mmse_count"],
        loss_type=algo_config["loss"],
    )
    print(
        f'Nb. pred tiles: {pred_tiled.shape[0]},',
        f'channels: {pred_tiled.shape[1]},',
        f'shape: {pred_tiled.shape[2:]}'
    )
    
    # stich predictions
    pred = stitch_predictions(predictions=pred_tiled, dset=test_dset)
    print(f"Predictions shape: {pred.shape}")

    # ignore pixels
    ignored_pixels = get_ignored_pixels(pred[0]) # how many pixels to ignore...
    ... # ignore_pixels()
    
    # plot something
    ...
    
    # compute metrics
    target = test_dset.dsets[0]._data
    rinv_psnr_arr = []
    # ssim_arr = []
    # micro_ssim_arr = []
    for ch_id in range(pred.shape[-1]):
        rinv_psnr = avg_range_inv_psnr(
            target=target[...,ch_id].copy(), # the fuck is target here??
            prediction=pred[...,ch_id].copy()
        )
        rinv_psnr_arr.append(rinv_psnr)
        # ssim_mean, ssim_std = avg_ssim(tar[...,ch_id], pred_unnorm[ch_id])
        # psnr_arr.append(psnr)
        # ssim_arr.append((ssim_mean,ssim_std))
    print(f'RangeInvPSNR: ',' <--> '.join([str(x) for x in rinv_psnr_arr]))
    
    # write metrics on a JSON
    # NOTE: JSON file will be updated at each evaluation
    # - we will keep a JSON file for each experiment
    # - each experiment will be tracked by a timestamp
    # - the JSON file will tracked with git
    log_experiment(
        log_dir="../log/",
        dset_id=test_data_config.data_type,
        algorithm_id=params["algorithm"],
        eval_info={test_data_config.grid_size},
        ckpt_dir=ckpt_path
    )
    

if __name__ == "__main__":
    fix_seeds(0)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="The path to the training checkpoints.",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="The path to the data directory.",
        required=True,
    )
    args = parser.parse_args()
    evaluate(ckpt_path=args.ckpt_path, data_path=args.data_path)

