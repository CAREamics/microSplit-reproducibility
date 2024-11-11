import sys
from typing import Callable

from careamics.config import VAEAlgorithmConfig
from careamics.lightning import VAEModule
from careamics.lvae_training.eval_utils import (
    get_dset_predictions, stitch_predictions_new
)
from careamics.utils.metrics import avg_range_inv_psnr

# TODO: sorry for this :(
sys.path.insert(0, "/home/federico.carrara/Documents/projects/microSplit-reproducibility/")

from datasets import create_train_val_datasets
from utils.io import load_checkpoint, load_config, log_experiment
from utils.visual import plot_splitting_results


def evaluate(
    data_path: str,
    ckpt_path: str,
    mmse_count: int,
    subdset_type: str,
    data_configs_fn: Callable,
    load_data_fn : Callable,
    use_val_dset: bool = False,
) -> None:
    """Evaluate the model.
    
    Parameters
    ----------
    data_path : str
        The path to the data directory.
    ckpt_path : str
        The path to the checkpoint to evaluate.
    mmse_count : int
        The number of samples to generate for each tile.
    subdset_type : str
        The subdataset type to evaluate on.
    data_configs_fn : Callable
        The function to get the data configurations for the specific experiment.
        Select among the ones in `configs.data`.
    load_data_fn : Callable
        The function to load the data for the specific experiment.
        Select among the ones in `datasets`.
    use_val_dset : bool, optional
        Whether to use the validation dataset for evaluation, by default False.    
    """
    print(f"Evaluating model at {ckpt_path}")
    
    # get datasets and dataloaders
    train_data_config, val_data_config, test_data_config = data_configs_fn(subdset_type)
    _, val_dset, test_dset, _ = create_train_val_datasets(
        datapath=data_path,
        train_config=train_data_config,
        val_config=val_data_config,
        test_config=test_data_config,
        load_data_func=load_data_fn,
    )
    eval_dset = val_dset if use_val_dset else test_dset
    
    # load config
    algo_config = VAEAlgorithmConfig(**load_config(ckpt_path, "algorithm"))
    
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
        dset=eval_dset,
        batch_size=32,
        num_workers=4,
        mmse_count=mmse_count,
        loss_type=algo_config.loss,
    )
    print(
        f'Nb. pred tiles: {pred_tiled.shape[0]},',
        f'channels: {pred_tiled.shape[1]},',
        f'shape: {pred_tiled.shape[2:]}'
    )
    
    # stich predictions
    pred = stitch_predictions_new(predictions=pred_tiled, dset=eval_dset)
    print(f"Predictions shape: {pred.shape}")
    
    # get target data
    target = eval_dset._data 
    # NOTE: this target works for HTIba1Ki67, might need to change for other
    # TODO: make generic `get_target()` function
    
    # compute metrics
    rinv_psnr_arr = []
    # ssim_arr = []
    # micro_ssim_arr = []
    for ch_id in range(pred.shape[-1]):
        rinv_psnr = avg_range_inv_psnr(
            target=target[..., ch_id].copy(),
            prediction=pred[..., ch_id].copy()
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
        log_dir="./log/",
        dset_id=str(test_data_config.data_type).split(".")[-1],
        algorithm=algo_config.algorithm,
        metrics={"rinv_psnr": rinv_psnr_arr},
        eval_info={
            "grid_size": test_data_config.grid_size,
            "subdset_type": subdset_type,
        },
        ckpt_dir=ckpt_path
    )
    
    # plot results
    plot_splitting_results(preds=pred, gts=target, preds_std=None)

