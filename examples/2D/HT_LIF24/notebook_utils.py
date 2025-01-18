import matplotlib.pyplot as plt
import numpy as np
from careamics.lvae_training.eval_utils import get_predictions
import os

def get_all_channel_list(target_channel_list):
    """
    Adds the input channel index to the target channel list.
    """
    input_channel_index_dict = {
        '01' : 8,
        '02' : 9,
        '03' : 10,
        '12' : 11,
        '13' : 12,
        '23' : 13,
    }
    return target_channel_list + [input_channel_index_dict[''.join([str(i) for i in target_channel_list])]]

def get_unnormalized_predictions(model, dset, exposure_duration, target_channel_idx_list,num_workers=4, batch_size=8):
    """
    Get the stitched predictions which have been unnormlized.
    """
    # You might need to adjust the batch size depending on the available memory
    stitched_predictions, _ = get_predictions(
        model=model,
        dset=dset,
        batch_size=8,
        num_workers=num_workers,
        mmse_count=2,#experiment_params["mmse_count"],
        tile_size=model.model.image_size,
    )
    stitched_predictions = stitched_predictions[exposure_duration]
    stitched_predictions = stitched_predictions[...,:len(target_channel_idx_list)]
    mean_params, std_params = dset.get_mean_std()
    stitched_predictions = stitched_predictions*std_params['target'].squeeze().reshape(1,1,1,-1) + mean_params['target'].squeeze().reshape(1,1,1,-1)
    return stitched_predictions

def get_target(dset):
    return dset._data[...,:-1].copy()

def get_input(dset):
    return dset._data[...,-1].copy()



def pick_random_patches_with_content(tar, patch_size):    
    H, W = tar.shape[1:3]
    std_patches = []
    indices = []
    for i in range(1000):
        h_start = np.random.randint(H - patch_size)
        w_start = np.random.randint(W - patch_size)
        std_tmp= []
        for ch_idx in range(tar.shape[-1]):
            std_tmp.append(tar[0,h_start:h_start+patch_size,w_start:w_start+patch_size,ch_idx].std())
        
        std_patches.append(np.mean(std_tmp))
        indices.append((h_start,w_start))
    
    # sort by std
    indices = np.array(indices)[np.argsort(std_patches)][-40:]

    distances = np.linalg.norm(indices[:,None] - indices[None], axis=-1)
    # pick the indices of the indices that are at least patch_size pixels apart
    final_indices = [0]
    for i in range(1, len(indices)):
        if np.all(distances[i,final_indices] >= patch_size):
            final_indices.append(i)

    final_indices = indices[final_indices,:]
    return final_indices

def full_frame_evaluation(stitched_predictions, tar, inp):

    ncols = tar.shape[-1] + 1
    nrows = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    ax[0,0].imshow(inp)
    for i in range(ncols -1):
        vmin = stitched_predictions[...,i].min()
        vmax = stitched_predictions[...,i].max()
        ax[0,i+1].imshow(tar[...,i], vmin=vmin, vmax=vmax)
        ax[1,i+1].imshow(stitched_predictions[...,i], vmin=vmin, vmax=vmax)

    # disable the axis for ax[1,0]
    ax[1,0].axis('off')
    ax[0,0].set_title("Input", fontsize=15)
    ax[0,1].set_title("Channel 1", fontsize=15)
    ax[0,2].set_title("Channel 2", fontsize=15)
    # set y labels on the right for ax[0,2]
    ax[0,2].yaxis.set_label_position("right")
    ax[0,2].set_ylabel("Target", fontsize=15)

    ax[1,2].yaxis.set_label_position("right")
    ax[1,2].set_ylabel("Predicted", fontsize=15)


def find_recent_metrics():
    last_idx = 1
    fpath_schema = "./lightning_logs/version_{run_idx}/metrics.csv"
    assert os.path.exists(fpath_schema.format(run_idx=last_idx)), f"File {fpath_schema.format(run_idx=last_idx)} does not exist"
    while os.path.exists(fpath_schema.format(run_idx=last_idx)):
        last_idx += 1
    last_idx -= 1
    return fpath_schema.format(run_idx=last_idx)


def plot_metrics(df):
    _,ax = plt.subplots(figsize=(12,3),ncols=4)
    df['reconstruction_loss_epoch'].to_frame('recons_loss').dropna().reset_index(drop=True).plot(ax=ax[0],marker='o')
    df[['kl_loss_epoch']].dropna().reset_index(drop=True).plot(ax=ax[1],marker='o')
    df[['val_loss']].dropna().reset_index(drop=True).plot(ax=ax[2],marker='o')
    df[['val_psnr']].dropna().reset_index(drop=True).plot(ax=ax[3], marker='o')
    plt.tight_layout()
    # switch on the grid
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()
    # set background color to gray
    ax[0].set_facecolor('lightgray')
    ax[1].set_facecolor('lightgray')
    ax[2].set_facecolor('lightgray')
    ax[3].set_facecolor('lightgray')
    # set a common x label for all the subplots
    for a in ax:
        a.set_xlabel("Epoch")