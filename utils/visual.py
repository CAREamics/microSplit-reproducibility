from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from careamics.utils.metrics import scale_invariant_psnr
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _add_colorbar(img: torch.Tensor, fig, ax):
    """Add colorbar to a `matplotlib` image."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

def plot_splitting_results(
    preds: torch.Tensor,
    gts: torch.Tensor,
    preds_std: Optional[torch.Tensor] = None,
    idx: Optional[int] = None,
) -> None:
    """Plot a predicted image with the associated GT.
    
    Parameters
    ----------
    preds : torch.Tensor
        The predicted images. Shape is (N, H, W, F).
    gts : torch.Tensor
        The ground truth images. Shape is (N, H, W, F).
    preds_std : Optional[torch.Tensor]
        The predicted std deviation. If `None`, it won't be plotted. Default is `None`.
    idx : Optional[int], optional
        The index of the image to plot, by default None.
    """
    N, F = preds.shape[0], preds.shape[-1]
    
    if idx is None:
        idx = np.random.randint(0, N - 1)
        
    ncols = 4 if preds_std else 3
    fig, axes = plt.subplots(F, ncols, figsize=(6 * ncols, 5 * F))
    for i in range(F):
        # GT
        gt = gts[idx, :, :, i]
        axes[i, 0].set_title(f"FP {i+1} - GT")
        im_gt = axes[i, 0].imshow(gt)
        _add_colorbar(im_gt, fig, axes[i, 0])
        # Pred
        pred = preds[idx, :, :, i]
        axes[i, 1].set_title(f"FP {i+1} - Pred")
        im_pred = axes[i, 1].imshow(pred)
        _add_colorbar(im_pred, fig, axes[i, 1])
        # MAE
        norm_pred = (pred - pred.min()) / (pred.max() - pred.min())
        norm_gt = (gt - gt.min()) / (gt.max() - gt.min())
        mae = np.abs(norm_pred - norm_gt)
        axes[i, 2].set_title(f"FP {i+1} - MAE")
        im_mae = axes[i, 2].imshow(mae, cmap='RdPu')
        _add_colorbar(im_mae, fig, axes[i, 2])
        psnr = scale_invariant_psnr(pred, gt)
        axes[i, 2].text(
            0.66, 0.1, f'PSNR: {psnr:.2f}', 
            transform=axes[i, 2].transAxes,
            fontsize=12, 
            verticalalignment='center', 
            bbox=dict(facecolor='white', alpha=0.5)
        )
        # Pred Std
        if preds_std:
            axes[i, 3].set_title(f"FP {i+1} - Pred Std")
            im_std = axes[i, 3].imshow(preds_std[idx, :, :, i])
            _add_colorbar(im_std, fig, axes[i, 3])
    plt.show()