import random

import numpy as np
import torch

def fix_seeds(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def get_ignored_pixels(pred: torch.Tensor) -> int:
    """Get the number of ignored pixels in the predictions.
    
    Some of the pixels present in the last few rows and columns of predicted images
    should be discarded. Indeed, the prediction there is always zero, since they don't 
    come in batches as not multiples of the tile size.
        
    To spot these areas, we analyze the pixel-wise std of a predicted image. When that
    is zero, we know that the last few rows and columns are all zero.
    
    NOTE: this function is just to understand the number of ignored pixels.
    The function to discard these pixels is `ignore_pixels`.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted image. Shape is (H, W, C).
        
    Returns
    -------
    int
        Number of ignored pixels.
    """
    ignored_pixels = 1
    while(pred[-ignored_pixels:, -ignored_pixels:, ...].std() == 0):
        ignored_pixels+=1
    ignored_pixels-=1
    print(f'In {pred.shape}, last {ignored_pixels} many rows and columns are zero.')
    return ignored_pixels