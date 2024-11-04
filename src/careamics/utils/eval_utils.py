import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from careamics.lightning import VAEModule
from careamics.config.tile_information import TileInformation
    

def get_tiled_predictions(
    model: VAEModule,
    dloader: DataLoader,
    mmse_count: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[TileInformation]]:
    """Get tiled predictions from a model for the entire dataset.

    Parameters
    ----------
    model : VAEModule
        Lightning model used for prediction.
    dloader : DataLoader
        DataLoader to predict on. Dataset must be a `InMemoryTiledPredDataset` object.
    mmse_count : int, optional
        Number of samples to generate for each input and then to average over for
        MMSE estimation, by default 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[TileInformation]]
        Tuple containing:
            - predictions: Predicted unmixed images for the dataset. Shape is (N, F, H, W).
            - predictions_std: Standard deviation of the predicted unmixed images. Shape is (N, F, H, W).
            - reconstructions: Reconstructed (spectral) images. Shape is (N, C, H, W).
            - tiles_info: Information about the tiles coordinates. Length is N.
    """
    predictions = []
    predictions_std = []
    reconstructions = []
    tiles_info = []
    with torch.no_grad():
        for batch in tqdm(dloader, desc="Predicting patches"):
            inp, tinfo = batch
            inp = inp.cuda()
            tiles_info.extend(tinfo)

            rec_img_list = []
            unmix_img_list = []
            for _ in range(mmse_count):
                rec, unmix, _ = model(inp)
                
                if model.model.predict_logvar is None:
                    unmix_img = unmix
                    rec_img = rec
                    logvar = torch.tensor([-1])
                else:
                    unmix_img, _ = torch.chunk(unmix, chunks=2, dim=1)
                    rec_img, logvar = torch.chunk(rec, chunks=2, dim=1)
                
                unmix_img_list.append(unmix_img.cpu().unsqueeze(0)) # add MMSE dim
                rec_img_list.append(rec_img.cpu().unsqueeze(0)) # add MMSE dim

            # aggregate results
            unmixs = torch.cat(unmix_img_list, dim=0)
            unmix_mmse_imgs = torch.mean(unmixs, dim=0)
            unmix_mmse_std = torch.std(unmixs, dim=0)
            predictions.append(unmix_mmse_imgs.cpu().numpy())
            predictions_std.append(unmix_mmse_std.cpu().numpy())
            recs = torch.cat(rec_img_list, dim=0)
            rec_msse_imgs = torch.mean(recs, dim=0)
            reconstructions.append(rec_msse_imgs.cpu().numpy())

    predictions = np.concatenate(predictions)
    predictions_std = np.concatenate(predictions_std)
    reconstructions = np.concatenate(reconstructions)
    return predictions, predictions_std, reconstructions, tiles_info


def coarsen_img(
    img: np.ndarray, 
    downscaling_factor: int
) -> np.ndarray:
    """Coarsen an image by summing over blocks of pixels.
    
    Parameters
    ----------
    img : np.ndarray
        Image to coarsen. Shape is (C, Z, Y, X) or (C, Y, X).
    downscaling_factor : int
        Factor by which to downscale the image.
    """
    
    assert isinstance(downscaling_factor, int), "Downscaling factor must be a single int!"
    
    if img.ndim == 4:
        C, Z, Y, X = img.shape
        return img.reshape(
            C, 
            Z // downscaling_factor, downscaling_factor, 
            Y // downscaling_factor, downscaling_factor, 
            X // downscaling_factor, downscaling_factor
        ).sum(axis=(2, 4, 6))
    elif img.ndim == 3:
        C, Y, X = img.shape
        return img.reshape(
            C,
            Y // downscaling_factor, downscaling_factor, 
            X // downscaling_factor, downscaling_factor
        ).sum(axis=(2, 4))
