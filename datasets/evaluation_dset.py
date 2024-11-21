"""
This data loader is designed just for test images. 
"""
import numpy as np
from data.data_utils import GridIndexManager, TilingMode


class EvaluationDset:
    def __init__(self, image: np.ndarray, normalizer_fn, image_size:int , grid_size:int, tiling_mode=TilingMode.ShiftBoundary):
        """
        Args:
            image: N x H x W
            normalizer_fn: function to normalize the image
            image_size: size of the patch which will be input to the model
            grid_size: size of the tile used for tiling
            tiling_mode: TilingMode
        """
        assert len(image.shape) == 3, "Image should be N x H x W"
        # N x 1 x H x W
        self._data = image[..., np.newaxis]

        self._normalizer = normalizer_fn
        self._img_sz = image_size
        self._grid_sz = grid_size
        self._tiling_mode = tiling_mode
        
        numC = 1
        grid_shape = (1, grid_size, grid_size,numC)
        patch_shape = (1, self._img_sz, self._img_sz,numC)
        self.idx_manager = GridIndexManager(self._data.shape, grid_shape, patch_shape, self._tiling_mode)
    
    def __len__(self):
        return self.idx_manager.total_grid_count()

    def get_data_shape(self):
        return self._data.shape

    def get_grid_size(self):
        return self._grid_sz

    def __getitem__(self, idx):
        loc_list = self.idx_manager.get_patch_location_from_dataset_idx(idx)
        img_idx, h_start, w_start,ch_idx = loc_list
        assert ch_idx == 0, "Only single channel images are supported"
        img = self._data[img_idx,h_start:h_start+self._img_sz,w_start:w_start+self._img_sz,0]
        return self._normalizer(img)[None]