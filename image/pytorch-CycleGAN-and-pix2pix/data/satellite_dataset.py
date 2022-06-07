import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import rasterio
import torch
import numpy as np
from skimage.exposure import rescale_intensity


# Stretchs histogram to 95%
def contrast_stretching(band: np.ndarray) -> np.ndarray:
    percentile_025 = np.percentile(band, 2.5)
    percentile_975 = np.percentile(band, 97.5)
    return rescale_intensity(band, in_range=(percentile_025, percentile_975))

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

class SatelliteDataset(BaseDataset):
    """
    This dataset class can load aligned datasets of images with different amount of channels

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def read_sar_tiff(self, path: str):
        bands = list()
        with rasterio.open(path, mode='r') as image:
            for i in range(1, image.count + 1):
                bands.append(normalize(image.read(i)))


            # Allow division by zero
            np.seterr(divide='ignore', invalid='ignore')

            vv = image.read(1)
            vh = image.read(2)

            # Calculate RVI
            rvi = (4 * vh.astype(float))/(vv.astype(float)+vh.astype(float))
            rvi = np.nan_to_num(rvi)
            rvi = contrast_stretching(rvi)
            bands.append(rvi)

            return np.dstack(tuple(bands)).transpose(2,0,1)




    def read_optical_tiff(self, path: str):
        bands = list()
        with rasterio.open(path, mode='r') as image:
            for i in range(1, image.count + 1):
                bands.append(normalize(image.read(i)))

            red = image.read(4)
            nir = image.read(8)

            # Allow division by zero
            np.seterr(divide='ignore', invalid='ignore')

            # Calculate NDVI
            ndvi = (nir.astype(float) - red.astype(float)) / (nir + red)
            ndvi = np.nan_to_num(ndvi)
            ndvi = contrast_stretching(normalize(ndvi))

            bands.append(ndvi)

            return np.dstack(tuple(bands)).transpose(2, 0, 1)


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.B_size]
        A_img = self.read_sar_tiff(A_path)
        B_img = self.read_optical_tiff(B_path)

        # Data type cast
        A = A_img.astype('float32')
        B = B_img.astype('float32')

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
