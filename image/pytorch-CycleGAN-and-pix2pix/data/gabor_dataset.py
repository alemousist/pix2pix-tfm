import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import rasterio
import torch
import numpy as np

from skimage.filters import gabor_kernel
from skimage import io
from scipy import ndimage as ndi

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

class GaborDataset(BaseDataset):
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

        self.gabor_kernels = self.create_gabor_kernels()


    def create_gabor_kernels(self):
        kernels = []
        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (2, 7):
                for frequency in (0.1, 0.15):
                    kernel = np.real(gabor_kernel(frequency, theta=theta,
                                                  sigma_x=sigma, sigma_y=sigma))
                    kernels.append(kernel)
        return kernels

    def add_gabor_filtered_images(self, image):
        filtered_bands = list()
        for kernel in self.gabor_kernels:
            filtered_bands.append(ndi.convolve(image, kernel, mode='wrap'))
        return filtered_bands

    def read_sar_tiff(self, path: str):
        bands = list()
        with rasterio.open(path, mode='r') as image:
            for i in range(1, image.count + 1):
                bands.append(normalize(image.read(i)))

            filtered_bands = list()
            for i in range(0, image.count):
                filtered_bands.extend(self.add_gabor_filtered_images(bands[i]))
            bands.extend(filtered_bands)
            return np.dstack(tuple(bands)).transpose(2,0,1)


    def read_optical_tiff(self, path: str):
        bands = list()
        with rasterio.open(path, mode='r') as image:
            for i in range(1, image.count + 1):
                bands.append(normalize(image.read(i)))
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
