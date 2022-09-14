import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import rasterio
import torch
import numpy as np
from skimage.exposure import rescale_intensity
from skimage import feature
import findpeaks
import albumentations as A
import random
from skimage.filters import threshold_otsu
from skimage.filters import gabor_kernel
from skimage import io
from scipy import ndimage as ndi

class SatelliteDataset(BaseDataset):
    """
    This dataset class can load aligned datasets of images with different amount of channels

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """

        parser.add_argument('--scenario', type=str, default='base', help='Scenario selection')

        return parser


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
        self.transform = A.Compose([
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
        self.phase = self.opt.phase
        self.gabor_kernels = self.create_gabor_kernels()
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

        self.scenario = self.opt.scenario

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
            filtered_band = self.normalize(self.contrast_stretching(ndi.convolve(image, kernel, mode='wrap')))
            filtered_bands.append(filtered_band)
        return filtered_bands


    def normalize(self, array):
        array_min, array_max = array.min(), array.max()
        return (array - array_min) / (array_max - array_min)

    # Stretchs histogram to 95%
    def contrast_stretching(self, band: np.ndarray) -> np.ndarray:
        percentile_025 = np.percentile(band, 2.5)
        percentile_975 = np.percentile(band, 97.5)
        return rescale_intensity(band, in_range=(percentile_025, percentile_975))

    def read_sar_tiff_rt(self, path: str):
        bands = list()
        with rasterio.open(path, mode='r') as image:
            vv = self.normalize(self.contrast_stretching(image.read(1)))
            vh = self.normalize(self.contrast_stretching(image.read(2)))

            bands.append(vv)
            bands.append(vh)

            # Allow division by zero
            #np.seterr(divide='ignore', invalid='ignore')

            # Calculate RVI
            #rvi = (4 * vh.astype(float))/(vv.astype(float)+vh.astype(float))
            #rvi = np.nan_to_num(rvi)
            #rvi = self.normalize(rvi)
            #bands.append(rvi)

            # VH / VV
            #rate = vh.astype(float)/vv.astype(float)
            #rate = np.nan_to_num(rate)
            #rate = self.normalize(rate)
            #bands.append(rate)

            # RFDI
            #rfdi = (vv.astype(float) - vh.astype(float))/(vv.astype(float) + vh.astype(float))
            #rfdi = np.nan_to_num(rfdi)
            #rfdi = self.normalize(rfdi)
            #bands.append(rfdi)

            #Contours of vh and vv
            # vh_aux = findpeaks.stats.scale(vh, verbose=0)
            # denoised_vh = findpeaks.stats.denoise(vh_aux, method='fastnl', window=25, verbose=0)
            # vh_contours = feature.canny(denoised_vh, sigma=5)
            # bands.append(vh_contours)
            # vv_aux= findpeaks.stats.scale(vv, verbose=0)
            # denoised_vv = findpeaks.stats.denoise(vv_aux, method='fastnl', window=25, verbose=0)
            # vv_contours = feature.canny(denoised_vv, sigma=5)
            # bands.append(vv_contours)

            #Denoised vh and vv
            # vh_aux = findpeaks.stats.scale(vh, verbose=0)
            # denoised_vh = findpeaks.stats.denoise(vh_aux, method='fastnl', window=25, verbose=0)
            # bands.append(denoised_vh)
            # vv_aux= findpeaks.stats.scale(vv, verbose=0)
            # denoised_vv = findpeaks.stats.denoise(vv_aux, method='fastnl', window=25, verbose=0)
            # bands.append(denoised_vv)


            return np.dstack(tuple(bands))

    def global_otsu(self, image):
        threshold_global_otsu = threshold_otsu(image)
        global_otsu = image >= threshold_global_otsu
        return global_otsu

    def read_sar_tiff(self, path: str):
        bands = list()
        with rasterio.open(path, mode='r') as image:
            vv = self.normalize(self.contrast_stretching(image.read(1)))
            vh = self.normalize(self.contrast_stretching(image.read(2)))

            if self.scenario == 'base':
                bands.append(vv)
                bands.append(vh)

            elif self.scenario == 'scenario-1':
                rate = self.normalize(self.contrast_stretching(image.read(4)))
                rate_minus_rvi = self.normalize(self.contrast_stretching(image.read(6)))
                bands.append(vv)
                bands.append(vh)
                bands.append(rate)
                bands.append(rate_minus_rvi)

            elif self.scenario == 'scenario-2':
                rvi = self.normalize(self.contrast_stretching(image.read(3)))
                rate = self.normalize(self.contrast_stretching(image.read(4)))
                rfdi = self.normalize(self.contrast_stretching(image.read(5)))
                rate_minus_rvi = self.normalize(self.contrast_stretching(image.read(6)))
                rate_mult_rfdi = self.normalize(self.contrast_stretching(image.read(7)))
                bands.append(vv)
                bands.append(vh)
                bands.append(rvi)
                bands.append(rate)
                bands.append(rfdi)
                bands.append(rate_minus_rvi)
                bands.append(rate_mult_rfdi)

            elif self.scenario == 'scenario-3':
                otsu_mask = self.global_otsu(vh)
                bands.append(vv)
                bands.append(vh)
                bands.append(otsu_mask)

            elif self.scenario == 'scenario-4':
                # Canny
                vh = findpeaks.stats.scale(vh, verbose=0)
                denoised_vh = findpeaks.stats.denoise(vh, method='fastnl', window=5, verbose=0)
                vh_contours = feature.canny(denoised_vh, sigma=5)
                vv = findpeaks.stats.scale(vv, verbose=0)
                denoised_vv = findpeaks.stats.denoise(vv, method='fastnl', window=5, verbose=0)
                vv_contours = feature.canny(denoised_vv, sigma=5)
                bands.append(vv)
                bands.append(vh)
                bands.append(vh_contours) # Canny
                bands.append(vv_contours) # Canny

            elif self.scenario == 'scenario-5':
                bands.append(vv)
                bands.append(vh)
                bands.extend(self.add_gabor_filtered_images(vv)) # Gabor (+16)
                bands.extend(self.add_gabor_filtered_images(vh)) # Gabor (+16)

            elif self.scenario == 'scenario-6':
                bands.append(vv)
                bands.append(vh)

            elif self.scenario == 'scenario-7':
                rate = self.normalize(self.contrast_stretching(image.read(4)))
                rfdi = self.normalize(self.contrast_stretching(image.read(5)))
                rate_minus_rvi = self.normalize(self.contrast_stretching(image.read(6)))
                bands.append(vv)
                bands.append(vh)
                bands.append(rate)
                bands.append(rfdi)
                bands.append(rate_minus_rvi)

            elif self.scenario == 'scenario-8':
                rate = self.normalize(self.contrast_stretching(image.read(4)))
                rate_minus_rvi = self.normalize(self.contrast_stretching(image.read(6)))
                bands.append(vv)
                bands.append(vh)
                bands.append(rate)
                bands.append(rate_minus_rvi)
            
            return np.dstack(tuple(bands))

    def read_optical_tiff(self, path: str):
        bands = list()
        with rasterio.open(path, mode='r') as image:
            if int(self.output_nc) == 3:
                bands.append(self.normalize(self.contrast_stretching(image.read(2))))
                bands.append(self.normalize(self.contrast_stretching(image.read(3))))
                bands.append(self.normalize(self.contrast_stretching(image.read(4))))
            elif int(self.output_nc) == 13:
                for i in range(1, image.count + 1):
                    bands.append(self.normalize(self.contrast_stretching(image.read(i))))
            else:
                raise RuntimeError('Wrong amount of output bands')

            return np.dstack(tuple(bands))


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

        # Read images
        A_img = self.read_sar_tiff(A_path)
        B_img = self.read_optical_tiff(B_path)

        # Augmentation on train
        if self.phase is 'train':
            seed = random.randint(0, 99999)
            random.seed(seed)
            A_img = self.transform(image=A_img)["image"]
            random.seed(seed)
            B_img = self.transform(image=B_img)["image"]

        # Channel reordering
        A_img = A_img.transpose(2, 0, 1)
        B_img = B_img.transpose(2, 0, 1)

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
