from numpy.core.records import ndarray
from kornia.losses.ssim import SSIMLoss
from .pix2pix_model import Pix2PixModel
import torch
from skimage.exposure import rescale_intensity
import numpy as np
from skimage.metrics import  structural_similarity



def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

# Stretchs histogram to 95%
def contrast_stretching(band: ndarray) -> ndarray:
    percentile_025 = np.percentile(band, 2.5)
    percentile_975 = np.percentile(band, 97.5)
    return rescale_intensity(band, in_range=(percentile_025, percentile_975))

class SatelliteModel(Pix2PixModel):
    """This is a subclass of Pix2PixModel for translating SAR images into Optical images.
    The model training requires '-dataset_model satellite' dataset.
    By default, the colorization dataset will automatically set '--input_nc 2' and '--output_nc 13'.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, we use 'colorization' dataset for this model.
        See the original pix2pix paper (https://arxiv.org/pdf/1611.07004.pdf) and colorization results (Figure 9 in the paper)
        """
        Pix2PixModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataset_mode='satellite')
        return parser

    def __init__(self, opt):
        """Initialize the class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # reuse the pix2pix model
        Pix2PixModel.__init__(self, opt)
        # specify the images to be visualized.
        self.visual_names = ['real_A_rgb', 'real_B_rgb', 'fake_B_rgb']
        self.loss_names = ['G_GAN', 'G_L1', 'G_SSIM', 'D_real', 'D_fake']

        if self.isTrain:
            self.criterionSSIM = SSIMLoss(5)

    def getNdviLoss(self, tensor1, tensor2):

        image = tensor1.cpu().detach().numpy().transpose(0,2,3,1)[0, :, :, :]
        red = contrast_stretching(normalize(image[:, :, 3])) * 255
        nir = contrast_stretching(normalize(image[:, :, 7])) * 255

        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir.astype(float) - red.astype(float)) / (nir + red)
        ndvi = np.nan_to_num(ndvi)
        ndvi1 = contrast_stretching(ndvi)

        image = tensor2.cpu().detach().numpy().transpose(0,2,3,1)[0, :, :, :]
        red = contrast_stretching(normalize(image[:, :, 3])) * 255
        nir = contrast_stretching(normalize(image[:, :, 7])) * 255

        # Allow division by zero
       # np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir.astype(float) - red.astype(float)) / (nir + red)
        ndvi = np.nan_to_num(ndvi)
        ndvi2 = contrast_stretching(ndvi)


        loss = np.mean(np.abs(ndvi1 - ndvi2))
        return loss

    def calculateSSIM(self, tensor1, tensor2):
        image1 = tensor1.cpu().detach().numpy().transpose(0,2,3,1)[0, :, :, :]
        image2 = tensor2.cpu().detach().numpy().transpose(0,2,3,1)[0, :, :, :]

        ssim = 0
        for band in range(14):
            im1 = contrast_stretching(normalize(image1[:, :, band])) * 255
            im2 = contrast_stretching(normalize(image2[:, :, band])) * 255
            ssim_partial = structural_similarity(im1, im2)
            ssim += (1-ssim_partial)

        return ssim

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B

        # SSIM
        #self.loss_G_SSIM = self.criterionSSIM(self.fake_B, self.real_B) * 10
        self.loss_G_SSIM = self.calculateSSIM(self.fake_B, self.real_B) * 0.9

        # L1
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # combine loss and calculate gradients
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_SSIM + self.loss_G_L1
        self.loss_G.backward()

    def getNdvi(self, tensor):
        red = contrast_stretching(normalize(tensor[:, :, 3])) * 255
        nir = contrast_stretching(normalize(tensor[:, :, 7])) * 255

        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir.astype(float) - red.astype(float)) / (nir + red)
        ndvi = contrast_stretching(ndvi)

        return ndvi

    def sar2rgb(self, tensor):
        """Convert a Sentinel1 tensor image to a RGB numpy output
        Parameters:
            tensor:  2-channel numpy array (dB scale for VV and VH polarizations)

        Returns:
            rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
        """
        red = contrast_stretching(normalize(tensor[:, :, 0])) * 255
        green = contrast_stretching(normalize(tensor[:, :, 1])) * 255
        blue = contrast_stretching(normalize(red + green)) * 255
        rgb_image = np.dstack((red.astype('int'), green.astype('int'), blue.astype('int')))
        return rgb_image


    def optical2rgb(self, tensor):
        """Convert a Sentinel2 tensor image to a RGB numpy output
        Parameters:
            tensor:  13-channel numpy array

        Returns:
            rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
        """
        red = contrast_stretching(normalize(tensor[:, :, 3])) * 255
        green = contrast_stretching(normalize(tensor[:, :, 2])) * 255
        blue = contrast_stretching(normalize(tensor[:, :, 1])) * 255
        rgb_image = np.dstack((red.astype('int'), green.astype('int'), blue.astype('int')))
        return rgb_image


    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        real_A = self.real_A.cpu().detach().numpy().transpose(0,2,3,1)[0, :, :, :]
        real_B = self.real_B.cpu().detach().numpy().transpose(0,2,3,1)[0, :, :, :]
        fake_B = self.fake_B.cpu().detach().numpy().transpose(0,2,3,1)[0, :, :, :]

        self.real_A_rgb = self.sar2rgb(real_A)
        self.real_B_rgb = self.optical2rgb(real_B)
        self.fake_B_rgb = self.optical2rgb(fake_B)
