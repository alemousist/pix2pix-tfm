from numpy.core.records import ndarray
from kornia.losses.ssim import SSIMLoss
from .pix2pix_model import Pix2PixModel
import torch
from skimage.exposure import rescale_intensity
import numpy as np
from skimage.metrics import structural_similarity
import math


def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)



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
        parser.set_defaults(dataset_mode='satellite',
                            direction='AtoB',
                            netG='unet_256',
                            gan_mode='lsgan',
                            norm='instance',
                            ngf='128',
                            ndf='128',
                            batch_size='12'
                           )
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
        self.metrics_names = ['ssim_value', 'issm_value', 'fsim_value', 'l1_value']
        self.loss_names = ['G_GAN',
                           #'G_SmoothL1',
                           'G_L1',
                           'G_SSIM',
                           #'G_ISSM',
                           #'G_FSIM',
                           'D_real',
                           'D_fake']

        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc

        if self.isTrain:
            self.criterionSSIM = SSIMLoss(5)
            #self.criterionSmoothL1 = torch.nn.SmoothL1Loss()

    def get_ndvi(self, multiband):
        red = multiband[:, :, 3]
        nir = multiband[:, :, 7]

        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir.astype(float) - red.astype(float)) / (nir + red)
        ndvi = self.byte_normalization(self.contrast_stretching(ndvi))

        return ndvi

    def get_ndvi_loss(self, tensor1, tensor2):
        image = tensor1.cpu().detach().numpy().transpose(0,2,3,1)[0, :, :, :]
        ndvi1 = self.get_ndvi(image)

        image = tensor2.cpu().detach().numpy().transpose(0,2,3,1)[0, :, :, :]
        ndvi2 = self.get_ndvi(image)

        loss = np.mean(np.abs(ndvi1 - ndvi2))
        return loss

    def byte_normalization(self, img):
        return ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')

    # Stretchs histogram to 95%
    def contrast_stretching(self, band: ndarray) -> ndarray:
        percentile_025 = np.percentile(band, 2.5)
        percentile_975 = np.percentile(band, 97.5)
        return rescale_intensity(band, in_range=(percentile_025, percentile_975))

    # Es muy lento
    def calc_FSIM(self, tensor1, tensor2):
        import image_similarity_measures
        from image_similarity_measures.quality_metrics import issm, ssim, fsim

        batch_size, _, _, _ = tensor1.shape
        total = 0
        for item in range(batch_size):
            image1 = tensor1.cpu().detach().numpy().transpose(0, 2, 3, 1)[item, :, :, :]
            image2 = tensor2.cpu().detach().numpy().transpose(0, 2, 3, 1)[item, :, :, :]
            total += fsim(self.byte_normalization(image1), self.byte_normalization(image2))

        return float(total / batch_size)

    def calc_ISSM(self, tensor1, tensor2):
        import image_similarity_measures
        from image_similarity_measures.quality_metrics import issm, ssim, fsim

        batch_size, _, _, _ = tensor1.shape
        total = 0
        for item in range(batch_size):
            image1 = tensor1.cpu().detach().numpy().transpose(0, 2, 3, 1)[item, :, :, :]
            image2 = tensor2.cpu().detach().numpy().transpose(0, 2, 3, 1)[item, :, :, :]
            part = issm(self.byte_normalization(image1), self.byte_normalization(image2))
            total += part

        return float(total / batch_size)

    def calc_SSIM(self, tensor1, tensor2):
        import image_similarity_measures
        from image_similarity_measures.quality_metrics import issm, ssim, fsim

        batch_size, _, _, _ = tensor1.shape
        total = 0
        for item in range(batch_size):
            image1 = tensor1.cpu().detach().numpy().transpose(0, 2, 3, 1)[item, :, :, :]
            image2 = tensor2.cpu().detach().numpy().transpose(0, 2, 3, 1)[item, :, :, :]
            #part = ssim(self.byte_normalization(image1), self.byte_normalization(image2))
            part = structural_similarity(self.byte_normalization(image1), self.byte_normalization(image2), channel_axis=2)
            total += part

        return float(total / batch_size)

    def calc_metrics(self):

        self.l1_value = self.criterionL1(self.real_B, self.fake_B)
        self.ssim_value = self.calc_SSIM(self.real_B, self.fake_B)
        self.issm_value = self.calc_ISSM(self.real_B, self.fake_B)
        self.fsim_value = self.calc_FSIM(self.real_B, self.fake_B)

        #print(f'SSIM: {ssim_value} \t ISSM: {issm_value} \t FSIM: {fsim_value}')

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B

        #self.loss_G_ISSM = (1 - self.calc_ISSM(self.real_B, self.fake_B))*1
        self.loss_G_SSIM = self.criterionSSIM(self.real_B, self.fake_B) * 4

        self.loss_G_L1 = self.criterionL1(self.real_B, self.fake_B) * self.opt.lambda_L1


        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_SSIM

        self.loss_G.backward()


    def sar2rgb(self, tensor):
        """Convert a Sentinel1 tensor image to a RGB numpy output
        Parameters:
            tensor:  2-channel numpy array (dB scale for VV and VH polarizations)

        Returns:
            rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
        """
        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')

        red = self.byte_normalization(self.contrast_stretching(tensor[:, :, 0]))
        green = self.byte_normalization(self.contrast_stretching(tensor[:, :, 1]))
        blue = self.byte_normalization(self.contrast_stretching(np.nan_to_num(red / green)))
        rgb_image = np.dstack((red.astype('int'), green.astype('int'), blue.astype('int')))
        return rgb_image



    def optical2rgb(self, tensor):
        """Convert a Sentinel2 tensor image to a RGB numpy output
        Parameters:
            tensor:  13-channel numpy array

        Returns:
            rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
        """

        if self.output_nc > 3:
            red = self.byte_normalization(self.contrast_stretching(tensor[:, :, 3]))
            green = self.byte_normalization(self.contrast_stretching(tensor[:, :, 2]))
            blue = self.byte_normalization(self.contrast_stretching(tensor[:, :, 1]))
        elif self.output_nc == 3:
            red = self.byte_normalization(self.contrast_stretching(tensor[:, :, 2]))
            green = self.byte_normalization(self.contrast_stretching(tensor[:, :, 1]))
            blue = self.byte_normalization(self.contrast_stretching(tensor[:,:,0]))
        else:
            raise RuntimeError('Wrong amount of output bands')

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
