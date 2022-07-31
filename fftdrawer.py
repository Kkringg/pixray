from DrawingInterface import DrawingInterface

from torchvision.utils import save_image

from aphantasia.image import to_valid_rgb, fft_image, dwt_image, pixel_image
import torch
from util import str2bool

# import tensorflow as tf

import numpy as np
import PIL.Image
from numpy.lib.function_base import copy
#from PIL import Image

# canonical interpolation function, like https://p5js.org/reference/#/p5/map
def map_number(n, start1, stop1, start2, stop2):
  return ((n-start1)/(stop1-start1))*(stop2-start2)+start2;

class FftDrawer(DrawingInterface):
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--fft_use",     type=str, help="use fft or dwt or pixel", default="fft", dest='fft_use')
        parser.add_argument('--fft_decay',   default=1.5, type=float, dest='fft_decay')
        parser.add_argument('--fft_wave',    default='coif2', help='wavelets: db[1..], coif[1..], haar, dmey', dest='fft_wave')
        parser.add_argument('--fft_sharp',   default=0.3, type=float, dest='fft_sharp')
        parser.add_argument('--fft_colors',  default=1.5, type=float, dest='fft_colors')
        parser.add_argument('--fft_lrate',   default=0.3, type=float, help='Learning rate', dest='fft_lrate')
        return parser

    def __init__(self, settings):
        super(DrawingInterface, self).__init__()
        self.canvas_width = settings.size[0]
        self.canvas_height = settings.size[1]
        self.fft_use = settings.fft_use
        self.decay = settings.fft_decay
        self.wave = settings.fft_wave
        self.sharp = settings.fft_sharp
        self.colors = settings.fft_colors
        self.lrate = settings.fft_lrate
        self.img = None

    def load_model(self, settings, device):
        self.device = device

    def get_opts(self):
        return self.opts

    def rand_init(self, toksX, toksY):
        self.init_from_tensor(None)

    def get_fft_noise(self, init_tensor):
        shape = [1, 3, self.canvas_height, self.canvas_width]
        resume = None
        params, image_f, sz = fft_image(shape, sd=0.01, decay_power=self.decay, resume=resume)
        return params
        #test_img = TF.to_pil_image(params.cpu())
        #imageio.imwrite('test_image.png', np.array(test_img))
        
    def init_from_tensor(self, init_tensor, cur_it):
        self.params, self.image_f = (
            self.encode_image(init_tensor, cur_it) 
        )
        
    def encode_image(self, init_tensor, cur_it):    
    # def init_from_tensor(self, init_tensor, cur_it):
        shape = [1, 3, self.canvas_height, self.canvas_width]
        resume = None
        if init_tensor is not None:
            if cur_it == 0:
                save_image(init_tensor, "res_init_1.png")
            if cur_it == 1:
                save_image(init_tensor, "res_init_2.png")
            save_image(init_tensor, "res_init.png")
            # resume = "res_init.png"
        if self.fft_use == "dwt":
            print("Using DWT instead of FFT")
            params, image_f, sz = dwt_image(shape, self.wave, self.sharp, self.colors, resume=resume)
        elif self.fft_use == "pixel":
            params, image_f, sz = pixel_image(shape, sd=1, resume=resume)
        elif self.fft_use == "fft":
            params, image_f, sz = fft_image(shape, sd=0.01, decay_power=self.decay, resume=resume)    
            # save_image(params, "params_example.png") 
        else:
            raise ValueError(f"fft drawer does not know how to apply fft_use={self.fft_use}")
        # self.params = params
        # self.image_f = to_valid_rgb(image_f, colors=1.5)
        
        image_f = to_valid_rgb(image_f, colors=1.5)
        return params, image_f
        
    def get_opts(self, decay_divisor=1):
        # Optimizers
        optimizer = torch.optim.Adam(self.params, self.lrate / decay_divisor)
        self.opts = [optimizer]
        return self.opts

    def reapply_from_tensor(self, new_tensor, cur_it):
        # self.init_from_tensor(new_tensor, cur_it)
        params, *_ = self.encode_image(new_tensor, cur_it)
        for old_param, new_param in zip(self.params, params):
            old_param.data = new_param.data

    def get_z_from_tensor(self, ref_tensor):
        return None

    def get_num_resolutions(self):
        return None

    def synth(self, cur_iteration):
        if cur_iteration < 0:
            return self.img

        img = self.image_f(contrast=0.9)
        #img = self.image_f(contrast=1.0)
        self.img = img
        return img

    @torch.no_grad()
    def to_image(self):
        img = self.img.detach().cpu().numpy()[0]
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        img = np.uint8(img * 255)
        pimg = PIL.Image.fromarray(img, mode="RGB")
        return pimg

    def clip_z(self):
        pass

    def get_z(self):
        return None

    def get_z_copy(self):
        return None

    def set_z(self, new_z):
        return None

    @torch.no_grad()
    def to_svg(self):
        pass
