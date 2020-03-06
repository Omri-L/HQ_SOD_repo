from DKN.models import *
from LRP.unet_upsampled_model import *


class SOD_model(nn.Module):
    def __init__(self,
                 fdkn_kernel_size,
                 fdkn_filter_size,
                 device='cpu',
                 lres_model_params=None,
                 fdkn_model_params=None,
                 scale=8):
        super(SOD_model, self).__init__()
        self.device = device
        self.scale = scale

        # low-resolution predictor module
        self.LR_SOD_module = UNet16_upsampled(1, 64, True, True)

        # joint filter refinement module
        self.FDKN_module = FDKN(kernel_size=fdkn_kernel_size,
                                filter_size=fdkn_filter_size,
                                residual=True, device=device)

        # load pre-trained models
        if lres_model_params is not None:
            self.LR_SOD_module.load_state_dict(torch.load(lres_model_params))

        if fdkn_model_params is not None:
            self.FDKN_module.load_state_dict(torch.load(fdkn_model_params))

        # not in use - TODO: remove it
        self.upscale = nn.Upsample(scale_factor=self.scale, mode='bilinear')
        self.upscale_sig = nn.Upsample(scale_factor=self.scale, mode='bilinear')

    def forward(self, x):
        low_res_upscaled, low_res_upscaled_sigmoid, low_res_sigmoid = self.LR_SOD_module(x)
        output, output_sig = self.FDKN_module((x, low_res_upscaled))
        return output, output_sig, low_res_upscaled, low_res_upscaled_sigmoid, low_res_sigmoid
