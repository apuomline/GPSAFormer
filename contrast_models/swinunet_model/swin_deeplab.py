import torch
from torch import nn

from model.encoder import build_encoder
from model.decoder import build_decoder
from model.aspp import build_aspp

class SwinDeepLab(nn.Module):
    def __init__(self, encoder_config, aspp_config, decoder_config):
        super().__init__()
        self.encoder = build_encoder(encoder_config)
        self.aspp = build_aspp(input_size=self.encoder.high_level_size,
                               input_dim=self.encoder.high_level_dim,
                               out_dim=self.encoder.low_level_dim, config=aspp_config)
        self.decoder = build_decoder(input_size=self.encoder.high_level_size,
                                     input_dim=self.encoder.low_level_dim,
                                     config=decoder_config)

    def run_encoder(self, x):
        low_level, high_level = self.encoder(x)
        return low_level, high_level
    
    def run_aspp(self, x):
        return self.aspp(x)

    def run_decoder(self, low_level, pyramid):
        return self.decoder(low_level, pyramid)

    def run_upsample(self, x):
        return self.upsample(x)

    def forward(self, x):
        low_level, high_level = self.run_encoder(x)
        x = self.run_aspp(high_level)
        x = self.run_decoder(low_level, x)
        
        return x


def get_transdeeplab(config_path):
    model_config = importlib.import_module(f'model.configs.{config_path}')
    net = SwinDeepLab(
        model_config.EncoderConfig, 
        model_config.ASPPConfig, 
        model_config.DecoderConfig
    )
    
    if model_config.EncoderConfig.encoder_name == 'swin' and model_config.EncoderConfig.load_pretrained:
        net.encoder.load_from('/mnt/workspace/Rolling-Unet-free-isic/swin_tiny_patch4_window7_224.pth')
    if model_config.ASPPConfig.aspp_name == 'swin' and model_config.ASPPConfig.load_pretrained:
        net.aspp.load_from('/mnt/workspace/Rolling-Unet-free-isic/swin_tiny_patch4_window7_224.pth')
    if model_config.DecoderConfig.decoder_name == 'swin' and model_config.DecoderConfig.load_pretrained and not model_config.DecoderConfig.extended_load:
        net.decoder.load_from('/mnt/workspace/Rolling-Unet-free-isic/swin_tiny_patch4_window7_224.pth')
    if model_config.DecoderConfig.decoder_name == 'swin' and model_config.DecoderConfig.load_pretrained and model_config.DecoderConfig.extended_load:
        net.decoder.load_from_extended('/mnt/workspace/Rolling-Unet-free-isic/swin_tiny_patch4_window7_224.pth')
    
    return net