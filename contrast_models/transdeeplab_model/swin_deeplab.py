import torch
from torch import nn
import yaml
import sys
sys.path.append('/mnt/workspace/AgileFormer/contrast_models/')
from transdeeplab_model.encoder import build_encoder
from transdeeplab_model.decoder import build_decoder
from transdeeplab_model.aspp import build_aspp
import importlib
class SwinDeepLab(nn.Module):
    def __init__(self, num_classes=9,ds=False):
        super().__init__()
        self.ds=ds

     

        self.encoder = build_encoder(model_config.encoder_config)
        self.aspp = build_aspp(input_size=self.encoder.high_level_size,
                               input_dim=self.encoder.high_level_dim,
                               out_dim=self.encoder.low_level_dim, config=model_config.aspp_config)
        self.decoder = build_decoder(input_size=self.encoder.high_level_size,
                                     input_dim=self.encoder.low_level_dim,
                                     config= model_config.decoder_config)

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
