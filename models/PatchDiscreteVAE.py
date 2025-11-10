import torch
from torch import nn, einsum
import torch.nn.functional as F
import sys
from einops import rearrange
from .utils.utils_general import exists, default, eval_decorator
from .base.vae import VAEEncoder, VAEDecoder

class PatchDiscreteVAE(nn.Module):
    def __init__(
        self,
        input_H = 256,
        input_W = 256,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        num_resnet_blocks = 0,
        hidden_dim = 64,
        channels = 3,
        loss = 'mse',
        temperature = 0.9,
        straight_through = False,
        kl_div_loss_weight = 0.,
        normalization = None
    ):
        super().__init__()
        assert (input_H % (2**num_layers)) == 0 and (input_W % (2**num_layers)) == 0, 'input size has to be divisible by num_layers'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'

        self.input_H = input_H
        self.input_W = input_W
        self.input_size = (input_H, input_W)
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        # encoder / decoder modules
        self.encoder = VAEEncoder(
            in_channels=channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_resnet_blocks=num_resnet_blocks,
            num_tokens=num_tokens
        )
        self.decoder = VAEDecoder(
            codebook_dim=codebook_dim,
            out_channels=channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_resnet_blocks=num_resnet_blocks
        )

        loss_functions = {
            'mse': F.mse_loss,
            'smooth_l1': F.smooth_l1_loss,
            'cosine': self.cosine_reconstruction_loss
        }
        self.loss_fn = loss_functions[loss]
        self.kl_div_loss_weight = kl_div_loss_weight

        # take care of normalization within class
        self.normalization = normalization
