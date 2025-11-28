import torch
from torch import nn, einsum
import torch.nn.functional as F
import sys
from einops import rearrange
from ..utils.utils_general import exists, default, eval_decorator
from .vae import VAEEncoder, VAEDecoder

# from utils import utils_distributed

class DVAEEncoder(nn.Module):
    def __init__(
        self,
        input_H = 256,              # Input image size
        input_W = 256,              # Input image size
        num_tokens = 512,           # Quantization levels, i.e. vocabulary size the image is encoded in a maximum of "num_tokens" ways
        codebook_dim = 512,         # Dimension of codebook vectors
        num_layers = 3,             # Number of layers / downsampling factor of 2^num_layers
        num_resnet_blocks = 0,      # Number of ResNet blocks per layer
        hidden_dim = 64,            # Base hidden dimension of the model
        channels = 3,               # Number of input channels,
        temperature = 0.9,          # Gumbel softmax temperature
        straight_through = False    # Whether to use straight through gradient estimation
    ):
        super().__init__()
        assert (input_H % (2**num_layers)) == 0 and (input_W % (2**num_layers)) == 0, 'input size has to be divisible by num_layers'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        
        self.temperature = temperature
        self.straight_through = straight_through
        self.num_tokens = num_tokens # this is the embedding dimension (dimension of each codebook vector)
        self.codebook_dim = codebook_dim # this is the embedding space size (how many codebooks we have)
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        # encoder / decoder modules
        self.encoder = VAEEncoder(
            in_channels=channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_resnet_blocks=num_resnet_blocks,
            num_tokens=num_tokens
        )

    def cosine_reconstruction_loss(self, target, rec):
        target = target / (target.norm(dim=-1, keepdim=True) + 1e-9)
        rec = rec / (rec.norm(dim=-1, keepdim=True) + 1e-9)
        return (1 - (target * rec).sum(-1)).mean()


    def norm(self, images):
        if not exists(self.normalization):
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images
    
    def forward(self, x):
        logits = self.encoder(x)
        soft_one_hot = F.gumbel_softmax(logits, 
                                        tau=self.temperature, 
                                        dim=1, 
                                        hard=self.straight_through
                                        ) # (B, 8192, H, W)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        return sampled, logits