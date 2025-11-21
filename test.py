from matplotlib import pyplot as plt
from models.DVAE import DVAE
from models.PatchDVAE import PatchDVAE
import torch

if __name__ == '__main__':
    # vae = DVAE(input_H=256, input_W=256, channels=2, num_layers=3)
    # img = torch.randn(10, 2, 256, 256) # batch of 10 images, 2 channels, 256x256
    # loss, recons = vae(img, return_loss = True, return_recons = True)
    # print(recons.shape)
    
    patch_vae = PatchDVAE(input_H=256, input_W=256, channels=2, num_layers=3, patch_grid_H=2, patch_grid_W=2, codebook_dim=64)
    img = torch.randn(4, 2, 256, 256) # batch of 4 images, 2 channels, 256x256
    out = patch_vae(img)
    # loss, recons = patch_vae(img, return_loss = True, return_recons = True)
    # print(recons.shape)
    print(f"Original image shape: {img.shape}")
    print(f"Encoded tokens shape: {out.shape}")  # Expected shape: (4, 4, H', W') where H' and W' depend on the encoder output size
    