from matplotlib import pyplot as plt
from models.DVAE import DVAE
from models.PatchDVAE import PatchDVAE
import torch
import json

from ev_loader.DSEC_dataloader.provider import DatasetProvider

if __name__ == '__main__':
    with open('configs/base.json', 'r') as f:
        config = json.load(f)

    provider = DatasetProvider(
        dataset_path=config["data"]["path"],
        representation=config["data"]["representation"],
        num_bins=config["data"]["voxel_bins"],
        delta_t_ms=config["data"]["event_dt_ms"]
    )
    train_dataset = provider.get_train_dataset()
    
    patch_vae = PatchDVAE(input_H=480, 
                          input_W=640, 
                          channels=2, 
                          num_layers=3, 
                          patch_grid_H=2, 
                          patch_grid_W=2, 
                          codebook_dim=64
                          )
    # img = torch.randn(4, 2, 256, 256) # batch of 4 images, 2 channels, 256x256
    img = train_dataset[0]["representation"]["left"].unsqueeze(0)  # Get a single sample and add batch dimension
    out = patch_vae(img)
    # loss, recons = patch_vae(img, return_loss = True, return_recons = True)
    # print(recons.shape)
    print(f"Original image shape: {img.shape}")
    print(f"Encoded tokens shape: {out.shape}")  # Expected shape: (4, 4, H', W') where H' and W' depend on the encoder output size
    