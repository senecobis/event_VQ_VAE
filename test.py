from matplotlib import pyplot as plt
from models.DiscreteVAE import DiscreteVAE
import torch

if __name__ == '__main__':
    vae = DiscreteVAE(input_H=256, input_W=256, channels=2, num_layers=3)
    img = torch.randn(10, 2, 256, 256)
    loss, recons = vae(img, return_loss = True, return_recons = True)
    print(recons.shape)