from torch import nn


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class VAEEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int, num_resnet_blocks: int, num_tokens: int):
        super().__init__()
        layers = []
        c_in = in_channels
        # downsampling conv blocks
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(c_in, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            ))
            c_in = hidden_dim
        
        # optional ResBlocks at bottleneck
        for _ in range(num_resnet_blocks):
            layers.append(ResBlock(hidden_dim))
        
        # produce logits over codebook tokens
        layers.append(nn.Conv2d(hidden_dim, num_tokens, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# New decoder module
class VAEDecoder(nn.Module):
    def __init__(self, codebook_dim: int, out_channels: int, hidden_dim: int, num_layers: int, num_resnet_blocks: int):
        super().__init__()
        layers = []
        c_in = codebook_dim

        # If using ResBlocks, map to hidden_dim first then apply them at the bottleneck
        if num_resnet_blocks > 0:
            layers.append(nn.Conv2d(codebook_dim, hidden_dim, kernel_size=1))
            for _ in range(num_resnet_blocks):
                layers.append(ResBlock(hidden_dim))
            c_in = hidden_dim

        # upsampling deconv blocks
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            ))
            c_in = hidden_dim

        # final 1x1 to output channels
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

