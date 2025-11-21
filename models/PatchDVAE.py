import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from .DVAEEncoder import DVAEEncoder
from .DVAE import DVAE

class PatchDVAE(DVAE):
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
        normalization = None,
        patch_grid_H=2,             # Number of patches along height (e.g., 2)
        patch_grid_W=2,             # Number of patches along width (e.g., 2)
    ):
        super().__init__(
            input_H=input_H,
            input_W=input_W,
            num_tokens=num_tokens,
            codebook_dim=codebook_dim,
            num_layers=num_layers,
            num_resnet_blocks=num_resnet_blocks,
            hidden_dim=hidden_dim,
            channels=channels,
            loss=loss,
            temperature=temperature,
            straight_through=straight_through,
            kl_div_loss_weight=kl_div_loss_weight,
            normalization=normalization
        )
        assert patch_grid_H >= 1 and patch_grid_W >= 1, 'number of patches must be greater than or equal to 1'
        assert (input_H % patch_grid_H) == 0 and (input_W % patch_grid_W) == 0, 'input size has to be divisible by patch grid size'

        self.patch_grid_H = patch_grid_H
        self.patch_grid_W = patch_grid_W

        # 1. Multiple Encoders (One for each patch position)
        # Total encoders = patch_grid_H * patch_grid_W
        self.num_patches = patch_grid_H * patch_grid_W
        self.encoders = nn.ModuleList([
            DVAEEncoder(
                input_H = input_H // patch_grid_H,
                input_W = input_W // patch_grid_W,
                num_tokens = num_tokens,
                codebook_dim = codebook_dim,
                num_layers = num_layers,
                num_resnet_blocks = num_resnet_blocks,
                hidden_dim = hidden_dim,
                channels = channels,
                temperature=temperature,
                straight_through=straight_through
            ) for _ in range(self.num_patches)
        ])

    # TODO vectorize the encoding
    def encode_patches(self, img):
            """
            Splits image, runs specific encoders, and stitches logits back.
            """
            # Split image into patches: (B, C, H, W) -> (B, num_patches, C, patch_H, patch_W)
            # We use einops to handle the grid splitting cleanly
            patches = rearrange(
                img, 
                'b c (h p1) (w p2) -> b (h w) c p1 p2', 
                h=self.patch_grid_H, 
                w=self.patch_grid_W
            )

            logits_list = []
            
            # Iterate through patches and their corresponding encoders
            for i in range(self.num_patches):
                # Extract the specific patch i for the whole batch
                patch_input = patches[:, i, ...] 
                
                # Pass through the i-th encoder
                # Output shape: (B, num_tokens, h_latent, w_latent)
                patch_logits = self.encoders[i](patch_input)
                logits_list.append(patch_logits)

            # Stack logits: (B, num_patches, num_tokens, h_latent, w_latent)
            stacked_logits = torch.stack(logits_list, dim=1)

            # Stitch logits back to full spatial resolution
            # (B, (h w), n, lh, lw) -> (B, n, (h lh), (w lw))
            full_logits = rearrange(
                stacked_logits,
                'b (h w) n lh lw -> b n (h lh) (w lw)',
                h=self.patch_grid_H,
                w=self.patch_grid_W
            )
            
            return full_logits
    
    def forward( 
        self,
        img,
        return_loss = False,
        return_recons = False,
        return_logits = False,
        temp = None
    ):
        device, num_tokens, input_H, input_W, kl_div_loss_weight = img.device, self.num_tokens, self.input_H, self.input_W, self.kl_div_loss_weight
        assert img.shape[-1] == input_W and img.shape[-2] == input_H, f'input must have the correct image size {input_H}x{input_W}, but is ({img.shape[-2]},{img.shape[-1]})'

        img = self.norm(img)

        sampled = self.encode_patches(img)
        out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss
        recon_loss = self.loss_fn(img, out)

        # kl divergence
        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim = -1)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device = device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out        
    
