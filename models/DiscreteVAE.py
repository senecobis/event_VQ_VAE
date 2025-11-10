import torch
from torch import nn, einsum
import torch.nn.functional as F
import sys
from einops import rearrange
from .utils.utils_general import exists, default, eval_decorator
from .base.vae import VAEEncoder, VAEDecoder

# from utils import utils_distributed

class DiscreteVAE(nn.Module):
    def __init__(
        self,
        input_H = 256,              # Input image size
        input_W = 256,              # Input image size
        num_tokens = 512,           # Quantization levels, i.e. vocabulary size the image is encoded in a maximum of "num_tokens" ways
        codebook_dim = 512,         # Dimension of codebook vectors
        num_layers = 3,             # Number of layers / downsampling factor of 2^num_layers
        num_resnet_blocks = 0,      # Number of ResNet blocks per layer
        hidden_dim = 64,            # Base hidden dimension of the model
        channels = 3,               # Number of input channels
        loss = 'mse',               # Reconstruction loss to use: 'mse', 'smooth_l1', 'cosine'
        temperature = 0.9,          # Gumbel softmax temperature
        straight_through = False,   # Whether to use straight through gradient estimation
        kl_div_loss_weight = 0.,    # Weight of the KL divergence loss
        normalization = None        # Tuple of channel-wise mean and std to use for normalization, e.g. ([0.5]*3, [0.5]*3) for imagenet pre-trained models
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

        # self._register_external_parameters()

    def cosine_reconstruction_loss(self, target, rec):
        target = target / (target.norm(dim=-1, keepdim=True) + 1e-9)
        rec = rec / (rec.norm(dim=-1, keepdim=True) + 1e-9)
        return (1 - (target * rec).sum(-1)).mean()

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (
                not utils_distributed.is_distributed
                or not utils_distributed.using_backend(
                    utils_distributed.DeepSpeedBackend)
        ):
            return

        deepspeed = utils_distributed.backend.backend_module
        deepspeed.zero.register_external_parameter(self, self.codebook.weight)

    def norm(self, images):
        if not exists(self.normalization):
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    @torch.no_grad()
    def get_grad_norm(self):
        norm = 0
        for _, p in self.named_parameters():
            try:
                norm += torch.linalg.norm(p.grad.detach().data).item()**2
            except:
                pass
        return norm**0.5

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits = True)
        codebook_indices = logits.argmax(dim = 1).flatten(1)
        return codebook_indices

    def decode(
        self,
        img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = int(self.input_H / 2**self.num_layers)
        w = int(self.input_W / 2**self.num_layers)

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

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

        logits = self.encoder(img)

        if return_logits:
            return logits

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau = temp, dim = 1, hard = self.straight_through) # (B, 8192, H, W)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
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

# TODO: finish implementing this
@torch.no_grad()
def evaluate(data_loader, model, device):
    sys.path.append("../")
    sys.path.append("../../")
    import mem.utils as utils

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    # log with codebook indecies are used
    codebook_indices_test = set([])

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            logits = model(images, return_logits = True)
            codebook_indices = logits.argmax(dim = 1).flatten(1)

            # log the used codebook indecies
            codebook_indices_test.update(torch.unique(codebook_indices).tolist())

            output = model.decode(codebook_indices)
            recon_loss = model.loss_fn(images, output)

            # kl divergence
            logits = rearrange(logits, 'b n h w -> b (h w) n')
            log_qy = F.log_softmax(logits, dim = -1)
            log_uniform = torch.log(torch.tensor([1. / model.num_tokens], device = device))
            kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)

        loss = recon_loss + (kl_div * model.kl_div_loss_weight)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['loss'].update(loss.item(), n=batch_size)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f'* Loss: {loss.item()}, Codebook indices: {len(codebook_indices_test)}')

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats[f'codebook_indices'] = len(codebook_indices_test)
    return stats
        