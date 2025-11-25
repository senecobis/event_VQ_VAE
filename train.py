import json
import os
from tqdm import tqdm
from datetime import datetime
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.cuda.amp import GradScaler, autocast

from ev_loader.DSEC_dataloader.provider import DatasetProvider
from models.PatchDVAE import PatchDVAE

class PatchDVAETrainer:
    def __init__(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        save_every: int = 10,
        use_amp: bool = True,
        n_workers: int = 4,
        checkpoint_dir: str = "checkpoints",
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp
        
        # 1. Setup Distributed Environment
        self._setup_ddp()
        
        # 2. Move Model to GPU and Wrap in DDP
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.model = model.to(self.device)
        
        # DDP Wrapper
        # find_unused_parameters=True is often needed if some patches/encoders 
        # aren't activated in every forward pass, though likely False is fine here.
        self.model = DDP(
            self.model, 
            device_ids=[self.local_rank], 
            output_device=self.local_rank,
            find_unused_parameters=False 
        )

        # 3. Prepare Data Loader with DistributedSampler
        self.sampler = DistributedSampler(dataset, shuffle=True)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False, # Sampler handles shuffling
            sampler=self.sampler,
            num_workers=n_workers
        )

        # 4. Optimizer & Scaler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scaler = GradScaler(enabled=use_amp)

        if self.is_main_process:
            # Model name for saving
            now = datetime.now()
            run_id = now.strftime("%Y-%m-%d_%H-%M")
            self.model_name = f"dvae_{run_id}"

            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"Trainer initialized on {torch.cuda.device_count()} GPUs.")
    
    def _setup_ddp(self):
        """Initializes the distributed process group."""
        # torchrun sets these env variables
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        
        torch.cuda.set_device(self.local_rank)
        init_process_group(backend="nccl")
        
    @property
    def is_main_process(self):
        return self.rank == 0

    def save_checkpoint(self, epoch, loss):
        """Saves model state. Only called by main process."""
        if not self.is_main_process:
            return
            
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict(), # Note .module for DDP
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }
        path = os.path.join(self.checkpoint_dir, f"{self.model_name}" ,f"epoch_{epoch}.pth")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def train_epoch(self, epoch):
        self.model.train()
        # Important: Set epoch for sampler to ensure shuffling changes every epoch
        self.sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        # Only show progress bar on main process
        iterator = tqdm(self.dataloader, desc=f"Epoch {epoch}") if self.is_main_process else self.dataloader

        for batch in iterator:
            # Handle case where dataset returns (img, label) or just img
            event_imgs = batch["representation"]["left"]
            event_imgs = event_imgs.to(self.device)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast(enabled=self.use_amp):
                # forward returns scalar loss because return_loss=True
                loss = self.model(event_imgs, return_loss=True)

            # Backward Pass
            self.scaler.scale(loss).backward()
            
            # Gradient Clipping (Optional but recommended for VAEs)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            
            if self.is_main_process:
                iterator.set_postfix({"loss": loss.item()})

        return epoch_loss / len(self.dataloader)

    def train(self):
        """Main training loop."""
        for epoch in range(1, self.num_epochs + 1):
            avg_loss = self.train_epoch(epoch)
            
            if self.is_main_process:
                print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}\n")
            
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch, avg_loss)
                
        self._cleanup()

    def _cleanup(self):
        destroy_process_group()


def main():
    # 1. Define Hyperparameters
    H, W = 256, 256
    BATCH_SIZE = 64
    config_dir = 'configs/base.json'
    
    # Open the real dataset
    with open(config_dir, 'r') as f:
        config = json.load(f)
        
    provider = DatasetProvider(
        dataset_path=config["data"]["path"],
        representation=config["data"]["representation"],
        num_bins=config["data"]["voxel_bins"],
        delta_t_ms=config["data"]["event_dt_ms"]
    )
    dataset = provider.get_train_dataset(load_opt_flow=False)

    # 2. Create Dummy Dataset (Replace with real data)
    # Example: CIFAR or ImageFolder
    # transform = transforms.Compose([
    #     transforms.Resize((H, W)),
    #     transforms.ToTensor(),
    # ])
    # dataset = datasets.FakeData(size=1000, image_size=(2, H, W), transform=transform)

    # 3. Initialize Model
    model = PatchDVAE(
        input_H = H,
        input_W = W,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        num_resnet_blocks = 0,
        hidden_dim = 64,
        channels = 2,
        loss = 'mse',
        temperature = 0.9,
        straight_through = False,
        kl_div_loss_weight = 0.,
        normalization = None,
        patch_grid_H=2,             # Number of patches along height (e.g., 2)
        patch_grid_W=2,             # Number of patches along width (e.g., 2)
    )

    # 4. Initialize Trainer and Fit
    trainer = PatchDVAETrainer(
        model=model,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_epochs=50,
        save_every=1,
        checkpoint_dir="checkpoints"
    )
    
    trainer.train()

if __name__ == "__main__":
    main()