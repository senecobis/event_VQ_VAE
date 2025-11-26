import json
import os
import wandb  # <--- IMPORT WANDB
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
        wandb_config: dict = None,  
        project_name: str = "implict-kernel" 
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
        
        self.model = DDP(
            self.model, 
            device_ids=[self.local_rank], 
            output_device=self.local_rank,
            find_unused_parameters=False 
        )

        # 3. Prepare Data Loader
        self.sampler = DistributedSampler(dataset, shuffle=True)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=self.sampler,
            num_workers=n_workers
        )

        # 4. Optimizer & Scaler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scaler = GradScaler(enabled=use_amp)

        # ### WANDB INITIALIZATION ###
        if self.is_main_process:
            # Model name for saving
            now = datetime.now()
            run_id = now.strftime("%Y-%m-%d_%H-%M")
            self.model_name = f"dvae_{run_id}"

            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"Trainer initialized on {torch.cuda.device_count()} GPUs.")

            wandb.init(
                project=project_name,
                name=self.model_name,
                config=wandb_config,
                # reinit=True
            )
            # Optional: Watch gradients (can slow down training slightly)
            # wandb.watch(self.model, log="all")
    
    def _setup_ddp(self):
        """Initializes the distributed process group."""
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        
        torch.cuda.set_device(self.local_rank)
        init_process_group(backend="nccl")
        
    @property
    def is_main_process(self):
        return self.rank == 0

    def save_checkpoint(self, epoch, loss):
        if not self.is_main_process:
            return
            
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }
        path = os.path.join(self.checkpoint_dir, f"{self.model_name}" ,f"epoch_{epoch}.pth")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        # ### WANDB LOGGING ###
        # Log the artifact (optional, but good for version control)
        # wandb.save(path)

    def train_epoch(self, epoch):
        self.model.train()
        self.sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        iterator = tqdm(self.dataloader, desc=f"Epoch {epoch}") if self.is_main_process else self.dataloader

        # We keep track of steps to log to wandb frequently
        step_count = 0

        for batch in iterator:
            event_imgs = batch["representation"]["left"]
            event_imgs = event_imgs.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                # Ensure your model forward returns the loss when return_loss=True
                loss = self.model(event_imgs, return_loss=True)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            current_loss = loss.item()
            epoch_loss += current_loss
            
            if self.is_main_process:
                # Update tqdm
                iterator.set_postfix({"loss": current_loss})
                
                # ### WANDB LOGGING ###
                # Log batch loss (and Learning Rate if using a scheduler)
                wandb.log({
                    "batch_train_loss": current_loss, 
                    "epoch": epoch,
                    "lr": self.optimizer.param_groups[0]['lr']
                })

            step_count += 1

        return epoch_loss / len(self.dataloader)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            avg_loss = self.train_epoch(epoch)
            
            if self.is_main_process:
                print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}\n")
                
                # ### WANDB LOGGING ###
                wandb.log({
                    "epoch_train_loss": avg_loss,
                    "epoch": epoch
                })
            
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch, avg_loss)
                
        self._cleanup()

    def _cleanup(self):
        if self.is_main_process:
            wandb.finish() # ### WANDB FINISH ###
        destroy_process_group()


def main():
    config_dir = 'configs/base.json'
    
    with open(config_dir, 'r') as f:
        file_config = json.load(f)
        
    provider = DatasetProvider(
        dataset_path=file_config["data"]["path"],
        representation=file_config["data"]["representation"],
        num_bins=file_config["data"]["voxel_bins"],
        delta_t_ms=file_config["data"]["event_dt_ms"]
    )
    dataset = provider.get_train_dataset(load_opt_flow=False)

    # 3. Initialize Model
    model = PatchDVAE(
        input_H = file_config["data"]["height"],
        input_W = file_config["data"]["width"],
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
        patch_grid_H=2,
        patch_grid_W=2,
    )

    # ### PREPARE WANDB CONFIG ###
    # Consolidate parameters to pass to wandb.config
    wandb_config = {
        "batch_size": file_config["data"]["batch_size"],
        "lr": file_config["optimizer"]["lr"],
        "epochs": file_config["optimizer"]["epochs"],
        "image_height": file_config["data"]["height"],
        "image_width": file_config["data"]["width"],
        "model_architecture": "PatchDVAE",
        "dataset_config": file_config
    }

    # 4. Initialize Trainer and Fit
    trainer = PatchDVAETrainer(
        model=model,
        dataset=dataset,
        batch_size=file_config["data"]["batch_size"],
        learning_rate=file_config["optimizer"]["lr"],
        num_epochs=file_config["optimizer"]["epochs"],
        n_workers=8,
        save_every=10,
        checkpoint_dir="checkpoints",
        wandb_config=wandb_config,  # Pass config
        project_name="implicit-kernel" # Name your project
    )
    
    trainer.train()

if __name__ == "__main__":
    main()