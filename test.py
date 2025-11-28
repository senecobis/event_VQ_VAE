import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from ev_loader.DSEC_dataloader.provider import DatasetProvider
from models.PatchDVAE import PatchDVAE

class PatchDVAETester:
    def __init__(
        self, 
        model, 
        dataset, 
        checkpoint_path, 
        device="cuda", 
        output_dir="results",
        wandb_run_path=None # Optional: link to existing wandb run
    ):
        self.model = model
        self.dataset = dataset
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.wandb_run_path = wandb_run_path
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1, # Usually better to test with batch_size 1 for visualization
            shuffle=False, 
            num_workers=4
        )
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the model weights
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self):
        print(f"Loading checkpoint from {self.checkpoint_path}...")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle cases where the model was trained with DDP (keys start with "module.")
        # but we are testing on a single GPU.
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                name = k[7:] # remove "module."
            else:
                name = k
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict)
        print("Model loaded successfully.")

    def compute_psnr(self, mse):
        if mse == 0:
            return 100
        # Assuming data is normalized roughly 0-1, max_val is 1.
        return 10 * torch.log10(1 / mse)

    def event_to_rgb(self, event_tensor):
        """
        Helper to visualize 2-channel event voxel grid (Polarity/Time bins).
        Input: (C, H, W) where C=2 usually.
        Output: (3, H, W) RGB tensor.
        Logic: Channel 0 -> Red, Channel 1 -> Blue.
        """
        c, h, w = event_tensor.shape
        rgb = torch.zeros((3, h, w), device=event_tensor.device)
        
        # Normalize roughly for visualization (events can be sparse or dense)
        # We take absolute value and clip.
        img = event_tensor.abs()
        img = img / (img.max() + 1e-6) # Normalize 0-1 locally per image
        
        # Map Channel 0 to Red
        if c >= 1:
            rgb[0] = img[0]
        # Map Channel 1 to Blue
        if c >= 2:
            rgb[2] = img[1]
            
        return rgb

    def test(self, num_visualizations=10):
        print("Starting Evaluation...")
        total_mse = 0.0
        total_psnr = 0.0
        count = 0
        
        # For saving images
        images_to_log = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataloader, desc="Testing")):
                # 1. Prepare Data
                # Note: DSEC loader returns dictionary
                original = batch["representation"]["left"].to(self.device)
                
                # 2. Forward Pass (Reconstruction)
                # DVAE returns just the reconstruction if return_loss=False
                reconstruction = self.model(original, return_loss=False)
                
                # 3. Calculate Metrics
                # Ensure sizes match (sometimes padding happens in VAEs)
                if original.shape != reconstruction.shape:
                    reconstruction = F.interpolate(reconstruction, size=original.shape[-2:])

                mse = F.mse_loss(reconstruction, original)
                psnr = self.compute_psnr(mse)
                
                total_mse += mse.item()
                total_psnr += psnr.item()
                count += 1
                
                # 4. Visualization (Save first N images)
                if i < num_visualizations:
                    # Convert first item in batch to RGB
                    orig_vis = self.event_to_rgb(original[0])
                    recon_vis = self.event_to_rgb(reconstruction[0])
                    
                    # Stack vertically: Top = Original, Bottom = Reconstruction
                    comparison = torch.cat([orig_vis, recon_vis], dim=1) # Concatenate along height
                    
                    # Save to disk
                    save_path = os.path.join(self.output_dir, f"recon_{i}.png")
                    self.save_image(comparison, save_path)
                    
                    # Add to wandb list
                    images_to_log.append(wandb.Image(comparison, caption=f"Top: GT, Bot: Recon (Sample {i})"))

        avg_mse = total_mse / count
        avg_psnr = total_psnr / count
        
        print(f"\n--- Results ---")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Visualizations saved to: {self.output_dir}")

        # If wandb is active
        if wandb.run is not None:
            wandb.log({
                "test_mse": avg_mse,
                "test_psnr": avg_psnr,
                "reconstructions": images_to_log
            })

    def save_image(self, tensor, path):
        # tensor is (3, H, W). Convert to numpy (H, W, 3) for matplotlib
        img_np = tensor.cpu().numpy().transpose(1, 2, 0)
        
        # Clamp to 0-1
        img_np = np.clip(img_np, 0, 1)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.axis('off')
        plt.title("Top: Original | Bottom: Recon")
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()


def main():
    CHECKPOINT_PATH = "checkpoints/dvae_2025-11-25_17-05/epoch_50.pth" 
    CONFIG_PATH = 'configs/base.json'
    
    with open(CONFIG_PATH, 'r') as f:
        file_config = json.load(f)

    provider = DatasetProvider(
        dataset_path=file_config["data"]["path"],
        representation=file_config["data"]["representation"],
        num_bins=file_config["data"]["voxel_bins"],
        delta_t_ms=file_config["data"]["event_dt_ms"]
    )
    test_dataset = ConcatDataset(provider.get_test_dataset())

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
        normalization = None,
        patch_grid_H=2,
        patch_grid_W=2,
    )

    # 5. Optional: Init WandB for logging test results
    wandb.init(project="implicit-kernel", name="TEST_dvae_2025-11-25_17-05", job_type="test")

    # 6. Run Tester
    tester = PatchDVAETester(
        model=model,
        dataset=test_dataset,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="test_results"
    )
    
    tester.test(num_visualizations=20)
    
    wandb.finish()

if __name__ == "__main__":
    main()