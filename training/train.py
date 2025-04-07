import os
import torch
from torch.utils.data import DataLoader
import argparse
from training.dataset import DualConditioningDataset
from model.sd_model import StableDiffusionWrapper
from training.trainer import DualConditioningTrainer
import sys
sys.path.append(".")
from config import CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description="Train Stable Diffusion with dual conditioning")
    parser.add_argument("--data_dir", type=str, default=CONFIG["train_data_dir"],
                        help="Directory with training data")
    parser.add_argument("--output_dir", type=str, default=CONFIG["output_dir"],
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"],
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=CONFIG["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional checkpoint to resume from")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Update config with command line arguments
    CONFIG["train_data_dir"] = args.data_dir
    CONFIG["output_dir"] = args.output_dir
    CONFIG["epochs"] = args.epochs
    CONFIG["batch_size"] = args.batch_size
    CONFIG["learning_rate"] = args.learning_rate
    
    # Create output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Create dataset
    dataset = DualConditioningDataset(
        data_dir=CONFIG["train_data_dir"],
        image_size=CONFIG["image_size"]
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0  # Set to 0 for simpler debugging
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Steps per epoch: {len(dataloader)}")
    
    # Create SD model
    sd_model = StableDiffusionWrapper(CONFIG)
    
    # Create trainer
    trainer = DualConditioningTrainer(sd_model, CONFIG)
    
    # Load checkpoint if specified
    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train model
    trainer.train(
        train_dataloader=dataloader,
        num_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["learning_rate"]
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()