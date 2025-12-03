#!/usr/bin/env python3
"""
Standalone script to generate synthetic time series data from a trained ImagenTime model.

Usage:
    python generate_synthetic_data.py --config ./configs/unconditional/<dataset>.yaml --checkpoint_path ./logs/<model_dir>/checkpoint.pth --num_samples 1000 --output_path ./generated_data.npy

Requirements:
    - A trained model checkpoint
    - The same config file used for training
"""

import os
import sys
import torch
import numpy as np
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import ImagenTime
from models.sampler import DiffusionProcess
from utils.utils import restore_checkpoint
from utils.utils_args import parse_args_uncond
from omegaconf import OmegaConf


def parse_generation_args():
    """Parse arguments specific to generation"""
    parser = argparse.ArgumentParser(description="Generate synthetic time series data")

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file used for training (same as training config)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)",
    )

    # Generation parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for generation (adjust based on GPU memory)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./generated_samples.npy",
        help="Path to save generated samples (.npy file)",
    )

    # Optional parameters
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def load_model_and_config(config_path, checkpoint_path, device):
    """Load trained model from checkpoint"""

    # Load configuration
    config = OmegaConf.load(config_path)

    # Create args namespace from config (similar to training script)
    args = argparse.Namespace(**config)
    args.device = device
    args.resume = True
    args.log_dir = os.path.dirname(checkpoint_path)

    # Add default diffusion parameters if missing
    if not hasattr(args, "beta1"):
        args.beta1 = 1e-5
    if not hasattr(args, "betaT"):
        args.betaT = 1e-2
    if not hasattr(args, "deterministic"):
        args.deterministic = False

    # Initialize model
    model = ImagenTime(args=args, device=device).to(device)

    # Load STFT embedder if needed
    if hasattr(args, "use_stft") and args.use_stft:
        # For generation, we don't need the actual dataloader, just initialize empty
        logging.warning(
            "STFT embedder initialization skipped for generation. "
            "Ensure the checkpoint contains the embedder state."
        )

    # Initialize img_shape for delay embedding transformations
    # This is needed for proper img_to_ts conversion
    if hasattr(model.ts_img, "img_shape") and model.ts_img.img_shape is None:
        # Set a default img_shape based on the model parameters
        # This will be overridden during the actual generation process
        batch_size = 1
        features = 6
        embedding = 8

        # Calculate approximate number of columns based on sequence length and delay
        seq_len = 24
        delay = 3
        approx_cols = (seq_len + delay - 1) // delay

        model.ts_img.img_shape = (batch_size, features, embedding, approx_cols)
        logging.info(f"Initialized img_shape: {model.ts_img.img_shape}")

    # Load checkpoint
    state = dict(model=model, epoch=0)
    checkpoint_dir = checkpoint_path

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load the checkpoint
    loaded_state = torch.load(checkpoint_path, map_location=device)
    state["epoch"] = loaded_state["epoch"]
    model.load_state_dict(loaded_state["model"], strict=False)

    # Load EMA model if available
    if (
        "ema_model" in loaded_state
        and hasattr(model, "model_ema")
        and model.model_ema is not None
    ):
        model.model_ema.load_state_dict(loaded_state["ema_model"])
        logging.info("Loaded EMA model weights")

    logging.info(f"Successfully loaded model from epoch {state['epoch']}")

    # Ensure correct training config values after model loading
    if hasattr(model, 'ts_img') and hasattr(model.ts_img, 'embedding'):
        model.ts_img.embedding = 8
        model.ts_img.delay = 3
        model.ts_img.seq_len = 24
        logging.info(f"Corrected model parameters: embedding={model.ts_img.embedding}")

    return model, args


def generate_samples(model, args, num_samples, batch_size, device):
    """Generate synthetic samples using the trained model"""

    model.eval()
    generated_samples = []

    # Calculate number of batches needed
    num_batches = (num_samples + batch_size - 1) // batch_size

    logging.info(f"Generating {num_samples} samples in {num_batches} batches...")

    with torch.no_grad():
        with model.ema_scope() if hasattr(model, "ema_scope") else torch.no_grad():

            # Initialize diffusion process
            process = DiffusionProcess(
                args,
                model.net,
                (args.input_channels, args.img_resolution, args.img_resolution),
            )

            for batch_idx in range(num_batches):
                # Calculate actual batch size for this iteration
                current_batch_size = min(
                    batch_size, num_samples - batch_idx * batch_size
                )

                logging.info(
                    f"Generating batch {batch_idx + 1}/{num_batches} "
                    f"(size: {current_batch_size})"
                )

                # Generate samples in image space
                x_img_sampled = process.sampling(sampling_number=current_batch_size)

                # For delay embedding, we need to set the img_shape before conversion
                # This is normally set during ts_to_img but we're generating directly in image space
                if hasattr(model.ts_img, "img_shape"):
                    batch_size, channels, height, width = x_img_sampled.shape

                    # Use hardcoded config parameters from training
                    features = 6  # input_channels
                    embedding_dim = 8  # embedding dimension from config

                    # Hardcoded config values from training
                    seq_len = 24
                    delay = 3

                    # Calculate theoretical number of windows needed
                    theoretical_cols = (seq_len + delay - 1) // delay

                    # Use the minimum of theoretical and actual width (since image might be padded)
                    actual_cols = min(theoretical_cols, width)

                    # Set the img_shape using actual generated image dimensions
                    model.ts_img.img_shape = (
                        batch_size,
                        features,
                        embedding_dim,
                        actual_cols,
                    )

                    logging.info(f"Generated image shape: {x_img_sampled.shape}")
                    logging.info(f"Set img_shape: {model.ts_img.img_shape}")
                    logging.info(f"Config params - seq_len: {seq_len}, delay: {delay}")
                    logging.info(
                        f"Theoretical cols needed: {theoretical_cols}, using: {actual_cols}"
                    )

                # Convert back to time series (embedding should already be correct from model initialization)
                x_ts = model.img_to_ts(x_img_sampled)

                # Apply dataset-specific postprocessing if needed
                if hasattr(args, "dataset") and args.dataset in ["temperature_rain"]:
                    x_ts = torch.clamp(x_ts, 0, 1)

                # Store samples
                generated_samples.append(x_ts.detach().cpu().numpy())

    # Concatenate all batches
    generated_samples = np.vstack(generated_samples)

    logging.info(
        f"Generated {generated_samples.shape[0]} samples of shape {generated_samples.shape[1:]}"
    )

    return generated_samples


def main():
    # Parse arguments
    gen_args = parse_generation_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Set random seed
    torch.manual_seed(gen_args.seed)
    np.random.seed(gen_args.seed)

    # Determine device
    if gen_args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = gen_args.device

    logging.info(f"Using device: {device}")

    try:
        # Load model and configuration
        logging.info("Loading model and configuration...")
        model, args = load_model_and_config(
            gen_args.config, gen_args.checkpoint_path, device
        )

        # Generate samples
        logging.info("Starting sample generation...")
        generated_samples = generate_samples(
            model, args, gen_args.num_samples, gen_args.batch_size, device
        )

        # Save generated samples
        output_path = Path(gen_args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path, generated_samples)
        logging.info(f"Generated samples saved to: {output_path}")
        logging.info(f"Sample shape: {generated_samples.shape}")

        # Print some statistics
        logging.info(f"Sample statistics:")
        logging.info(f"  Mean: {generated_samples.mean():.4f}")
        logging.info(f"  Std: {generated_samples.std():.4f}")
        logging.info(f"  Min: {generated_samples.min():.4f}")
        logging.info(f"  Max: {generated_samples.max():.4f}")

    except Exception as e:
        logging.error(f"Error during generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
