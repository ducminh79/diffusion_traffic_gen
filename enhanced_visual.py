import numpy as np
import os, sys
import torch
from utils.loggers import CompositeLogger, NeptuneLogger, PrintLogger
from utils.utils_args import parse_args_uncond
from models.model import ImagenTime
from models.sampler import DiffusionProcess
import logging
from utils.utils_data import gen_dataloader
from utils.utils import create_model_name_and_dir, restore_state, log_config_and_tags
from utils.utils_vis import prepare_data, PCA_plot, TSNE_plot, density_plot, jensen_shannon_divergence
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import pandas as pd
import json
from datetime import datetime
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
matplotlib.rcParams.update({'font.size': 12})

def calculate_original_stats(original_data):
    """
    Calculate statistics from the original data for denormalization
    
    Args:
        original_data: numpy array of original time series data
        
    Returns:
        dict containing min, max, mean, std of original data
    """
    stats = {
        'min_val': float(np.min(original_data)),
        'max_val': float(np.max(original_data)), 
        'mean': float(np.mean(original_data)),
        'std': float(np.std(original_data))
    }
    
    logging.info(f"Calculated original data statistics:")
    logging.info(f"  Min: {stats['min_val']:.6f}")
    logging.info(f"  Max: {stats['max_val']:.6f}")  
    logging.info(f"  Mean: {stats['mean']:.6f}")
    logging.info(f"  Std: {stats['std']:.6f}")
    
    return stats

def denormalize_data(data, original_stats, method='minmax'):
    """
    Denormalize data back to original scale using statistics from original data
    
    Args:
        data: normalized data array to denormalize
        original_stats: dict containing min, max, mean, std from original data
        method: 'minmax' or 'standard' normalization method
    
    Returns:
        denormalized data
    """
    if method == 'minmax':
        # Assumes data was normalized to [0,1] range: (data - min) / (max - min)
        # Denormalization: data * (max - min) + min
        denorm_data = data * (original_stats['max_val'] - original_stats['min_val']) + original_stats['min_val']
        logging.info(f"Min-Max denormalization applied")
    elif method == 'standard':
        # Assumes data was standardized: (data - mean) / std  
        # Denormalization: data * std + mean
        denorm_data = data * original_stats['std'] + original_stats['mean']
        logging.info(f"Standard denormalization applied")
    else:
        logging.warning(f"Unknown denormalization method '{method}'. Returning data as-is.")
        return data
    
    logging.info(f"Denormalized data from [{data.min():.4f}, {data.max():.4f}] to [{denorm_data.min():.4f}, {denorm_data.max():.4f}]")
    return denorm_data

def save_time_series_data(original_data, generated_data, save_dir, args):
    """
    Save original and generated time series data to various formats
    
    Args:
        original_data: numpy array of original time series
        generated_data: numpy array of generated time series
        save_dir: directory to save files
        args: arguments object containing dataset info
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as numpy arrays
    np.save(os.path.join(save_dir, 'original_timeseries.npy'), original_data)
    np.save(os.path.join(save_dir, 'generated_timeseries.npy'), generated_data)
    
    # Save as CSV files for easy analysis
    # Reshape data if it's multidimensional
    if len(original_data.shape) > 2:
        orig_reshaped = original_data.reshape(original_data.shape[0], -1)
        gen_reshaped = generated_data.reshape(generated_data.shape[0], -1)
    else:
        orig_reshaped = original_data
        gen_reshaped = generated_data
    
    # Create column names
    num_features = orig_reshaped.shape[1] if len(orig_reshaped.shape) > 1 else 1
    if num_features == 1:
        columns = ['value']
    else:
        columns = [f'feature_{i}' for i in range(num_features)]
    
    # Save to CSV
    pd.DataFrame(orig_reshaped, columns=columns).to_csv(
        os.path.join(save_dir, 'original_timeseries.csv'), index=False
    )
    pd.DataFrame(gen_reshaped, columns=columns).to_csv(
        os.path.join(save_dir, 'generated_timeseries.csv'), index=False
    )
    
    # Save metadata
    metadata = {
        'dataset': args.dataset,
        'timestamp': datetime.now().isoformat(),
        'original_shape': original_data.shape,
        'generated_shape': generated_data.shape,
        'num_samples': len(original_data),
        'sequence_length': original_data.shape[1] if len(original_data.shape) > 1 else 1,
        'num_features': num_features
    }
    
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Data saved to {save_dir}")
    logging.info(f"Original data shape: {original_data.shape}")
    logging.info(f"Generated data shape: {generated_data.shape}")

def plot_sample_comparisons(original_data, generated_data, save_dir, args, num_samples=5):
    """
    Plot side-by-side comparisons of original and generated time series samples
    
    Args:
        original_data: numpy array of original time series
        generated_data: numpy array of generated time series
        save_dir: directory to save plots
        args: arguments object
        num_samples: number of sample pairs to plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine data dimensions
    if len(original_data.shape) == 1:
        # 1D time series
        sequence_length = len(original_data)
        num_features = 1
        original_data = original_data.reshape(1, -1)
        generated_data = generated_data.reshape(1, -1)
    elif len(original_data.shape) == 2:
        # 2D: (samples, sequence_length) or (samples, features)
        num_features = 1 if original_data.shape[1] < 50 else original_data.shape[1]  # heuristic
        sequence_length = original_data.shape[1]
    else:
        # 3D: (samples, sequence_length, features)
        num_features = original_data.shape[2]
        sequence_length = original_data.shape[1]
        # Reshape for plotting
        original_data = original_data.reshape(original_data.shape[0], -1)
        generated_data = generated_data.reshape(generated_data.shape[0], -1)
    
    num_samples = min(num_samples, len(original_data))
    
    # Plot individual sample comparisons
    for i in range(num_samples):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original data
        axes[0].plot(original_data[i], color='blue', linewidth=1.5, alpha=0.8)
        axes[0].set_title(f'Original Time Series - Sample {i+1}')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Generated data
        axes[1].plot(generated_data[i], color='red', linewidth=1.5, alpha=0.8)
        axes[1].set_title(f'Generated Time Series - Sample {i+1}')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Value')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'comparison_sample_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Saved {num_samples} individual comparison plots")

def plot_overlay_comparisons(original_data, generated_data, save_dir, args, num_samples=5):
    """
    Plot overlayed comparisons of original and generated time series
    
    Args:
        original_data: numpy array of original time series
        generated_data: numpy array of generated time series  
        save_dir: directory to save plots
        args: arguments object
        num_samples: number of samples to overlay
    """
    num_samples = min(num_samples, len(original_data))
    
    # Single overlay plot
    plt.figure(figsize=(12, 6))
    
    for i in range(num_samples):
        plt.plot(original_data[i], color='blue', alpha=0.6, linewidth=1, 
                label='Original' if i == 0 else "")
        plt.plot(generated_data[i], color='red', alpha=0.6, linewidth=1, 
                label='Generated' if i == 0 else "")
    
    plt.title(f'Original vs Generated Time Series Overlay ({num_samples} samples)')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overlay_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Saved overlay comparison plot")

def plot_statistical_comparisons(original_data, generated_data, save_dir, args):
    """
    Plot statistical comparisons between original and generated data
    
    Args:
        original_data: numpy array of original time series
        generated_data: numpy array of generated time series
        save_dir: directory to save plots
        args: arguments object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Flatten data for statistical analysis
    orig_flat = original_data.flatten()
    gen_flat = generated_data.flatten()
    
    # 1. Value distribution comparison
    axes[0, 0].hist(orig_flat, bins=50, alpha=0.7, color='blue', label='Original', density=True)
    axes[0, 0].hist(gen_flat, bins=50, alpha=0.7, color='red', label='Generated', density=True)
    axes[0, 0].set_title('Value Distribution Comparison')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    data_for_box = [orig_flat, gen_flat]
    labels = ['Original', 'Generated']
    bp = axes[0, 1].boxplot(data_for_box, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    axes[0, 1].set_title('Value Distribution Box Plot')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Mean and std comparison across samples
    orig_means = np.mean(original_data, axis=1) if len(original_data.shape) > 1 else [np.mean(original_data)]
    gen_means = np.mean(generated_data, axis=1) if len(generated_data.shape) > 1 else [np.mean(generated_data)]
    
    axes[1, 0].scatter(range(len(orig_means)), orig_means, alpha=0.6, color='blue', label='Original')
    axes[1, 0].scatter(range(len(gen_means)), gen_means, alpha=0.6, color='red', label='Generated')
    axes[1, 0].set_title('Mean Values per Sample')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Correlation between original and generated means
    min_len = min(len(orig_means), len(gen_means))
    if min_len > 1:
        axes[1, 1].scatter(orig_means[:min_len], gen_means[:min_len], alpha=0.6, color='purple')
        axes[1, 1].plot([min(orig_means[:min_len]), max(orig_means[:min_len])], 
                       [min(orig_means[:min_len]), max(orig_means[:min_len])], 
                       'k--', alpha=0.5, label='Perfect Correlation')
        axes[1, 1].set_title('Original vs Generated Means Correlation')
        axes[1, 1].set_xlabel('Original Mean')
        axes[1, 1].set_ylabel('Generated Mean')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statistical_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Saved statistical comparison plots")

def main(args):
    with CompositeLogger([NeptuneLogger()]) if args.neptune \
            else PrintLogger() as logger:
        name = create_model_name_and_dir(args)
        log_config_and_tags(args, logger, name)
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        train_loader, test_loader = gen_dataloader(args)
        model = ImagenTime(args=args, device=args.device).to(args.device)
        
        if args.use_stft:
            model.init_stft_embedder(train_loader)
        else:
            # Fixed syntax error in original code
            _ = model.ts_to_img(next(iter(train_loader))[0].to(args.device))  # initialize delay embedder
        
        # restore checkpoint
        state = dict(model=model, epoch=0)
        ema_model = model.model_ema if args.ema else None
        restore_state(args, state, ema_model=ema_model)
        
        gen_sig = []
        real_sig = []
        model.eval()
        
        with torch.no_grad():
            with model.ema_scope():
                process = DiffusionProcess(args, model.net,
                                         (args.input_channels, args.img_resolution, args.img_resolution))
                for data in tqdm(test_loader, desc="Generating time series"):
                    # sample from the model
                    x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                    # --- convert to time series --
                    x_ts = model.img_to_ts(x_img_sampled)
                    # special case for temperature_rain dataset
                    if args.dataset in ['temperature_rain']:
                        x_ts = torch.clamp(x_ts, 0, 1)
                    gen_sig.append(x_ts.detach().cpu().numpy())
                    real_sig.append(data[0].detach().cpu().numpy())
        
        gen_sig = np.vstack(gen_sig)
        ori_sig = np.vstack(real_sig)
        
        logging.info("Data generation is complete")
        logging.info(f"Original data shape: {ori_sig.shape}")
        logging.info(f"Generated data shape: {gen_sig.shape}")
        
        # ===== NEW ENHANCEMENT: DENORMALIZATION =====
        # You'll need to modify this section based on how your data was normalized
        # during training. Here's a template:
        
        # Example normalization parameters (replace with actual values from your training)
        normalization_params = {
            # For min-max normalization:
            # 'min_val': your_training_min_value,
            # 'max_val': your_training_max_value,
            # For standard normalization:
            # 'mean': your_training_mean,
            # 'std': your_training_std
        }
        
        # Denormalize if parameters are available
        if normalization_params and any(param in normalization_params for param in ['min_val', 'max_val', 'mean', 'std']):
            ori_sig_denorm = denormalize_data(ori_sig, args.dataset, normalization_params)
            gen_sig_denorm = denormalize_data(gen_sig, args.dataset, normalization_params)
        else:
            logging.warning("No normalization parameters found. Using data as-is.")
            ori_sig_denorm = ori_sig
            gen_sig_denorm = gen_sig
        
        # ===== NEW ENHANCEMENT: SAVE DATA =====
        save_dir = os.path.join(args.output_dir if hasattr(args, 'output_dir') else 'output', 
                               f'{name}_results')
        save_time_series_data(ori_sig_denorm, gen_sig_denorm, save_dir, args)
        
        # ===== NEW ENHANCEMENT: CREATE LINE GRAPHS =====
        plot_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot sample comparisons
        plot_sample_comparisons(ori_sig_denorm, gen_sig_denorm, plot_dir, args, num_samples=5)
        
        # Plot overlay comparisons  
        plot_overlay_comparisons(ori_sig_denorm, gen_sig_denorm, plot_dir, args, num_samples=3)
        
        # Plot statistical comparisons
        plot_statistical_comparisons(ori_sig_denorm, gen_sig_denorm, plot_dir, args)
        
        logging.info(f"All plots saved to {plot_dir}")
        
        # ===== ORIGINAL ANALYSIS (STILL INCLUDED) =====
        prep_ori, prep_gen, sample_num = prepare_data(ori_sig, gen_sig)
        # PCA Analysis
        PCA_plot(prep_ori, prep_gen, sample_num, logger, args)
        # Do t-SNE Analysis together
        TSNE_plot(prep_ori, prep_gen, sample_num, logger, args)
        # Density plot
        density_plot(prep_ori, prep_gen, logger, args)
        # jensen shannon divergence
        jensen_shannon_divergence(prep_ori, prep_gen, logger)

if __name__ == '__main__':  # Fixed syntax error
    args = parse_args_uncond()  # load unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)