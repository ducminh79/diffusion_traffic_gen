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
import matplotlib
from tqdm import tqdm
from datetime import datetime
from reverse_preprocess import get_original_data_with_params, reverse_preprocess 

from export_to_csv import export_separate_csv, export_timeseries_to_csv

  

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# matplotlib.use('Agg')


def main(args):
    with CompositeLogger([NeptuneLogger()]) if args.neptune \
            else PrintLogger() as logger:

        name = create_model_name_and_dir(args)
        log_config_and_tags(args, logger, name)
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader = gen_dataloader(args) # combine train and test loaders?
        model = ImagenTime(args=args, device=args.device).to(args.device)
        if args.use_stft:
            model.init_stft_embedder(train_loader)
        else:
            _ = model.ts_to_img(next(iter(train_loader))[0].to(args.device)) # initialize delay embedder

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
                for data in tqdm(test_loader):
                    # sample from the model
                    x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                    # --- convert to time series --
                    x_ts = model.img_to_ts(x_img_sampled)

                    # special case for temperature_rain dataset
                    if args.dataset in ['temperature_rain']:
                        x_ts = torch.clamp(x_ts, 0, 1)

                    # special case for network_traffic dataset
                    if args.dataset in ['network_traffic']:
                        # Simple hard clamp to [0,1] for single feature
                        x_ts = torch.clamp(x_ts, 0, 1)

                    gen_sig.append(x_ts.detach().cpu().numpy())
                    real_sig.append(data[0].detach().cpu().numpy())

        gen_sig = np.vstack(gen_sig)

        # Save generated sequences for event-based reconstruction
        np.save('generated_sequences.npy', gen_sig)
        logging.info(f"Saved generated sequences: {gen_sig.shape}")

        ori_sig = np.vstack(real_sig)
        export_separate_csv(ori_sig, gen_sig, "network_traffic_data", denormalize=False)
        logging.info("Data generation is complete")
        prep_ori, prep_gen, sample_num = prepare_data(ori_sig, gen_sig)

        # Create output directory for saving plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"visuals/{args.dataset}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Saving visualizations to: {save_dir}")
        # print(prep_gen.shape)
        # print(prep_gen)
          # Get the original CSV data directly (not from dataloader)
        from utils.utils_data import real_data_loading

        # Load the raw sequences (before dataloader shuffling)
        raw_sequences = real_data_loading("network_traffic", args.seq_len)

        # Get normalization parameters
        original_data, min_vals, max_vals = get_original_data_with_params(
            "network_traffic",
            "data/ultra_long_range/youtube_traffic_bytes_only.csv"
        )

        # Reconstruct from the raw consecutive sequences (not shuffled dataloader output)
        reconstructed_original = reverse_preprocess(raw_sequences, (min_vals, max_vals))
        print("Reconstructed original data shape:", reconstructed_original.shape)
        print("Original CSV shape:", original_data.shape)
        print("First 5 rows of reconstructed original data:")
        print(reconstructed_original[:5])
        print("First 5 rows of original CSV:")
        print(original_data[:5])

        # Also reconstruct the generated data using the same approach
        # Convert gen_sig (vstack of generated sequences) to list format
        gen_sig_as_list = [gen_sig[i] for i in range(gen_sig.shape[0])]
        reconstructed_generated = reverse_preprocess(gen_sig_as_list, (min_vals, max_vals))
        print("\nReconstructed generated data shape:", reconstructed_generated.shape)
        print("First 5 rows of reconstructed generated data:")
        print(reconstructed_generated[:5])

        # Save both reconstructed datasets to CSV files
        import pandas as pd

        # Determine number of features based on shape
        num_features = reconstructed_original.shape[1] if len(reconstructed_original.shape) > 1 else 1

        # Create DataFrames with proper column names based on number of features
        if num_features == 1:
            # Single feature: only bytes_per_interval
            df_original = pd.DataFrame(reconstructed_original, columns=['bytes_per_interval'])
            df_generated = pd.DataFrame(reconstructed_generated, columns=['bytes_per_interval'])
        else:
            # Two features: since_last_packet and bytes_per_interval
            df_original = pd.DataFrame(reconstructed_original, columns=['since_last_packet', 'bytes_per_interval'])
            df_generated = pd.DataFrame(reconstructed_generated, columns=['since_last_packet', 'bytes_per_interval'])

        # Save to CSV files in the same directory as visualizations
        original_csv_path = f"{save_dir}/reconstructed_original_data.csv"
        generated_csv_path = f"{save_dir}/reconstructed_generated_data.csv"

        df_original.to_csv(original_csv_path, index=False)
        df_generated.to_csv(generated_csv_path, index=False)

        print(f"\n✓ Saved reconstructed original data to: {original_csv_path}")
        print(f"✓ Saved reconstructed generated data to: {generated_csv_path}")

        # Create time series plot: bytes_per_interval vs cumulative time
        import matplotlib.pyplot as plt

        if num_features == 1:
            # For single feature, use index as x-axis
            bytes_original = reconstructed_original.flatten()
            bytes_generated = reconstructed_generated.flatten()

            plt.figure(figsize=(12, 6))
            plt.plot(bytes_original, label='Original Data', alpha=0.7, linewidth=1)
            plt.plot(bytes_generated, label='Generated Data', alpha=0.7, linewidth=1)

            plt.xlabel('Sample Index')
            plt.ylabel('Bytes per Interval')
            plt.title('Netflix Traffic: Bytes per Interval')
        else:
            # Calculate cumulative time (sum of since_last_packet intervals)
            cumulative_time_original = np.cumsum(reconstructed_original[:, 0])
            cumulative_time_generated = np.cumsum(reconstructed_generated[:, 0])

            # Get bytes_per_interval values
            bytes_original = reconstructed_original[:, 1]
            bytes_generated = reconstructed_generated[:, 1]

            # Create the plot
            plt.figure(figsize=(12, 6))
            plt.plot(cumulative_time_original, bytes_original, label='Original Data', alpha=0.7, linewidth=1)
            plt.plot(cumulative_time_generated, bytes_generated, label='Generated Data', alpha=0.7, linewidth=1)

            plt.xlabel('Cumulative Time (sum of since_last_packet)')
            plt.ylabel('Bytes per Interval')
            plt.title('Netflix Traffic: Bytes per Interval vs Time')

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the plot
        timeseries_plot_path = f"{save_dir}/time_series_comparison.png"
        plt.savefig(timeseries_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved time series plot to: {timeseries_plot_path}")

        np.savetxt("output.txt", ori_sig.reshape(-1, 1))
        np.savetxt("output.txt", ori_sig.reshape(ori_sig.shape[0], -1))
        with open("output.txt", "w") as f:
            for i, row in enumerate(ori_sig):
                f.write(f"Row {i}\n")
                np.savetxt(f, row, fmt="%.6f")
                f.write("\n")

        # PCA Analysis
        PCA_plot(prep_ori, prep_gen, sample_num, logger, args, save_path=save_dir)
        # Do t-SNE Analysis together
        TSNE_plot(prep_ori, prep_gen, sample_num, logger, args, save_path=save_dir)
        # Density plot
        density_plot(prep_ori, prep_gen, logger, args, save_path=save_dir)
        # jensen shannon divergence
        jensen_shannon_divergence(prep_ori, prep_gen, logger)


if __name__ == '__main__':
    args = parse_args_uncond()  # load unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
