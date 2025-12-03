import numpy as np
import pandas as pd

def denormalize_minmax(normalized_data, data_min, data_max):
    """
    Reverse MinMax normalization: norm_data * (max - min) + min

    Args:
        normalized_data: Data normalized to [0, 1] range
        data_min: Original minimum values (per feature)
        data_max: Original maximum values (per feature)

    Returns:
        Denormalized data in original scale
    """
    return normalized_data * (data_max - data_min) + data_min

def export_timeseries_to_csv(ori_sig, gen_sig, output_path="timeseries_data.csv",
                           ori_min=None, ori_max=None, denormalize=True):
    """
    Export original and generated time series data to CSV format

    Args:
        ori_sig: Original signals, shape [batch, seq_len, features]
        gen_sig: Generated signals, shape [batch, seq_len, features]
        output_path: Path to save CSV file
        ori_min: Original data minimum values for denormalization (optional)
        ori_max: Original data maximum values for denormalization (optional)
        denormalize: Whether to denormalize the data
    """

    # Denormalize if requested and parameters provided
    if denormalize and ori_min is not None and ori_max is not None:
        ori_sig = denormalize_minmax(ori_sig, ori_min, ori_max)
        gen_sig = denormalize_minmax(gen_sig, ori_min, ori_max)
        print("Data denormalized to original scale")

    # Get dimensions
    batch_size, seq_len, num_features = ori_sig.shape

    # Reshape data: flatten batch and time dimensions
    ori_flat = ori_sig.reshape(-1, num_features)  # Shape: [batch*seq_len, features]
    gen_flat = gen_sig.reshape(-1, num_features)  # Shape: [batch*seq_len, features]

    # Create feature column names
    feature_cols = [f'feature_{i}' for i in range(num_features)]

    # Create DataFrames without sample_id and timestep
    df_orig = pd.DataFrame(ori_flat, columns=feature_cols)
    df_gen = pd.DataFrame(gen_flat, columns=feature_cols)

    # Add data type column to distinguish original vs generated
    df_orig['data_type'] = 'original'
    df_gen['data_type'] = 'generated'

    # Combine and save
    df_combined = pd.concat([df_orig, df_gen], ignore_index=True)
    df_combined.to_csv(output_path, index=False)

    print(f"Data exported to {output_path}")
    print(f"Shape: {df_combined.shape}")
    print(f"Original samples: {len(df_orig)}, Generated samples: {len(df_gen)}")

    return df_combined

def export_separate_csv(ori_sig, gen_sig, prefix="timeseries",
                       ori_min=None, ori_max=None, denormalize=True, flatten_timesteps=True):
    """
    Export to separate CSV files for original and generated data

    Args:
        ori_sig: Original signals, shape [batch, seq_len, features]
        gen_sig: Generated signals, shape [batch, seq_len, features]
        prefix: Filename prefix
        ori_min: Original data minimum values for denormalization (optional)
        ori_max: Original data maximum values for denormalization (optional)
        denormalize: Whether to denormalize the data
        flatten_timesteps: If True, each timestep becomes a row. If False, each sample becomes a row.
    """

    # Denormalize if requested and parameters provided
    if denormalize and ori_min is not None and ori_max is not None:
        ori_sig = denormalize_minmax(ori_sig, ori_min, ori_max)
        gen_sig = denormalize_minmax(gen_sig, ori_min, ori_max)
        print("Data denormalized to original scale")

    batch_size, seq_len, num_features = ori_sig.shape

    if flatten_timesteps:
        # Each timestep becomes a separate row: [batch*seq_len, features]
        ori_2d = ori_sig.reshape(-1, num_features)
        gen_2d = gen_sig.reshape(-1, num_features)
        columns = [f'feature_{i}' for i in range(num_features)]
        print(f"Flattened format: Each timestep as separate row")
    else:
        # Each sample becomes one row: [batch, seq_len*features]
        ori_2d = ori_sig.reshape(batch_size, -1)
        gen_2d = gen_sig.reshape(batch_size, -1)
        # Create column names like: feature_0_t0, feature_0_t1, ..., feature_1_t0, etc.
        columns = []
        for f in range(num_features):
            for t in range(seq_len):
                columns.append(f'feature_{f}_t{t}')
        print(f"Time series format: Each sample as one row with {len(columns)} columns")

    # Create DataFrames
    df_orig = pd.DataFrame(ori_2d, columns=columns)
    df_gen = pd.DataFrame(gen_2d, columns=columns)

    # Save files
    orig_path = f"{prefix}_original.csv"
    gen_path = f"{prefix}_generated.csv"

    df_orig.to_csv(orig_path, index=False)
    df_gen.to_csv(gen_path, index=False)

    print(f"Original data: {orig_path} ({df_orig.shape})")
    print(f"Generated data: {gen_path} ({df_gen.shape})")

    return df_orig, df_gen

def export_aggregated_csv(ori_sig, gen_sig, prefix="timeseries",
                         ori_min=None, ori_max=None, denormalize=True,
                         aggregation_method='mean'):
    """
    Export aggregated time series to match original CSV shape [batch, features]

    Args:
        ori_sig: Original signals, shape [batch, seq_len, features]
        gen_sig: Generated signals, shape [batch, seq_len, features]
        prefix: Filename prefix
        ori_min: Original data minimum values for denormalization (optional)
        ori_max: Original data maximum values for denormalization (optional)
        denormalize: Whether to denormalize the data
        aggregation_method: 'mean', 'last', 'first', 'max', 'min', 'median'
    """

    # Denormalize if requested and parameters provided
    if denormalize and ori_min is not None and ori_max is not None:
        ori_sig = denormalize_minmax(ori_sig, ori_min, ori_max)
        gen_sig = denormalize_minmax(gen_sig, ori_min, ori_max)
        print("Data denormalized to original scale")

    batch_size, seq_len, num_features = ori_sig.shape

    # Aggregate along time dimension
    if aggregation_method == 'mean':
        ori_agg = np.mean(ori_sig, axis=1)  # [batch, features]
        gen_agg = np.mean(gen_sig, axis=1)
    elif aggregation_method == 'last':
        ori_agg = ori_sig[:, -1, :]  # Take last timestep
        gen_agg = gen_sig[:, -1, :]
    elif aggregation_method == 'first':
        ori_agg = ori_sig[:, 0, :]  # Take first timestep
        gen_agg = gen_sig[:, 0, :]
    elif aggregation_method == 'max':
        ori_agg = np.max(ori_sig, axis=1)
        gen_agg = np.max(gen_sig, axis=1)
    elif aggregation_method == 'min':
        ori_agg = np.min(ori_sig, axis=1)
        gen_agg = np.min(gen_sig, axis=1)
    elif aggregation_method == 'median':
        ori_agg = np.median(ori_sig, axis=1)
        gen_agg = np.median(gen_sig, axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    # Create column names
    columns = [f'feature_{i}' for i in range(num_features)]

    # Create DataFrames
    df_orig = pd.DataFrame(ori_agg, columns=columns)
    df_gen = pd.DataFrame(gen_agg, columns=columns)

    # Save files
    orig_path = f"{prefix}_original_{aggregation_method}.csv"
    gen_path = f"{prefix}_generated_{aggregation_method}.csv"

    df_orig.to_csv(orig_path, index=False)
    df_gen.to_csv(gen_path, index=False)

    print(f"Original data: {orig_path} ({df_orig.shape}) - {aggregation_method} aggregated")
    print(f"Generated data: {gen_path} ({df_gen.shape}) - {aggregation_method} aggregated")

    return df_orig, df_gen

def export_simple_csv(gen_sig, output_path="synthetic_stock_data.csv",
                     ori_min=None, ori_max=None, denormalize=True):
    """
    Export generated data as simple CSV matching original format.
    Treats each timestep as an individual stock data point.

    Args:
        gen_sig: Generated signals, shape [batch, seq_len, features]
        output_path: Output CSV path
        ori_min: Original minimum values for denormalization
        ori_max: Original maximum values for denormalization
        denormalize: Whether to denormalize
    """

    # Denormalize if requested
    if denormalize and ori_min is not None and ori_max is not None:
        gen_sig = denormalize_minmax(gen_sig, ori_min, ori_max)
        print("Data denormalized to original scale")

    # Flatten: each timestep becomes one data point
    # Shape: [batch*seq_len, features]
    gen_flat = gen_sig.reshape(-1, gen_sig.shape[-1])

    # Create DataFrame with stock feature names (optional)
    columns = [f'feature_{i}' for i in range(gen_sig.shape[-1])]
    # Or use stock names: columns = ['price', 'volume', 'open', 'close', 'high', 'low']

    df = pd.DataFrame(gen_flat, columns=columns)
    df.to_csv(output_path, index=False)

    print(f"Synthetic stock data saved to: {output_path}")
    print(f"Shape: {df.shape} (should plot similarly to your original data)")

    return df

# Example usage:
if __name__ == "__main__":
    # Load your data (replace with actual loading)
    # ori_sig = np.load("original_signals.npy")  # shape: [batch, seq_len, features]
    # gen_sig = np.load("generated_signals.npy") # shape: [batch, seq_len, features]

    # Method 1: Combined CSV
    # export_timeseries_to_csv(ori_sig, gen_sig, "combined_timeseries.csv")

    # Method 2: Separate CSV files
    # export_separate_csv(ori_sig, gen_sig, "stock_timeseries")

    print("Use the functions above with your actual data")