import numpy as np
import torch

def event_based_preprocessing(data, seq_len):
    """
    Preprocessing specifically designed for event-based data like network traffic

    Args:
        data: Raw data with [since_last_packet, bytes_per_interval] columns
        seq_len: Sequence length for windowing

    Returns:
        preprocessed_data: Better normalized and structured data for event-based modeling
    """

    # Separate the two different types of features
    time_intervals = data[:, 0]  # since_last_packet
    byte_counts = data[:, 1]     # bytes_per_interval

    # 1. Log-transform time intervals to handle wide range and zeros
    # Add small epsilon to avoid log(0)
    log_time_intervals = np.log1p(time_intervals)  # log(1 + x) handles zeros better

    # 2. Different normalization for each feature type
    # Time intervals: normalize log-transformed values
    time_min, time_max = log_time_intervals.min(), log_time_intervals.max()
    norm_time = (log_time_intervals - time_min) / (time_max - time_min + 1e-8)

    # Byte counts: use robust scaling (less sensitive to outliers)
    byte_median = np.median(byte_counts)
    byte_mad = np.median(np.abs(byte_counts - byte_median))  # Median Absolute Deviation
    norm_bytes = (byte_counts - byte_median) / (byte_mad + 1e-8)
    norm_bytes = np.clip(norm_bytes, -5, 5)  # Clip extreme outliers
    norm_bytes = (norm_bytes + 5) / 10  # Scale to [0, 1]

    # 3. Combine normalized features
    normalized_data = np.column_stack([norm_time, norm_bytes])

    # 4. Create sequences with overlapping windows (like original)
    sequences = []
    for i in range(len(normalized_data) - seq_len + 1):
        sequences.append(normalized_data[i:i + seq_len])

    return sequences, {
        'time_min': time_min, 'time_max': time_max,
        'byte_median': byte_median, 'byte_mad': byte_mad
    }

def reverse_event_preprocessing(sequences, params):
    """
    Reverse the event-based preprocessing to get back original scale
    """
    # Reconstruct continuous data (average overlapping windows)
    num_sequences, seq_len, num_features = np.array(sequences).shape
    total_length = num_sequences + seq_len - 1
    reconstructed = np.zeros((total_length, num_features))
    count_matrix = np.zeros((total_length, num_features))

    for i, seq in enumerate(sequences):
        start_idx = i
        end_idx = i + seq_len
        reconstructed[start_idx:end_idx] += seq
        count_matrix[start_idx:end_idx] += 1

    reconstructed = reconstructed / (count_matrix + 1e-8)

    # Reverse normalization for time intervals
    norm_time = reconstructed[:, 0]
    time_intervals = norm_time * (params['time_max'] - params['time_min']) + params['time_min']
    time_intervals = np.expm1(time_intervals)  # Reverse log1p

    # Reverse normalization for byte counts
    norm_bytes = reconstructed[:, 1]
    byte_scaled = (norm_bytes * 10) - 5  # Reverse [0,1] -> [-5,5]
    byte_counts = byte_scaled * params['byte_mad'] + params['byte_median']

    return np.column_stack([time_intervals, byte_counts])

# Example usage for updating your data loading
def load_event_based_data(csv_path, seq_len):
    """
    Load and preprocess event-based data
    """
    # Load raw data
    raw_data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    # Apply event-based preprocessing
    sequences, params = event_based_preprocessing(raw_data, seq_len)

    return sequences, params