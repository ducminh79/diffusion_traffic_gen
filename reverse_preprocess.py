import numpy as np


def reverse_preprocess(processed_data, original_min_max_params=None):
    """
    Reverse the preprocessing steps to reconstruct original CSV data.

    Args:
        processed_data: List of sequences from real_data_loading
        original_min_max_params: Tuple of (min_vals, max_vals) from original normalization
                                If None, will estimate from the data (less accurate)

    Returns:
        reconstructed_data: Original data in CSV format (2D array)
    """
    # Convert to numpy array if it's a list
    if isinstance(processed_data, list):
        processed_data = np.array(processed_data)

    # Get dimensions
    num_sequences, seq_len, num_features = processed_data.shape

    # Step 1: Reconstruct the sliding window sequences back to continuous data
    # Since we used sliding windows, we need to average overlapping regions
    total_length = num_sequences + seq_len - 1
    reconstructed = np.zeros((total_length, num_features))
    count_matrix = np.zeros((total_length, num_features))

    # Add each sequence back to its position
    for i in range(num_sequences):
        start_idx = i
        end_idx = i + seq_len
        reconstructed[start_idx:end_idx] += processed_data[i]
        count_matrix[start_idx:end_idx] += 1

    # Average overlapping regions
    reconstructed = reconstructed / (count_matrix + 1e-7)

    # Step 2: Reverse the MinMax normalization
    if original_min_max_params is not None:
        min_vals, max_vals = original_min_max_params
        # Reverse: norm_data = (data - min) / (max - min)
        # So: data = norm_data * (max - min) + min
        reconstructed = reconstructed * (max_vals - min_vals) + min_vals
    else:
        print("Warning: No original min/max parameters provided. Cannot fully reverse normalization.")
        print("The data will remain in normalized [0,1] range.")

    # Step 3: Reverse the chronological flip (data was flipped with [::-1])
    # reconstructed = reconstructed[::-1]

    return reconstructed


def get_original_data_with_params(data_name, csv_path):
    """
    Load original data and return both the data and normalization parameters.
    This mimics the preprocessing but saves the parameters for reversal.

    Args:
        data_name: Name of the dataset
        csv_path: Path to the CSV file

    Returns:
        original_data: The original CSV data
        min_vals: Min values used for normalization
        max_vals: Max values used for normalization
    """
    # Load original data
    ori_data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    # Store original before any transformations
    original_data = ori_data.copy()

    # Flip the data to make chronological data (same as preprocessing)
    ori_data = ori_data[::-1]

    # Get min/max before normalization
    min_vals = np.min(ori_data, 0)
    max_vals = np.max(ori_data, 0)

    return original_data, min_vals, max_vals


# Example usage:
if __name__ == "__main__":
    # First, get the original data and normalization parameters
    original_csv_data, min_vals, max_vals = get_original_data_with_params(
        "network_traffic",
        "data/ultra_long_range/youtube_traffic_bytes_only.csv"
    )

    # Assume you have processed_data from your model
    # processed_data = your_processed_sequences_here

    # Reverse the preprocessing
    # reconstructed = reverse_preprocess(processed_data, (min_vals, max_vals))

    # Compare with original
    # print("Original shape:", original_csv_data.shape)
    # print("Reconstructed shape:", reconstructed.shape)
    # print("Max absolute difference:", np.max(np.abs(original_csv_data - reconstructed)))