import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Stock Data Comparison",
    page_icon="📈",
    layout="wide"
)

def load_data():
    """Load original and generated data"""
    # File upload widgets
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Data")
        original_file = st.file_uploader(
            "Upload original stock data CSV",
            type=['csv'],
            key="original"
        )

    with col2:
        st.subheader("Generated Data")
        generated_file = st.file_uploader(
            "Upload generated stock data CSV",
            type=['csv'],
            key="generated"
        )

    original_data = None
    generated_data = None

    if original_file is not None:
        original_data = pd.read_csv(original_file)
        st.success(f"Original data loaded: {original_data.shape}")

    if generated_file is not None:
        generated_data = pd.read_csv(generated_file)
        st.success(f"Generated data loaded: {generated_data.shape}")

    return original_data, generated_data

def plot_distributions(original_data, generated_data, feature_cols):
    """Plot distribution comparisons for each feature"""

    n_features = len(feature_cols)
    fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(15, 8))
    if n_features == 1:
        axes = [axes]
    axes = axes.flatten()

    for i, feature in enumerate(feature_cols):
        ax = axes[i]

        # Plot histograms
        ax.hist(original_data[feature], bins=50, alpha=0.7, label='Original', density=True)
        ax.hist(generated_data[feature], bins=50, alpha=0.7, label='Generated', density=True)

        ax.set_title(f'{feature} Distribution')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig

def plot_time_series_sample(original_data, generated_data, feature_cols, sample_size=1000):
    """Plot time series comparison"""

    # Sample data for plotting (to avoid overcrowding)
    if len(original_data) > sample_size:
        orig_sample = original_data.sample(n=sample_size, random_state=42)
    else:
        orig_sample = original_data

    if len(generated_data) > sample_size:
        gen_sample = generated_data.sample(n=sample_size, random_state=42)
    else:
        gen_sample = generated_data

    n_features = len(feature_cols)
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 3 * n_features))
    if n_features == 1:
        axes = [axes]

    for i, feature in enumerate(feature_cols):
        axes[i].plot(orig_sample[feature].values, alpha=0.7, label='Original', linewidth=1)
        axes[i].plot(gen_sample[feature].values, alpha=0.7, label='Generated', linewidth=1)

        axes[i].set_title(f'{feature} Time Series Comparison')
        axes[i].set_xlabel('Data Point')
        axes[i].set_ylabel(feature)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_correlation_heatmap(original_data, generated_data, feature_cols):
    """Plot correlation heatmaps"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Original data correlation
    orig_corr = original_data[feature_cols].corr()
    sns.heatmap(orig_corr, annot=True, cmap='coolwarm', center=0, ax=ax1)
    ax1.set_title('Original Data Correlation')

    # Generated data correlation
    gen_corr = generated_data[feature_cols].corr()
    sns.heatmap(gen_corr, annot=True, cmap='coolwarm', center=0, ax=ax2)
    ax2.set_title('Generated Data Correlation')

    plt.tight_layout()
    return fig

def plot_statistics_comparison(original_data, generated_data, feature_cols):
    """Plot statistical comparison"""

    # Calculate statistics
    stats_data = []

    for feature in feature_cols:
        stats_data.append({
            'Feature': feature,
            'Data Type': 'Original',
            'Mean': original_data[feature].mean(),
            'Std': original_data[feature].std(),
            'Min': original_data[feature].min(),
            'Max': original_data[feature].max(),
            'Median': original_data[feature].median()
        })

        stats_data.append({
            'Feature': feature,
            'Data Type': 'Generated',
            'Mean': generated_data[feature].mean(),
            'Std': generated_data[feature].std(),
            'Min': generated_data[feature].min(),
            'Max': generated_data[feature].max(),
            'Median': generated_data[feature].median()
        })

    stats_df = pd.DataFrame(stats_data)

    # Plot comparison
    metrics = ['Mean', 'Std', 'Min', 'Max', 'Median']
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))

    for i, metric in enumerate(metrics):
        pivot_data = stats_df.pivot(index='Feature', columns='Data Type', values=metric)
        pivot_data.plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_xlabel('Features')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def main():
    st.title("📈 Stock Data Comparison Dashboard")
    st.markdown("Compare original and generated stock data visually")

    # Load data
    original_data, generated_data = load_data()

    if original_data is not None and generated_data is not None:

        # Get feature columns (exclude data_type if present)
        feature_cols = [col for col in original_data.columns if col != 'data_type']

        # Ensure both datasets have same features
        common_features = list(set(feature_cols) & set(generated_data.columns))
        if not common_features:
            st.error("No common features found between datasets!")
            return

        feature_cols = common_features
        st.write(f"Comparing {len(feature_cols)} features: {feature_cols}")

        # Display basic info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Data Shape", f"{original_data.shape[0]} × {original_data.shape[1]}")
        with col2:
            st.metric("Generated Data Shape", f"{generated_data.shape[0]} × {generated_data.shape[1]}")

        # Tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Distributions",
            "📈 Time Series",
            "🔗 Correlations",
            "📋 Statistics",
            "📑 Data Preview"
        ])

        with tab1:
            st.subheader("Distribution Comparison")
            st.markdown("Compare the probability distributions of each feature")
            fig1 = plot_distributions(original_data, generated_data, feature_cols)
            st.pyplot(fig1)

        with tab2:
            st.subheader("Time Series Comparison")
            st.markdown("Compare temporal patterns (sample of data points)")
            sample_size = st.slider("Sample size for plotting", 100, min(5000, len(original_data), len(generated_data)), 1000)
            fig2 = plot_time_series_sample(original_data, generated_data, feature_cols, sample_size)
            st.pyplot(fig2)

        with tab3:
            st.subheader("Feature Correlations")
            st.markdown("Compare correlation patterns between features")
            fig3 = plot_correlation_heatmap(original_data, generated_data, feature_cols)
            st.pyplot(fig3)

        with tab4:
            st.subheader("Statistical Comparison")
            st.markdown("Compare basic statistics across features")
            fig4 = plot_statistics_comparison(original_data, generated_data, feature_cols)
            st.pyplot(fig4)

        with tab5:
            st.subheader("Data Preview")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Data (first 10 rows)**")
                st.dataframe(original_data.head(10))

            with col2:
                st.write("**Generated Data (first 10 rows)**")
                st.dataframe(generated_data.head(10))

    else:
        st.info("👆 Please upload both original and generated data CSV files to start comparing")

        st.markdown("""
        ### How to use:
        1. Upload your original stock data CSV file
        2. Upload your generated stock data CSV file
        3. Explore the different visualization tabs to compare the data

        ### Expected CSV format:
        - Feature columns: `feature_0`, `feature_1`, etc. (or custom names)
        - Optional `data_type` column will be ignored
        - Data should be in denormalized/original scale for meaningful comparisons
        """)

if __name__ == "__main__":
    main()