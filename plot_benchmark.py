import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import argparse
import os

def plot_csv(csv_file, n_rows=0, n_cols=0):
    # Read CSV
    df = pd.read_csv(csv_file)

    # Sort by Polars % improvement descending
    df = df.sort_values('%_diff_vs_pandas', ascending=False).reset_index(drop=True)

    colors = df['winner'].apply(lambda x: 'green' if x == 'Polars' else 'gray')
    bar_width = 0.4

    plt.figure(figsize=(14, 9))
    y_pos = range(len(df))

    # Plot horizontal bars
    plt.barh(y_pos, df['polars_time_sec'], color=colors, alpha=0.85, height=bar_width, label='Polars')
    plt.barh(y_pos, df['pandas_time_sec'], left=df['polars_time_sec'], color="#CCCCCC", alpha=0.9, height=bar_width,
             label='Pandas')

    # Add +% labels
    for i, row in enumerate(df.itertuples()):
        improvement = row._4  # '%_diff_vs_pandas' column
        label = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        plt.text(row.polars_time_sec + 0.01 * row.polars_time_sec, i, label, va='center', fontsize=10,
                 fontweight='bold')

    plt.yticks(y_pos, df['operation'], fontsize=10)
    plt.xlabel("Execution Time (seconds)", fontsize=12)
    plt.ylabel("Data Engineering Operation", fontsize=12)
    plt.suptitle("Pandas vs Polars Performance Benchmark", fontsize=18, fontweight='bold', y=1.02)
    plt.title(
        f"Dataset: {n_rows:,} rows × {n_cols} columns | Green = Faster (Polars), Gray = Slower (Pandas)\nPercentage = how much faster Polars is (+% means Polars faster)",
        fontsize=11, loc='left', pad=25)

    # Minor ticks and grid
    ax = plt.gca()
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(n=5))
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.grid(which='major', color='black', linestyle='-', linewidth=0.8, alpha=0.7)

    plt.legend(fontsize=11)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plot_file = f"plots/benchmark_plot_{n_rows}_rows_{n_cols}_cols.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {plot_file}")


# Run as script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Pandas vs Polars benchmark")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--cols", type=int, default=0)
    args = parser.parse_args()
    plot_csv(args.csv, args.rows, args.cols)
