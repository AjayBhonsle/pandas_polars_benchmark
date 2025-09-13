import pandas as pd
import polars as pl
import time
import argparse
import subprocess
import os
import sys

from benchmark_data import generate_data

# -----------------------
# CLI arguments
# -----------------------
parser = argparse.ArgumentParser(description="Pandas vs Polars benchmark")
parser.add_argument('--rows', type=int, default=50000, help="Number of rows")
parser.add_argument('--extra-cols', type=int, default=5, help="Number of extra numeric columns")
args = parser.parse_args()

n_rows = args.rows
n_extra_cols = args.extra_cols
n_total_cols = 5 + n_extra_cols

print(f"Generating dataset: {n_rows:,} rows × {n_total_cols} columns...")
pdf, pldf = generate_data(n_rows, n_extra_cols)

results = []

def benchmark(operation_name, pandas_func, polars_func, repeat=2):
    """Benchmark a single operation"""
    # Warmup
    pandas_func(); polars_func()

    # Pandas timing
    pandas_times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = pandas_func()
        pandas_times.append(time.perf_counter() - start)
    pandas_time = sum(pandas_times)/repeat

    # Polars timing
    polars_times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = polars_func()
        polars_times.append(time.perf_counter() - start)
    polars_time = sum(polars_times)/repeat

    results.append({
        "operation": operation_name,
        "pandas_time_sec": pandas_time,
        "polars_time_sec": polars_time
    })

# -----------------------
# Benchmarks
# -----------------------

# Column operations
benchmark("Select Columns",
          lambda: pdf[["id","name","salary"]],
          lambda: pldf.select(["id","name","salary"]))

benchmark("Add Column bonus",
          lambda: pdf.assign(bonus=pdf["salary"]*0.2),
          lambda: pldf.with_columns((pl.col("salary")*0.2).alias("bonus")))

benchmark("Drop Extra Columns",
          lambda: pdf.drop(columns=[f"num_col{i}" for i in range(1,n_extra_cols+1)] if n_extra_cols>0 else []),
          lambda: pldf.drop([f"num_col{i}" for i in range(1,n_extra_cols+1)] if n_extra_cols>0 else []))

# Filtering
benchmark("Filter salary > 9000",
          lambda: pdf[pdf["salary"]>9000],
          lambda: pldf.filter(pl.col("salary")>9000))

benchmark("Filter salary>8000 & dept==IT",
          lambda: pdf[(pdf['salary']>8000)&(pdf['dept']=="IT")],
          lambda: pldf.filter((pl.col("salary")>8000)&(pl.col("dept")=="IT")))

# Grouping & Aggregation
benchmark("GroupBy dept avg/max salary",
          lambda: pdf.groupby("dept")["salary"].agg(["mean","max"]),
          lambda: pldf.group_by("dept").agg([
              pl.col("salary").mean().alias("avg_salary"),
              pl.col("salary").max().alias("max_salary")
          ]))

# Sorting
benchmark("Sort by salary desc",
          lambda: pdf.sort_values("salary",ascending=False),
          lambda: pldf.sort("salary",descending=True))

# Null handling
pdf_null = pdf.copy()
if n_rows>=200:
    pdf_null.loc[100:200,"salary"] = None
pldf_null = pldf.with_columns(
    pl.when(pl.col("id").is_in(list(range(101,201)))).then(None).otherwise(pl.col("salary")).alias("salary")
)
benchmark("Fill Null Salary",
          lambda: pdf_null.fillna({"salary":0}),
          lambda: pldf_null.fill_null(0))

# String operations
benchmark("Uppercase Names",
          lambda: pdf["name"].str.upper(),
          lambda: pldf.with_columns(pl.col("name").str.to_uppercase()))

benchmark("Name contains '5'",
          lambda: pdf[pdf["name"].str.contains("5")],
          lambda: pldf.filter(pl.col("name").str.contains("5")))

# Datetime operations
benchmark("Extract Hire Year",
          lambda: pdf["hire_date"].dt.year,
          lambda: pldf.with_columns(pl.col("hire_date").dt.year().alias("hire_year")))

# Multi-column arithmetic
if n_extra_cols > 0:
    benchmark(
        "Sum Extra Columns",
        lambda: pdf[[f"num_col{i}" for i in range(1, n_extra_cols+1)]].sum(axis=1),
        lambda: pldf.with_columns(
            sum([pl.col(f"num_col{i}") for i in range(1, n_extra_cols+1)]).alias("sum_extra")
        )
    )
# Update values based on condition
benchmark("Update Bonus if salary>10000",
          lambda: pdf.assign(bonus=lambda df: df["salary"].where(df["salary"]<=10000, df["salary"]*1.1)),
          lambda: pldf.with_columns(
              pl.when(pl.col("salary")>10000).then(pl.col("salary")*1.1).otherwise(pl.col("salary")).alias("bonus")
          ))

# Conditional new column
benchmark("High Salary Flag",
          lambda: pdf.assign(high_salary=lambda df: df["salary"]>9000),
          lambda: pldf.with_columns((pl.col("salary")>9000).alias("high_salary"))
          )

# Cumulative sum
# Rolling mean (window=5)
benchmark("Rolling mean salary (window=5)",
          lambda: pdf["salary"].rolling(window=5).mean(),
          lambda: pldf.with_columns(pl.col("salary").rolling_mean(5).alias("rolling_mean_salary")))

# Sampling
sample_size = min(10000, n_rows)
benchmark(f"Sample {sample_size} rows",
          lambda: pdf.sample(n=sample_size),
          lambda: pldf.sample(n=sample_size))

# Distinct
benchmark("Distinct Departments",
          lambda: pdf["dept"].unique(),
          lambda: pldf.select(pl.col("dept").unique()))

# -----------------------
# Save CSV summary
# -----------------------
summary_df = pd.DataFrame(results)
summary_df["%_diff_vs_pandas"] = ((summary_df["pandas_time_sec"] - summary_df["polars_time_sec"])
                                  / summary_df["pandas_time_sec"]*100).round(2)
summary_df["winner"] = summary_df.apply(lambda row: "Polars" if row["polars_time_sec"] < row["pandas_time_sec"] else "Pandas", axis=1)

summary_csv = f"benchmark_results_{n_rows}_rows_{n_extra_cols}_extra_cols.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Benchmark CSV saved: {summary_csv}")

# -----------------------
# Call plot script
# -----------------------
subprocess.run([sys.executable, os.path.join(os.getcwd(), "plot_benchmark.py"),
                "--csv", summary_csv,
                "--rows", str(n_rows),
                "--cols", str(n_total_cols)])
