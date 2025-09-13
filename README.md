# Pandas vs Polars Benchmark

This project benchmarks **Pandas vs Polars** for common data engineering operations.

## Features

- Dynamic dataset size & columns
- Benchmarks: select, filter, add/drop/rename columns, groupby, sort, fill null, string & datetime operations
- CSV summary with % difference and winner
- Auto-generated performance plots

## Usage

```bash
python benchmarks/benchmark_run.py --rows 5000000 --extra-cols 10
