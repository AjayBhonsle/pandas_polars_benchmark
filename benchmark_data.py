import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime, timedelta


def generate_data(n_rows, n_extra_cols):
    """Generate dataset with dynamic rows and extra numeric columns"""
    data = {
        "id": np.arange(1, n_rows + 1),
        "name": [f"User{i}" for i in range(1, n_rows + 1)],
        "salary": np.random.randint(4000, 12000, size=n_rows),
        "dept": np.random.choice(["IT", "Finance", "HR", "Admin", "Sales", "Support"], size=n_rows),
        "hire_date": [datetime(2000, 1, 1) + timedelta(days=int(x)) for x in np.random.randint(0, 7300, size=n_rows)]
    }
    for j in range(1, n_extra_cols + 1):
        data[f"num_col{j}"] = np.random.randint(1, 10000, size=n_rows)

    return pd.DataFrame(data), pl.DataFrame(data)
