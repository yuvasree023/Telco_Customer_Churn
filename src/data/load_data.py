import os
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Loads CSV data into a pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


# --- Put your file path here when you call the function ---
file_path = (
    r"C:\Users\91735\Desktop\projects\Telco-Customer-Churn-ML\src\data\raw\Telco-Customer-Churn.csv"
)
# Load the data into 'df'
df = load_data(file_path)

# Preview the loaded data
print(df.head())