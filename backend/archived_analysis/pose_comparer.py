import pandas as pd
import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt

def load_and_preprocess_data(csv_path):
    """Loads landmark data from a CSV and preprocesses it."""
    df = pd.read_csv(csv_path)

    # Pivot the table to have landmarks as columns
    df_pivot = df.pivot_table(index='frame_number', columns='landmark_name', values=['x', 'y', 'z'])

    # Flatten the multi-level columns
    df_pivot.columns = [f'{val}_{col}' for val, col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    # Fill missing values
    df_pivot = df_pivot.ffill().bfill()

    return df_pivot

if __name__ == "__main__":
    df = load_and_preprocess_data('backend/assets/front_kick.csv')
    my_df = load_and_preprocess_data('backend/assets/my_front_kick.csv')
    print(df.head())

    plt.figure(figsize=(12, 6))
    for col in df.columns[1:]:
        plt.plot(df['frame_number'], df[col], label=col, color= "blue")
    for col in my_df.columns[1:]:
        plt.plot(my_df['frame_number'], my_df[col], label=col, color= "orange", alpha=0.5)

    plt.show()
