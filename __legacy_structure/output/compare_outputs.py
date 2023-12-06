import os

import pandas as pd
import numpy as np


def custom_elementwise_compare(a, b, rtol=1e-5, atol=1e-8):
    """
    Custom element-wise comparison function with precision consideration.
    """
    if a == b:
        return True
    else:
        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            # If both elements are floats, perform numeric comparison with precision
            return np.isclose(a, b, rtol=rtol, atol=atol)
        elif pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            # If both elements are numeric, perform numeric comparison with precision
            return np.isclose(a, b, rtol=rtol, atol=atol)
        elif type(a) is not type(b):
            # If elements have different types, consider them not equal
            print(f"Element types are different: a={type(a)}, b={type(b)}")
            return False


def compare_dataframes(df1, df2, rtol=1e-5, atol=1e-8, log_file=None):
    # Check if the shape of the data frames is the same
    if df1.shape != df2.shape:
        log_message = "Data frames have different shapes."
        print(log_message)
        if log_file:
            with open(log_file, "a") as log:
                log.write(log_message + "\n")
        return None

    # Create a mask for NaN values in both data frames
    nan_mask_df1 = pd.isna(df1)
    nan_mask_df2 = pd.isna(df2)

    # Create a new data frame with boolean values indicating equality
    equality_df = np.vectorize(custom_elementwise_compare)(df1, df2, rtol=rtol, atol=atol) | (
            nan_mask_df1 & nan_mask_df2)

    # Print details about the differences and write to log file
    nanzero_list_df1 = []
    nanzero_list_df2 = []
    if not equality_df.all().all():
        log_message = "Data frames are not equal."
        differences = np.where(~equality_df)
        for i, j in zip(*differences):
            if df1.iat[i, j] == 0 and np.isnan(df2.iat[i, j]):
                nanzero_list_df1.append((i, j))
            elif np.isnan(df1.iat[i, j]) and df2.iat[i, j] == 0:
                nanzero_list_df2.append((i, j))
            else:
                diff_message = f"Difference at position ({i}, {j}): df1={df1.iat[i, j]}, df2={df2.iat[i, j]}"
                print(diff_message)
                log_message += diff_message + "\n"
        if len(nanzero_list_df1) is not 0:
            diff_message = f"{len(nanzero_list_df1)} differences of type: df1=0.0, df2=nan\n"
            log_message += diff_message
            print(diff_message)
        if len(nanzero_list_df2) is not 0:
            diff_message = f"{len(nanzero_list_df2)} differences of type: df1=nan, df2=0.0\n"
            log_message += diff_message
            print(diff_message)
        if log_file:
            with open(log_file, "a") as log:
                log.write(log_message)
        return equality_df
    else:
        log_message = "Data frames are equal."
        print(log_message)
        if log_file:
            with open(log_file, "a") as log:
                log.write(log_message + "\n")
        return None


for f in os.listdir("./"):
    log_file_location = "./log_file.txt"
    if 'csv' in f:
        print(f"Comparing {f}...")
        with open(log_file_location, "a") as log:
            log.write(f"Comparing {f}...\n")
        df_new = pd.read_csv(f"./{f}")
        df_old = pd.read_csv(f"./original_output/{f}")
        compare_dataframes(df_new, df_old, log_file=log_file_location)
        with open(log_file_location, "a") as log:
            log.write("\n\n")
        print("\n\n")
