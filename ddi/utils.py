import os
import pandas as pd

def get_rawdata_filepath(filename):
    return os.path.join(os.path.dirname(__file__), '..', 'raw_data', filename)

def get_data_filepath(filename):
    return os.path.join(os.path.dirname(__file__), 'data', filename)


def df_optimized(df, verbose=True):
    in_size = df.memory_usage(index=True).sum()
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            df[col] = round(df[col],4)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    out_size = df.memory_usage(index=True).sum()
    df.replace({False: 0, True: 1}, inplace=True) # converting bool into int
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("optimized size by {} % | {} GB".format(ratio, GB))
    return df
