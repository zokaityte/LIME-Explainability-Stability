import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))

import big_data

# Example big data use

categorical_cols_smoldata = ["Dst Port",
                            "Fwd PSH Flags",
                            # "Bwd PSH Flags", # Removed in filter_zeros
                            # "Fwd URG Flags", # Removed in filter_zeros
                            # "Bwd URG Flags", # Removed in filter_zeros
                            #"FIN Flag Cnt",
                            #"SYN Flag Cnt",
                            #"RST Flag Cnt",
                            #"PSH Flag Cnt",
                            #"ACK Flag Cnt",
                            #"URG Flag Cnt",
                            # "CWE Flag Count", # Removed in filter_zeros
                            #"ECE Flag Cnt", # :(
                            "Protocol",
                            "Timestamp" # Must encode timestamp to only show hours!
                            ]

categorical_cols_bigdata = ["Destination Port",
                            "Fwd PSH Flags",
                            #"Bwd PSH Flags", # Removed in filter_zeros
                            "Fwd URG Flags",
                            #"Bwd URG Flags", # Removed in filter_zeros
                            "act_data_pkt_fwd",
                            #"FIN Flag Cnt",
                            #"SYN Flag Cnt",
                            #"RST Flag Cnt",
                            #"PSH Flag Cnt",
                            #"ACK Flag Cnt",
                            #"URG Flag Cnt",
                            #"CWE Flag Count",
                            #"ECE Flag Cnt",
                            ]

csv_data_path="duomenys_laimas_0924.csv"

data = big_data.bigdata(csv_path=csv_data_path)
data.remap_columns() # Remap Label (default) to int

if "duomenys_laimas_0924.csv" in csv_data_path:
    data.filter_zeros() # Filter where unique count is 1 or 0
    data.conditional_prob_transform(categorical_cols_bigdata, target_col="Label", alpha=10) # Maybe after distribution plot? But always before corr!
elif "data_smol.csv" in csv_data_path:
    data.transform_timestamp("Timestamp") # Remap tiemestamp to only show the hour number!
    data.filter_zeros() # Filter where unique count is 1 or 0
    data.conditional_prob_transform(categorical_cols_smoldata, target_col="Label", alpha=10) # Maybe after distribution plot? But always before corr!

data.plot_distributions() # Plot distributions and save them to png
data.stats() # Calculate correlations and plot them, other statistics as well
data.remove_most_corred() # Drop most correlated

# Save full dataframe
main_data = data.get_main_data()
main_data.to_csv("final_main.csv", index=False)

# Split and save finals for train/val/test
data.split() # Splits and saves to csv files