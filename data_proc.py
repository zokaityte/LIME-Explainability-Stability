import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data_management')))

import big_data

# Example big data use

data = big_data.bigdata(csv_path="data_smol.csv")
data.remap_columns() # Remap Label (default) to int
data.filter_zeros() # Filter where unique count is 1 or 0
data.plot_distributions() # Plot distributions and save them to png
data.stats() # Calculate correlations and plot them, other statistics as well
data.remove_most_corred() # Drop most correlated

# Save full dataframe
main_data = data.get_main_data()
main_data.to_csv("final_main.csv", index=False)

# Split and save finals for train/val/test
data.split() # Splits and saves to csv files