import copy
import json

# Define the common configuration for each experiment
base_experiment = {
    "model_path": "model_checkpoints/random_state_42min_samples_split_2max_depth_30min_samples_leaf_1max_features_sqrt_val.pkl",  # Default model path
    "dataset_path": "./data/big_data_zero_corr_enc",
    "label_names": ["BENIGN", "DDoS", "PortScan", "Bot", "Infiltration", "Web Attack � Brute Force", "Web Attack � XSS",
                    "Web Attack � Sql Injection", "FTP-Patator", "SSH-Patator", "DoS slowloris", "DoS Slowhttptest",
                    "DoS Hulk", "DoS GoldenEye", "Heartbleed", "DoS attacks-SlowHTTPTest", "DoS attacks-Hulk",
                    "Brute Force -Web", "Brute Force -XSS", "SQL Injection", "DDoS attacks-LOIC-HTTP", "Infilteration",
                    "DoS attacks-GoldenEye", "DoS attacks-Slowloris", "FTP-BruteForce", "SSH-Bruteforce", "DDOS attack-LOIC-UDP",
                    "DDOS attack-HOIC"],
    "categorical_columns_names": ["Fwd PSH Flags", "Fwd URG Flags", "FIN Flag Count", "RST Flag Count", "PSH Flag Count",
                                  "ACK Flag Count", "URG Flag Count"],
    "explainer_config": {
        "kernel_width": None,
        "kernel": None,
        "sample_around_instance": False,
        "num_features": 10,
        "num_samples": 5000,
        "sampling_func": None,
        "sampling_func_params": {}
    },
    "times_to_run": 30,
    "random_seed": None
}

# Define the specific configurations for each sampling function
sampling_configs = [
    {"sampling_func": "gaussian", "sampling_func_params": {}},
    {"sampling_func": "gamma", "sampling_func_params": {"shape_param": 1, "scale": 1}},
    {"sampling_func": "beta", "sampling_func_params": {"alpha": 0.5, "beta_param": 0.5}},
    {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 3}},
    {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 4}},
    {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 5}},
    {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 6}},
    {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 7}},
    {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.1}},
    {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.4}},
    {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.5}},
    {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.7}},
    {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.8}}
]

# Define the list of class labels to explain for each experiment
class_labels = [0, 17, 20]

# Create the list of experiments
experiments = []
for config in sampling_configs:
    for label in class_labels:
        experiment = copy.deepcopy(base_experiment)
        experiment["class_label_to_explain"] = label
        experiment["explainer_config"]["sampling_func"] = config["sampling_func"]
        experiment["explainer_config"]["sampling_func_params"] = config["sampling_func_params"]
        experiments.append(experiment)

# Combine into a final config
config = {"experiments": experiments}

# Write to JSON file
with open("experiments_config.json", "w") as f:
    json.dump(config, f, indent=4)

print("Configuration file 'experiments_config.json' generated successfully.")
