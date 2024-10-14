# LIME Explainability Stability

Lime implementation based on: https://github.com/marcotcr/lime

### Setup

1. Python 3.11
2. Install requirements:
`
pip install -r requirements.txt
`

### Running experiment

1. Add datasets to `datasets` folder, containing `train_data.csv`, `val_data.csv` and `test_data.csv` files.
2. Add trained scikit-learn classifier models to `models` folder, containing `.pkl` file, trained on dataset.
3. Adjust experiments config in `experiments_config.json`.
4. Run experiment with:
`python main.py`
5. Results will be saved in `_results` folder. It contains csv file with results for each experiment and folders for each experiment with plots.

Sample `experiments_config.json`:
```json
{
  "experiments": [
        {
      "model_path": "./model_checkpoints/sample_dataset_1_rf_model.pkl",
      "dataset_path": "./datasets/sample_dataset_1",
      "label_names": ["Benign", "FTP-BruteForce", "SSH-Bruteforce"],
      "categorical_columns_names": ["Fwd PSH Flags", "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "ECE Flag Cnt"],
      "explainer_config": {
        "kernel_width": null,
        "kernel": null,
        "sample_around_instance": false,
        "num_features": 10,
        "num_samples": 1000,
        "sampling_func": "gaussian",
        "sampling_func_params": {}
      },
      "times_to_run": 30,
      "random_seed": null
    }
  ]
}
```

### Running tests
1. Run tests with:
`python -m unittest discover -s tests`
```