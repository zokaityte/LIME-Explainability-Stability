# LIME Explainability Stability

This repository contains the official implementation of the experiments from the paper:

**["Comparative Analysis of Perturbation Techniques in LIME for Intrusion Detection Enhancement"](https://doi.org/10.3390/make7010021)**  
*Mantas Bacevicius, Agne Paulauskaite-Taraseviciene, Gintare Zokaityte, Lukas Kersys, Agne Moleikaityte*  
**Machine Learning and Knowledge Extraction,** MDPI, 2025  
[DOI: 10.3390/make7010021](https://doi.org/10.3390/make7010021)

Lime implementation based on: https://github.com/marcotcr/lime

### Setup

1. Python 3.11
2. Install requirements:
`
pip install -r requirements.txt
`

### Running experiment

1. Prepare dataset folder, containing `train.csv`, `val.csv` and `test.csv` files, label being the last column.
2. Prepare scikit-learn classifier model as `.pkl` file, trained on dataset. Model can be trained using `scripts/models_training/models_training.py` 
3. Adjust experiments config in `experiments_config.json` or generate new one with: `scripts/generate_experiments_config.py` script.
4. Run experiment with:
`python main.py`
5. Results will be saved in `_experiment_results` folder. It contains csv file with results for each experiment and folders for each experiment with plots.

Sample `experiments_config.json`:
```json
{
  "experiments": [
        {
      "model_path": "./tests/resources/sample_dataset_1_rf_model.pkl",
      "dataset_path": "./tests/resources/sample_dataset_1",
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
