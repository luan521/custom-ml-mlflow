# Custom ml model using MLFlow
This repository contains code for building a custom ML model using MLflow, a powerful open-source platform for managing the machine learning lifecycle. The project addresses two specific challenges commonly encountered during model deployment: accessing the predict_proba method output and combining the XGBoost and scikit-learn libraries seamlessly, applying the method StackingClassifier (scikit-learn) for combine three differents XGBClassifier models (XGBoost).

Usage
-
Train the model:
```
python src/train_model.py
```

The notebook located at notebooks/load_model.ipynb allows you to perform inference using each model tracked in the mlruns directory. Simply update the specific (experiment_id, run_id) values in cell 4 to make predictions with the desired model.

Folder structure
-
```
├── conda.yaml:  environment configuration needed to reproduce the Python environment required for the model
├── dataset
│   ├── generate_dataset.ipynb: generate the datasets for train and test the model
│   ├── test.csv: train dataset
│   └── train.csv: test dataset
├── mlruns: centralized repository for managing metadata and artifacts of machine learning experiments.
├── notebooks
│   └── load_model.ipynb: model inference
└── src
    ├── custom_stacking_classifier.py:  defines the custom stacking classifier implementation.
    ├── train_model.py: train the model
    └── utils.py
```