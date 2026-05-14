"""Classical ML models and training pipeline.

Modules
-------
trainer          Walk-forward CV training loop with MLflow tracking and
                 model-registry promotion of the best run.
lightgbm_model   LightGBM regressor wrapped in the common ``BaseModel``
                 interface.
lstm_model       PyTorch LSTM sequence model wrapped in the same interface,
                 included for architectural breadth.
"""
