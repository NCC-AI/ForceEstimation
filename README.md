
NCC-AI/ ForceEstimation
==============================

Estimate how strong the forceps is pulling organ using deep neural network.

Objective
------------
Visualize how strong the forceps is pulling organ from video.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models(.h5py), model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    │   └── log            <- accuracy and loss history
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── reader           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── generators       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   └── cnn_lstm.py
        │
        ├── visualization  <- Scripts to create exploratory and results oriented visualizations
        │   └── visualize.py
        │
        └── train.py

Shared Data Path
------------

         
Method
------------
- Regression.

Author
------------
Hiroaki Takano

Reference
------------
  
