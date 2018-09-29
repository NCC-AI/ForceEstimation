
NCC-AI/ SceneClassification
==============================

classification of surgical video scene

Objective
------------
Classify surgery scene into 11 cases in real time.

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
        ├── generator       <- Scripts to turn raw data into features for modeling
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
Surgery Movie. Its duration is about two hours.

All data used in this project is in the DLB1 extended HDD (12T). 

    /mnt
    └──hdd
        └─data
            ├── movies
            │   ├── case05_HEIGHT720_FPS30.mp4
            │   ├── case06_HEIGHT720_FPS30.mp4 
            │
            └── movie_frames
                ├── case05_HEIGHT720_FPS30
                │   ├─ 1.jpg
                │   ├─ 2.jpg
                │
                ├── case06_HEIGHT720_FPS30
                │   ├─ 1.jpg
                │   ├─ 2.jpg
                │
         
Method
------------
- Classify each frame(not time series) using  *Xception or Inceptionv3, ResNet*.
- 3D CNN (time series)

Author
------------
Hiroki Matsuzaki

Reference
------------
[Xception, InceptionV3, ResNet50](https://keras.io/ja/applications/)
