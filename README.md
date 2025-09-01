# Comparison of the performance of features based on Symbolic Dynamics and Recurrence Microstates for time series classification

This repository contains the data and scripts used in the article for feature extraction and machine learning analysis of dynamical systems.

**Comparison of the performance of features based on Symbolic Dynamics and Recurrence Microstates for time series classification**  
**Authors: H. C. Costa, J. V. M. Silveira, B. R. R. Boaretto, C.Masoller, S. R. Lopes, T. L. Prado**  

---

## Repository Structure

### 1. Features
Extracted features are organized into **Recurrence Microstates (RM)** and **Symbolic Dynamics (SD)**, and further divided by method and number of classes:

```
features/
├── RM/
│   ├── flows/
│   │   ├── 10_Classes/
│   │   ├── 20_Classes/
│   │   └── 40_Classes/
│   └── maps/
│       ├── 10_Classes/
│       ├── 20_Classes/
│       └── 40_Classes/
└── SD/
    ├── flows/
    │   ├── 10_Classes/
    │   ├── 20_Classes/
    │   └── 40_Classes/
    └── maps/
        ├── 10_Classes/
        ├── 20_Classes/
        └── 40_Classes/
```

- `RM/` → features based on recurrence microstates.  
- `SD/` → features based on symbolic dynamics.  
- `flows/` and `maps/` → different methods used to generate features.  
- `10_Classes`, `20_Classes`, `40_Classes` → datasets with different numbers of classes.

### 2. Scripts
Scripts are organized into:

```
scripts/
├── data_generation/
│   ├── RM/      # scripts to generate RM features
│   └── SD/      # scripts to generate SD features
└── random-forest/
    ├── flows/ # Random Forest training on flows
    ├── maps/ # Random Forest training on maps
    └── Importances/ # Calculation of feature importances for flows and maps
```

- `data_generation/` → scripts to generate the features from raw time series.  
  - `RM/` → recurrence microstates feature extraction.  
  - `SD/` → symbolic dynamics feature extraction.  
- `random-forest/` → scripts that train Random Forest models:  
  - `flows/` → training on features from flows.  
  - `maps/` → training on features from maps.  
  - `Importances/` → calculation of feature importances for both flows and maps.


## Dynamical Systems Included
- $\beta$X
- Gauss Map
- Logistic Map  
- Hénon Map  
- Ikeda Map  
- Tent Map
- Lorenz System  
- Rössler System

## Features
- **Recurrence Microstates (RM)**  
- **Symbolic Dynamics (SD)**  
- **Combined approaches (RM + SD)**

## Machine Learning Models
- Random Forest  


---

## How to Reproduce the Results
1. Run the scripts in `scripts/data_generation/` to generate the features in the `features/` folder.  
2. Run the scripts in `scripts/random-forest/` to train models and generate confusion matrices in the `results/` folder.

---

## Citation
If you use this dataset or scripts, please cite the corresponding article:

**Article Title: Comparison of the performance of features based on Symbolic Dynamics and Recurrence Microstates for time series classification**  
**Authors: H. C. Costa, J. V. M. Silveira, B. R. R. Boaretto, C.Masoller, S. R. Lopes, T. L. Prado**  
Journal/Preprint,
Year and DOI Not available yet

