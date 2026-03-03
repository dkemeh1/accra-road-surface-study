# Accra Road Surface and Blindspot Analysis

This repository contains the code and data used in the study:

**"Mapping Road Surface Conditions in Accra Using Multi-Sensor Satellite Data"**

It includes:

* Python scripts for the road surface machine learning pipeline
* Python scripts for statistical analysis of blindspots
* A QGIS project used for spatial analysis and mapping
* Experiment comparison results

---

# Folder Structure

```
FOR_PUBLICATION
│
├── pipeline.py
├── qgis_analysis.py
├── README.md
├── requirements.txt
│
├── experiment_comparison_results.xlsx
│
├── data
│   ├── imagery
│   │
│   └── findings
│       ├── Blindspots_named_FINAL.csv
│       └── non_blindspots.csv
│
└── Accra_QGIS_Project
    ├── accra_mapping_project.qgz
    └── data
        ├── Blindspots_named_FINAL.gpkg
        ├── Blindspots_fixed_geometric.gpkg
        ├── built_up.gpkg
        └── Accra_boundary.gpkg
```

---

# Requirements

Install the required Python libraries:

```
pip install -r requirements.txt
```

---

# Running the Machine Learning Pipeline

To run the satellite road surface classification pipeline:

```
python pipeline.py
```

The pipeline performs the following steps:

1. Extract road surface tags from OpenStreetMap
2. Create weak training labels
3. Segment roads into 100 m segments
4. Extract satellite features
5. Train models using spatial cross-validation
6. Predict road surface classes
7. Detect road surface changes
8. Compare model experiments

Results will be saved inside:

```
data/experiments/
```

---

# Running the Blindspot Statistical Analysis

To perform statistical comparison between blindspots and non-blindspots:

```
python qgis_analysis.py
```

This script performs:

* Descriptive statistics
* Mann-Whitney U test
* Welch t-test
* District-level blindspot summary

The output file will be saved as:

```
data/findings/analysis_FINAL_with_districts.xlsx
```

---

# Opening the QGIS Project

To explore the spatial analysis and maps:

1. Open **QGIS**
2. Open the project file:

```
Accra_QGIS_Project/accra_mapping_project.qgz
```

All required spatial layers are included in the `Accra_QGIS_Project/data` folder.

---

# Output Files Included

This repository includes the final experiment comparison results:

```
experiment_comparison_results.xlsx
```

This file summarizes the model performance across all experiments.

---

# Author

Desmond Kemeh
Ariel University
