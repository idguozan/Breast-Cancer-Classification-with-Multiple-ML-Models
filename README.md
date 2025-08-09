#  Breast Cancer Classification with Multiple Machine Learning Models

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-blue.svg)](https://plotly.com/)

[![GitHub stars](https://img.shields.io/github/stars/idguozan/Breast-Cancer-Classification-with-Multiple-ML-Models?style=social)](https://github.com/idguozan/Breast-Cancer-Classification-with-Multiple-ML-Models/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/idguozan/Breast-Cancer-Classification-with-Multiple-ML-Models?style=social)](https://github.com/idguozan/Breast-Cancer-Classification-with-Multiple-ML-Models/network)
[![GitHub issues](https://img.shields.io/github/issues/idguozan/Breast-Cancer-Classification-with-Multiple-ML-Models)](https://github.com/idguozan/Breast-Cancer-Classification-with-Multiple-ML-Models/issues)

</div>

---

<div align="center">
<h3> A comprehensive machine learning analysis system that performs <strong>automatic breast cancer diagnosis classification</strong> using <strong>6 different machine learning algorithms</strong> on the <strong>Wisconsin Breast Cancer</strong> dataset.</h3>
</div>

---

##  Table of Contents

<details>
<summary>Click to expand</summary>

- [ Features](#-features)
- [ Quick Start](#-quick-start)
- [ Installation](#-installation)
- [ Usage](#-usage)
- [ Project Structure](#️-project-structure)
- [ Models](#-models)
- [ Results](#-results)
- [ Analysis Pipeline](#-analysis-pipeline)
- [ Visualizations](#-visualizations)
- [ Contributing](#-contributing)
- [ License](#-license)

</details>

---

##  Features

<table>
<tr>
<td width="33%">

###  Machine Learning Models
- **Random Forest Classifier** - Ensemble learning with decision trees
- **Logistic Regression** - Linear classification with regularization
- **Decision Tree Classifier** - Interpretable tree-based model
- **Support Vector Machine (SVM)** - Kernel-based classification
- **K-Nearest Neighbors (KNN)** - Instance-based learning
- **Naive Bayes** - Probabilistic classifier

</td>
<td width="33%">

###  Advanced Analysis
- **Exploratory Data Analysis (EDA)** - Comprehensive data exploration
- **Hyperparameter Optimization** - GridSearchCV with cross-validation
- **Cross-Validation** - 5-fold stratified CV for robust evaluation
- **Feature Importance Analysis** - Understanding model decisions
- **Learning Curves** - Model performance vs training size
- **ROC-AUC Analysis** - Detailed classification performance

</td>
<td width="34%">

###  Rich Visualizations
- **Interactive Dashboard** (Plotly) - Dynamic model exploration
- **Individual ROC Curves** - Separate curves for each model
- **Confusion Matrix Heatmaps** - Classification accuracy visualization
- **Feature Distribution Plots** - Data distribution analysis
- **Correlation Matrices** - Feature relationship mapping
- **Model Performance Comparisons** - Side-by-side evaluations

</td>
</tr>
</table>

---

##  Quick Start

<div align="center">

** Get started in less than 5 minutes!**

</div>

```bash
# Clone the repository
git clone https://github.com/idguozan/Breast-Cancer-Classification-with-Multiple-ML-Models.git
cd Breast-Cancer-Classification-with-Multiple-ML-Models

# Install dependencies
pip install -r requirements.txt

# Run the complete analysis
python main.py
```

<div align="center">

** That's it! Your analysis results will be saved in the `results/` directory.**

</div>

---

##  Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/idguozan/Breast-Cancer-Classification-with-Multiple-ML-Models.git
cd Breast-Cancer-Classification-with-Multiple-ML-Models
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

##  Usage

### Complete Analysis Pipeline
```bash
python main.py
```

### Simple Analysis (Minimal Dependencies)
```bash
python run_simple_analysis.py
```

### Jupyter Notebook EDA
```bash
jupyter notebook notebooks/breast_cancer_eda.ipynb
```

### Run Tests
```bash
python -m pytest tests/ -v
# or
python tests/test_modules.py
```

---

##  Project Structure

```
Breast-Cancer-Classification-with-Multiple-ML-Models/
│
├──  data/
│   └── breast-cancer.csv          # Main dataset
│
├──  src/
│   ├── __init__.py
│   ├── data_processor.py          # Data processing module
│   ├── model_trainer.py           # Model training module
│   └── visualizer.py              # Visualization module
│
├──  notebooks/
│   └── breast_cancer_eda.ipynb    # Exploratory data analysis
│
├──  tests/
│   └── test_modules.py            # Unit tests
│
├──  results/
│   ├── *.png                      # Chart outputs
│   ├── *.html                     # Interactive dashboard
│   ├── *.joblib                   # Saved models
│   └── final_report.txt           # Detailed report
│
├──  requirements.txt            # Python dependencies
├──  main.py                     # Main application
└──  README.md                   # This file
```

---

##  Models

<div align="center">

|  Model |  Description |  Strengths |
|:--------:|:---------------|:-------------|
| **Random Forest** | Ensemble learning algorithm | Resistant to overfitting, feature importance |
| **Logistic Regression** | Linear classification | Fast, interpretable |
| **Decision Tree** | Tree-based decision making | Visual, interpretable |
| **SVM** | Maximum margin classifier | High-dimensional data |
| **KNN** | Neighborhood-based | Simple, effective |
| **Naive Bayes** | Probabilistic classification | Fast, requires less data |

</div>

---

##  Results

<div align="center">

###  Model Performance Metrics

**Accuracy** •  **Precision** •  **Recall** •  **F1-Score** •  **ROC-AUC**

</div>

###  Sample Output
```
📊 MODEL PERFORMANCE SUMMARY:
================================================
Model                Accuracy  Precision  Recall  F1-Score    AUC
Random Forest        0.9649    0.9650     0.9649  0.9649     0.9920
SVM                  0.9561    0.9565     0.9561  0.9561     0.9890
Logistic Regression  0.9474    0.9480     0.9474  0.9474     0.9850
```

<div align="center">

###  Output Files

|  **Visualizations** |  **Interactive Dashboard** |  **Models** |  **Report** |
|:---------------------:|:-----------------------------:|:--------------:|:--------------:|
| Charts in PNG format | HTML format | Saved in JobLib format | Detailed analysis in TXT format |

</div>

---

##  Analysis Pipeline

### Data Preprocessing
- Missing value checking
- Feature scaling (StandardScaler)
- Target variable encoding (M→0, B→1)
- Stratified train-test split

### Hyperparameter Optimization
- Automatic optimization with GridSearchCV
- 5-fold cross-validation
- Scoring: ROC-AUC

### Evaluation Metrics
- Robust evaluation with cross-validation
- Confusion matrix analysis
- ROC curve and AUC calculation
- Learning curves

---

##  Visualizations

The project generates comprehensive visualizations including:

- **ROC Curves** - Individual and combined ROC curves for all models
- **Confusion Matrices** - Heatmaps showing classification performance
- **Feature Importance** - Charts showing which features contribute most to predictions
- **Interactive Dashboard** - Plotly-based interactive exploration tool
- **Learning Curves** - Training progress and overfitting analysis
- **Correlation Matrix** - Feature relationship visualization

---

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

---

##  License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

##  Authors

-  **Developed by** - [Ozan İdgü](https://github.com/idguozan)

##  Development Timeline

This project was developed by **Ozan İdgü** between **February 12, 2025** and **March 20, 2025**.

---

##  Acknowledgments

- Researchers who contributed to the Wisconsin Breast Cancer Database
- Scikit-learn community
- Open source Python ecosystem

---

<div align="center">

###  **If you like this project, don't forget to give it a star!** ⭐

---

<p>
<img src="https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge" alt="Made with love">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="ML">
</p>

**© 2025 Ozan İdgü. All rights reserved.**

</div>
