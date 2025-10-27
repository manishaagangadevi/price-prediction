# 🚀 Amazon Multi-Modal Price Prediction Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkgreen?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-purple?style=for-the-badge&logo=lightgbm)](https://lightgbm.readthedocs.io/)

**An end-to-end Machine Learning pipeline leveraging multi-modal features (Text, Image, Tabular) to accurately predict e-commerce product prices.**

This project tackles the Amazon ML Challenge 2025, demonstrating advanced feature engineering, robust pipeline construction, and insightful model analysis for a complex regression task.

## ✨ Project Highlights

* **Multi-Modal Fusion:** Integrates features derived from product descriptions, images, and engineered attributes.
* **Scalable Feature Engineering:** Implements efficient pipelines for processing large-scale text (TF-IDF) and image (ResNet50 embeddings) datasets.
* **Robust `sklearn` Pipeline:** Utilizes `ColumnTransformer` and `Pipeline` for a reproducible, end-to-end workflow from raw data to prediction.
* **Iterative Modeling & Analysis:** Systematically evaluates the contribution of different feature sets, providing key insights into model behavior.
* **Optimized Performance:** Leverages techniques like PCA for dimensionality reduction and LightGBM for efficient gradient boosting.

## ⚙️ Architecture & Methodology

The core system is a unified `sklearn` `Pipeline` designed for seamless integration of diverse data types.

[ Raw Product Data (Text, Image Links) ] | v [ Custom Feature Engineering (Brand, IPQ) ] | v [ Multi-Modal Preprocessing Pipeline (ColumnTransformer) ] | |--> [ Text Pipeline ] --> [ TF-IDF ] -------------> | |--> [ Image Pipeline ] --> [ ResNet50 + PCA ] ---> | |--> [ Tabular Pipeline ] --> [ Scaler/Encoder ] -> | | v | [ Combined Feature Matrix ] | | | v +-----------------------------------> [ LightGBM Regressor ] | v [ Predicted Price (log1p) ]


### 1. Strategic Feature Engineering

Domain-specific features were extracted to significantly boost predictive power:
* **Item Pack Quantity (IPQ):** Regex-based extraction of pack size/volume (e.g., "6 pk", "32 oz").
* **Brand:** Categorical feature derived from text analysis.

### 2. Deep Learning for Image Understanding

A dedicated pipeline (`02_Image_Processing.ipynb`) handles image feature extraction:
* **Automated Download:** Fetches 150,000 images via provided URLs.
* **Transfer Learning:** Employs a pre-trained **ResNet50** CNN to generate 2048-dimensional embeddings for each image.
* **Efficient Storage:** Embeddings are persisted as `.pkl` files for rapid loading during model training.

### 3. Unified Preprocessing with `ColumnTransformer`

Handles heterogeneous data types within a single transformer:
* **Text:** `TfidfVectorizer` (n-grams 1-2, 20k features).
* **Images:** `PCA` applied to ResNet50 embeddings (reducing dimensions from 2048 to 128).
* **Tabular:** `StandardScaler` for `IPQ`, `OrdinalEncoder` for `Brand`.

### 4. Gradient Boosted Regression

A **LightGBM Regressor** is trained on the combined feature set to predict the `log1p`-transformed price, effectively handling the skewed target distribution.

## 📊 Performance & Insights

Iterative model development revealed the relative importance of different modalities:

| Model Version | Key Features Added         | Validation SMAPE (%) | Improvement vs. Prev |
| :------------ | :------------------------- | :------------------- | :------------------- |
| **v1** | Text (TF-IDF)              | 54.85%               | -                    |
| **v2** | + IPQ (Numeric)            | 54.54%               | -0.31%               |
| **v3** | + Brand (Categorical)      | 54.04%               | -0.50%               |
| **v4 (Final)**| + Images (ResNet50 + PCA)  | **53.08%** | **-0.96%** |

**Key Learnings:**
* Textual and engineered features formed the strong baseline, capturing most price variance.
* Image features provided a statistically significant but marginal improvement, suggesting potential redundancy with detailed text descriptions.
* The structured pipeline allowed for efficient experimentation and clear attribution of performance gains.

## 📁 Repository Structure

amazon-price-prediction/ ├── data/ │ ├── models/ │ │ └── resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 # Local ResNet50 weights │ ├── train.csv │ └── test.csv │ └── *.pkl # Generated embeddings (ignored by .gitignore) ├── notebooks/ │ ├── 01_EDA_and_Baseline.ipynb # Exploration & text-only model iteration │ ├── 02_Image_Processing.ipynb # Pipeline for image download & embedding generation │ └── 03_Final_Model.ipynb # Master notebook: multi-modal model training & evaluation ├── src/ │ └── utils.py # Helper function for reliable image downloading ├── outputs/ │ └── submission_final_images.csv # Final predictions file ├── .gitignore ├── README.md # This file └── requirements.txt # Project dependencies


## 🚀 Getting Started

Follow these steps to replicate the project environment and results:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/manishaagangadevi/price-prediction.git
    cd price-prediction
    ```
2.  **Set up Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate  # Mac/Linux
    pip install -r requirements.txt
    ```
3.  **Download Pre-trained Model:**
    * Download the [ResNet50 weights](https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5).
    * Place the `resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5` file inside the `data/models/` directory.
4.  **Run Image Processing Pipeline:**
    * Execute the `notebooks/02_Image_Processing.ipynb` notebook.
    * **Note:** This step is computationally intensive and time-consuming (12-24+ hours) as it downloads 150k images and generates embeddings.
5.  **Run Final Model Training:**
    * Once the image processing is complete, execute the `notebooks/03_Final_Model.ipynb` notebook.
    * This loads all features, trains the final LightGBM model, evaluates it, and generates the final predictions file in `outputs/`.