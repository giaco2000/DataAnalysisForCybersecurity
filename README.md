# PE Malware Detection with Machine Learning

## 📋 Project Overview

This repository contains a **Knowledge Discovery and Data Mining** project focused on **Portable Executable (PE) malware detection** using machine learning techniques. The project implements a comprehensive analysis pipeline for binary classification of executable files as either **goodware** or **malware**.

## 🎯 Project Goal

The main objective is to develop and evaluate machine learning models capable of accurately distinguishing between benign software (goodware) and malicious software (malware) using static analysis features extracted from PE files.

## 📊 Dataset

- **Source**: Subset of [EMBER Dataset](https://github.com/elastic/ember)
- **Size**: 12,000 samples
- **Features**: 2,382 attributes (2,381 numeric features + 1 binary class label)
- **Feature Extraction**: LIEF library for PE file analysis
- **Classes**: 
  - `0`: Goodware (benign software)
  - `1`: Malware (malicious software)

## 🔬 Technical Approach

### Data Preprocessing & Feature Engineering
- **Exploratory Data Analysis** with Pandas
- **Feature distribution analysis** and correlation study
- **Useless feature removal** (constant values)
- **Class distribution analysis** and balance assessment

### Feature Selection & Dimensionality Reduction
- **Mutual Information (MI) ranking** for feature importance
- **Principal Component Analysis (PCA)** for dimensionality reduction
- **Feature selection** with configurable thresholds
- **Boxplot analysis** for feature discrimination

### Machine Learning Models
- **Decision Trees** with criterion optimization (Gini/Entropy)
- **Random Forest** with hyperparameter tuning
- **K-Nearest Neighbors (KNN)** with optimal k selection
- **Ensemble methods** for improved performance

### Model Evaluation
- **Stratified 5-fold Cross-Validation** for robust evaluation
- **F1-Score optimization** for imbalanced data handling
- **Confusion Matrix** and **Classification Reports**
- **Hyperparameter optimization** through grid search

## 🛠️ Technologies Used

- **Python** (Data Science stack)
- **Pandas** (Data manipulation and analysis)
- **Scikit-learn** (Machine learning algorithms)
- **LIEF** (PE file feature extraction)
- **Matplotlib** (Data visualization)

## 📈 Key Features

- Comprehensive **feature analysis** and **selection pipeline**
- **Cross-validation** with multiple algorithms comparison
- **Hyperparameter optimization** for best model configuration
- **Performance evaluation** with multiple metrics
- **PCA vs. Feature Selection** comparison study

## 🎓 Academic Context

**Course**: Data Analysis for 
Cybersecurity  
 

