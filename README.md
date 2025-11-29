# Motorbike Recommendation & Market Segmentation Analysis

**Implemented by:** NamAn Thanh Bang  
**Supervisor:** Khuat Thuy Phuong

---

## Overview

This project provides a motorbike recommendation system and market segmentation model based on used motorbike listings collected from Chotot. The system combines Vietnamese NLP preprocessing, TF-IDF vectorization, cosine similarity, and clustering algorithms to classify motorbikes into meaningful market tiers.

---

## Project Structure
```
D:.
│   .gitattributes
│   app.py
│   requirements.txt
│   utils.py
│
├── .devcontainer
│     └── devcontainer.json
│
├── assets
│     └── hinh_elbow.png
│
├── data
│     ├── data_motobikes.csv
│     ├── data_motobikes_cleaned_text.csv
│     └── encoded_data_motobikes.csv
│
├── files
│     ├── emojicon.txt
│     ├── english-vnmese.txt
│     ├── teencode.txt
│     ├── vietnamese-stopwords.txt
│     └── wrong-word.txt
│
├── models
│     ├── gmm.pkl
│     ├── isolation_forest.pkl
│     ├── kmeans.pkl
│     ├── kproto.pkl
│     ├── scaler_robust.pkl
│     ├── scaler_standard_kproto.pkl
│     ├── tfidf_vectorizer.pkl
│     └── xe_cosine_sim.pkl
│
└── __pycache__
      └── utils.cpython-310.pyc
```
---

## Methodology Summary

### Recommendation Engine
- TF-IDF vectorization of listing descriptions.
- Cosine similarity to match user queries with relevant motorbikes.
- Vietnamese NLP preprocessing including stopword removal, teencode normalization, emoji parsing, and spelling correction.

### Market Segmentation
- K-Means clustering for numerical features (price, year, odometer).
- K-Prototypes for mixed categorical and numerical attributes.
- Gaussian Mixture Models (GMM) for probabilistic clustering and soft assignment.

---

## Installation

```bash
pip install -r requirements.txt

python app.py
