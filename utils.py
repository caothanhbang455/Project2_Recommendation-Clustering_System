import pandas as pd
import numpy as np
import pickle
import re
import streamlit as st
from pyvi import ViTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_samples, silhouette_score

# --- SYSTEM CONFIGURATION ---
@st.cache_resource
def load_resources():
    def load_file_to_dict(path):
        d = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2: d[parts[0]] = parts[1]
        except: pass
        return d
    
    def load_file_to_list(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        except: return []

    teencode = load_file_to_dict('files/teencode.txt')
    emoji = load_file_to_dict('files/emojicon.txt')
    stopwords = load_file_to_list('files/vietnamese-stopwords.txt')
    stopwords.append('xe')
    
    return teencode, emoji, stopwords

# --- PREPROCESSING ---
def preprocess_text(text, teencode, emoji_dict, stopwords):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    for emo, meaning in emoji_dict.items():
        text = text.replace(emo, f' {meaning} ')
    for code, meaning in teencode.items():
        text = re.sub(r'\b' + re.escape(code) + r'\b', f' {meaning} ', text)
    text = ViTokenizer.tokenize(text)
    text = re.sub(r'[!"#$%&\'()*+,/:;<=>?@\[\]\\^`{|}~]', ' ', text)
    tokens = text.split()
    cleaned_tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    return ' '.join(cleaned_tokens).strip()

# --- LOAD MODELS ---
@st.cache_resource
def load_recommender_system():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/xe_cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
    try:
        df = pd.read_csv('data/data_motobikes_cleaned_text.csv') 
    except:
        df = pd.read_csv('data/data_motobikes.csv')
    
    tfidf_matrix = vectorizer.transform(df['content'].fillna('').astype(str))
    return vectorizer, tfidf_matrix, cosine_sim, df


def load_clustering_model(algorithm):
    """Load model, scaler, and isolation forest"""
    
    # 1. Load Data
    df_encoded = pd.read_csv('data/encoded_data_motobikes.csv')
    
    # Clean junk columns
    junk_cols = ['Unnamed: 0', 'Unnamed: 0.1', 'id', 'stt', 'Cluster']
    cols_to_drop = [c for c in junk_cols if c in df_encoded.columns]
    if cols_to_drop:
        df_encoded = df_encoded.drop(columns=cols_to_drop)

    # 2. Load Isolation Forest (Global for all models based on notebook logic)
    # NOTE: You need to save your IsolationForest model to this path first!
    try:
        with open('models/isolation_forest.pkl', 'rb') as f:
            iso_model = pickle.load(f)
    except:
        iso_model = None # Handle case if user hasn't saved it yet

    # 3. Load Specific Models
    if algorithm == 'KMeans':
        with open('models/scaler_robust.pkl', 'rb') as f: scaler = pickle.load(f)
        with open('models/kmeans.pkl', 'rb') as f: model = pickle.load(f)
        return scaler, model, iso_model, df_encoded, 'numeric'

    elif algorithm == 'GMM':
        with open('models/scaler_robust.pkl', 'rb') as f: scaler = pickle.load(f)
        with open('models/gmm.pkl', 'rb') as f: model = pickle.load(f)
        return scaler, model, iso_model, df_encoded, 'numeric'

    elif algorithm == 'K-Prototypes':
        with open('models/scaler_standard_kproto.pkl', 'rb') as f: scaler = pickle.load(f)
        with open('models/kproto.pkl', 'rb') as f: model = pickle.load(f)
        return scaler, model, iso_model, df_encoded, 'mixed'

    return None, None, None, None, None

# --- RECOMMENDATION LOGIC ---
def out_recommend_motorbike(text, top_k, data, vectorizer, tfidf_matrix, teencode, emoji, stopwords):
    text_processed = preprocess_text(text, teencode, emoji, stopwords)
    text_feat = vectorizer.transform([text_processed])
    cosine_scores = cosine_similarity(tfidf_matrix, text_feat).flatten()
    best_indexes = cosine_scores.argsort()[-top_k:][::-1]
    result = data.iloc[best_indexes].copy()
    result["cosine_score"] = cosine_scores[best_indexes]
    return result

def recommend(df, cosine_sim, idx, k):
    sim_scores = cosine_sim[idx]
    best_k_indexes = sim_scores.argsort()[-(k+1):-1][::-1] 
    cosine_scores = sim_scores[best_k_indexes]
    result = df.iloc[best_k_indexes].copy()
    result['cosine_score'] = cosine_scores
    return result

# --- CLUSTERING LOGIC ---
def run_clustering_inference(df, scaler, model, iso_model, mode):
    """
    Returns: 
    - labels: Cluster labels
    - X_visual: Data for PCA
    - outliers: Boolean array (True if outlier) -1 from IsoForest
    """
    X_visual = None
    outliers = None

    # 1. Detect Outliers (if model exists)
    if iso_model and mode == 'numeric':
        X_raw = df.values
        # Note: Check if iso_model expects scaled or raw data. 
        # In notebook: outlier_detector.fit_predict(X_scaled)
        # So we must scale first
        X_scaled_iso = scaler.transform(X_raw)
        iso_preds = iso_model.predict(X_scaled_iso) 
        outliers = iso_preds == -1 # True if outlier
    else:
        outliers = np.zeros(len(df), dtype=bool) # No outliers detected

    # 2. Clustering
    if mode == 'numeric':
        X = df.values
        X_scaled = scaler.transform(X)
        labels = model.predict(X_scaled)
        X_visual = X_scaled
    elif mode == 'mixed':
        cat_cols = ["Thương hiệu", "Dòng xe", "Loại xe", "Xuất xứ"]
        num_cols = ["Khoảng giá min", "Khoảng giá max", "Năm đăng ký", "Số Km đã đi", "Giá"]
        
        X_cat = df[cat_cols].to_numpy()
        X_num = df[num_cols].to_numpy()
        X_num_scaled = scaler.transform(X_num)
        
        X_combined = np.concatenate([X_cat, X_num_scaled], axis=1)
        categorical_idx = [0, 1, 2, 3]
        
        labels = model.predict(X_combined, categorical=categorical_idx)
        X_visual = X_num_scaled # PCA on numeric only
        
    return labels, X_visual, outliers

# --- METRICS & PROFILING ---
@st.cache_data
def calculate_cluster_profiles(df, labels):
    """Calculates mean values for numeric cols per cluster"""
    df_temp = df.copy()
    df_temp['Cluster'] = labels
    
    # Lấy danh sách cột số
    numeric_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
    
    # QUAN TRỌNG: Loại bỏ cột 'Cluster' ra khỏi danh sách tính toán trung bình
    if 'Cluster' in numeric_cols:
        numeric_cols.remove('Cluster')
        
    # Groupby và tính mean trên các cột còn lại
    return df_temp.groupby('Cluster')[numeric_cols].mean().reset_index()

@st.cache_data
def calculate_silhouette_metrics(X_scaled, labels):
    # Sampling for performance
    if len(X_scaled) > 5000:
        indices = np.random.choice(len(X_scaled), 5000, replace=False)
        X_sample = X_scaled[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X_scaled
        labels_sample = labels
        
    silhouette_avg = silhouette_score(X_sample, labels_sample)
    sample_silhouette_values = silhouette_samples(X_sample, labels_sample)
    return silhouette_avg, sample_silhouette_values, labels_sample

def predict_new_sample(input_data, scaler, model, mode_type):
    try:
        if mode_type == 'numeric':
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df.values)
            label = model.predict(input_scaled)[0]
            return label
            
        elif mode_type == 'mixed':
            cat_cols = ["Thương hiệu", "Dòng xe", "Loại xe", "Xuất xứ"]
            num_cols = ["Khoảng giá min", "Khoảng giá max", "Năm đăng ký", "Số Km đã đi", "Giá"]
            
            input_cat = np.array([[input_data[c] for c in cat_cols]])
            input_num = np.array([[input_data[c] for c in num_cols]])
            input_num_scaled = scaler.transform(input_num)
            
            input_combined = np.concatenate([input_cat, input_num_scaled], axis=1)
            categorical_idx = [0, 1, 2, 3]
            
            label = model.predict(input_combined, categorical=categorical_idx)[0]
            return label
    except Exception as e:
        return None