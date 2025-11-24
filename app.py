import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# Import functions from utils
from utils import (
    load_resources, 
    load_recommender_system, 
    load_clustering_model, 
    run_clustering_inference,
    out_recommend_motorbike, 
    recommend,
    calculate_silhouette_metrics,
    calculate_cluster_profiles,
    predict_new_sample
)

# --- 1. HELPER FUNCTION FOR DEEP ANALYSIS (MATPLOTLIB) ---
def draw_comparison_analysis(X_scaled):
    """
    Recreates the detailed 3x3 Matplotlib comparison chart (K=2 vs K=3)
    """
    # 1. Re-calculate K=2 and K=3 on the fly for the report
    with st.spinner("Computing Deep Analysis Report (K=2 vs K=3)..."):
        # K=2
        model_k2 = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels_k2 = model_k2.fit_predict(X_scaled)
        
        # K=3
        model_k3 = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_k3 = model_k3.fit_predict(X_scaled)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

    # 2. Drawing
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    fig = plt.figure(figsize=(18, 12))

    # --- ROW 1: K=2 ---
    # 1. Cluster sizes (K=2)
    ax1 = plt.subplot(3, 3, 1)
    k2_sizes = [np.sum(labels_k2 == i) for i in range(2)]
    colors_k2 = ['#FF6B6B', '#4ECDC4']
    bars1 = ax1.bar(range(2), k2_sizes, color=colors_k2, alpha=0.7, edgecolor='black')
    ax1.set_title('K=2: Cluster Sizes', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(2))
    ax1.set_ylabel('Samples')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}\n({height/len(labels_k2)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. Silhouette (K=2)
    ax2 = plt.subplot(3, 3, 2)
    # Downsample for speed if needed
    if len(X_scaled) > 5000:
        idx = np.random.choice(len(X_scaled), 5000, replace=False)
        X_sample = X_scaled[idx]
        l2_sample = labels_k2[idx]
        l3_sample = labels_k3[idx]
    else:
        X_sample = X_scaled
        l2_sample = labels_k2
        l3_sample = labels_k3
        
    k2_silhouettes = [silhouette_samples(X_sample, l2_sample)[l2_sample == i].mean() for i in range(2)]
    bars2 = ax2.bar(range(2), k2_silhouettes, color=colors_k2, alpha=0.7, edgecolor='black')
    ax2.set_title('K=2: Avg Silhouette Scores', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold 0.5')
    ax2.legend()
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. PCA (K=2)
    ax3 = plt.subplot(3, 3, 3)
    for i in range(2):
        mask = labels_k2 == i
        ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors_k2[i], label=f'C{i}', alpha=0.6, s=15, edgecolors='white', linewidth=0.5)
    ax3.set_title('K=2: PCA View', fontsize=12, fontweight='bold')
    ax3.legend()

    # --- ROW 2: K=3 ---
    # 4. Cluster sizes (K=3)
    ax4 = plt.subplot(3, 3, 4)
    k3_sizes = [np.sum(labels_k3 == i) for i in range(3)]
    colors_k3 = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    bars3 = ax4.bar(range(3), k3_sizes, color=colors_k3, alpha=0.7, edgecolor='black')
    ax4.set_title('K=3: Cluster Sizes', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(3))
    ax4.set_ylabel('Samples')
    for bar in bars3:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}\n({height/len(labels_k3)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 5. Silhouette (K=3)
    ax5 = plt.subplot(3, 3, 5)
    k3_silhouettes = [silhouette_samples(X_sample, l3_sample)[l3_sample == i].mean() for i in range(3)]
    bars4 = ax5.bar(range(3), k3_silhouettes, color=colors_k3, alpha=0.7, edgecolor='black')
    ax5.set_title('K=3: Avg Silhouette Scores', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 1)
    ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    for bar in bars4:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 6. PCA (K=3)
    ax6 = plt.subplot(3, 3, 6)
    for i in range(3):
        mask = labels_k3 == i
        ax6.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors_k3[i], label=f'C{i}', alpha=0.6, s=15, edgecolors='white', linewidth=0.5)
    ax6.set_title('K=3: PCA View', fontsize=12, fontweight='bold')
    ax6.legend()

    # --- ROW 3: COMPARISON ---
    # 7. Pie K=2
    ax7 = plt.subplot(3, 3, 7)
    ax7.pie(k2_sizes, labels=[f'Cluster {i}' for i in range(2)], colors=colors_k2, autopct='%1.1f%%', startangle=90)
    ax7.set_title('K=2 Distribution', fontsize=12, fontweight='bold')

    # 8. Pie K=3
    ax8 = plt.subplot(3, 3, 8)
    ax8.pie(k3_sizes, labels=[f'Cluster {i}' for i in range(3)], colors=colors_k3, autopct='%1.1f%%', startangle=90)
    ax8.set_title('K=3 Distribution', fontsize=12, fontweight='bold')

    # 9. Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    comparison_data = [
        ['Metric', 'K=2', 'K=3'],
        ['Avg Silhouette', f'{np.mean(k2_silhouettes):.3f}', f'{np.mean(k3_silhouettes):.3f}'],
        ['Min Silhouette', f'{np.min(k2_silhouettes):.3f}', f'{np.min(k3_silhouettes):.3f}'],
        ['Max Silhouette', f'{np.max(k2_silhouettes):.3f}', f'{np.max(k3_silhouettes):.3f}'],
    ]
    table = ax9.table(cellText=comparison_data, cellLoc='center', loc='center', colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header color
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.suptitle('Deep Analysis: K=2 vs K=3 Comparison Report', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig

# --- 2. CONFIG PAGE ---
st.set_page_config(
    page_title="MotoAI Enterprise",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CUSTOM CSS ---
st.markdown("""
<style>
    /* Main Background */
    .main { background-color: #F8FAFC; font-family: 'Inter', sans-serif; }
    
    /* --- SIDEBAR FIX --- */
    [data-testid="stSidebar"] { 
        background-color: #FFFFFF !important; 
        border-right: 1px solid #E2E8F0; 
    }

    /* QUAN TRỌNG: Ép toàn bộ chữ trong Sidebar thành màu tối */
    [data-testid="stSidebar"] * {
        color: #1E293B !important; /* Màu xám đen */
    }
    
    /* Fix riêng cho nút Radio (Navigation) để nó nổi bật hơn */
    [data-testid="stSidebar"] .stRadio label {
        color: #1E293B !important;
        font-weight: 600;
    }

    /* --- HEADINGS --- */
    h1, h2, h3 { color: #0F172A; font-weight: 700; }
    
    /* Metric Box */
    .metric-box {
        text-align: center;
        padding: 15px;
        background: white;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 24px; font-weight: 800; color: #1E293B; }
    .metric-label { font-size: 13px; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 8px;
        font-weight: 600;
        color: #64748B;
        background-color: white;
        border: 1px solid #E2E8F0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
        border: none;
    }
            
    .block-container {
        padding-top: 2rem !important; /* Giảm khoảng cách trên cùng */
        padding-bottom: 2rem !important;
    }
    
    /* Recommendation Card - Grid Style */
    .rec-card-grid {
        background: white; 
        border-radius: 12px; 
        padding: 15px; 
        border: 1px solid #E2E8F0;
        transition: transform 0.2s; 
        margin-bottom: 15px; 
        height: 350px;
        display: flex; 
        flex-direction: column; 
        justify-content: space-between;
    }
    .rec-card-grid:hover { transform: translateY(-5px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
    
    /* Badge Style for Clusters */
    .cluster-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
        margin-top: 5px;
    }

    /* Fix màu chữ cho các text khác trong app chính nếu bị trắng */
    .stMarkdown, .stText, p {
        color: #1E293B; 
    }
    
    /* Button Styles */
    div.stButton > button:first-child {
        background-color: white; border: 1px solid #E2E8F0; color: #1E293B;
    }
    div.stButton > button:active { background-color: #EFF6FF; }
            
    /* CSS cho trang chi tiết */
    .detail-header {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    }
    .spec-box {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        margin-bottom: 10px;
    }
    .spec-label {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 2px;
    }
    .spec-value {
        color: #0F172A;
        font-size: 16px;
        font-weight: 600;
    }
    .price-tag {
        font-size: 28px; 
        font-weight: 800; 
        color: #10B981; /* Màu xanh lá tiền tệ */
        background: #ECFDF5;
        padding: 5px 15px;
        border-radius: 8px;
        display: inline-block;
    }
            
</style>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
teencode, emoji, stopwords = load_resources()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #3B82F6, #2563EB); border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 40px;">🏍️</span>
            </div>
            <h2 style="margin-top: 10px; color: #1E293B;">MotoAI Admin</h2>
            <p style="color: #64748B; font-size: 12px;">Analytics & Recommendation Engine</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧭 Navigation")
    app_mode = st.radio(
        "Select Module:",
        [
            "Project Introduction", 
            "User Mode (Recommender)", 
            "Admin Mode (Clustering)"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    with st.expander("ℹ️ System Info"):
        st.info("Last Updated: Nov 2025\nVersion: 2.3.0 (Dynamic GMM)")

# ==================================================
# MODULE 0: PROJECT INTRODUCTION
# ==================================================
if app_mode == "Project Introduction":
    st.markdown("""
    <div style="text-align: center; padding: 30px 0;">
        <h1 style="color: #1E3A8A; font-size: 36px;">Project Overview & Methodology</h1>
        <p style="font-size: 18px; color: #64748B;">Motorbike Recommendation & Market Segmentation Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 1. Problem Statement")
    st.markdown("""
    The used motorbike market on Chotot involves a massive volume of listings, making it difficult for users to find vehicles that match their specific needs and for sellers to understand appropriate pricing strategies. This project addresses two core challenges:
    * **Information Overload:** Users struggle to filter through thousands of unstandardized listings.
    * **Market Ambiguity:** Lack of clear price segmentation makes it hard to categorize vehicles into Budget, Mid-range, or Premium tiers.
    """)
    
    st.divider()

    col_intro_1, col_intro_2 = st.columns(2, gap="large")

    with col_intro_1:
        st.markdown("### 2. Recommendation Engine")
        st.markdown("""
        **Objective:** To provide personalized motorbike suggestions based on user search queries using Content-Based Filtering.
        
        **Methodology & Technologies:**
        * **TF-IDF Vectorization:** We utilize Term Frequency-Inverse Document Frequency to convert unstructured text descriptions (listings) into numerical vectors, highlighting unique features of each bike.
        * **Cosine Similarity:** To measure the relevance between a user's query and the database listings. The system calculates the cosine of the angle between vectors; a value closer to 1 indicates a high degree of similarity.
        * **NLP Preprocessing:** Integration of Vietnamese stopword removal, teencode normalization, and emoji handling to clean raw input data.
        """)

    with col_intro_2:
        st.markdown("### 3. Market Segmentation (Clustering)")
        st.markdown("""
        **Objective:** To group motorbikes into distinct market segments based on price, usage (odometer), and age (year).
        
        **Methodology & Technologies:**
        * **K-Means Clustering:** A robust algorithm used for partitioning numeric data (Price, Year, Km) into $K$ distinct non-overlapping subgroups (clusters).
        * **K-Prototypes:** An extension of K-Means that handles mixed data types, allowing us to cluster based on both categorical features (Brand, Type) and numerical features.
        * **Gaussian Mixture Models (GMM):** A probabilistic model that assumes all data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.
        """)
    
    st.info("**Navigation:** Use the sidebar to switch between the User Search Interface and the Admin Analytics Dashboard.")

# ==================================================
# MODULE 1: USER MODE (RECOMMENDER)
# ==================================================
elif app_mode == "User Mode (Recommender)":
    
    # --- SESSION STATE INITIALIZATION ---
    if 'page' not in st.session_state:
        st.session_state.page = 0
    if 'selected_bike' not in st.session_state:
        st.session_state.selected_bike = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None

    # Load Resources
    # Load Resources
    # Load Resources
    with st.spinner("🚀 Booting up & Analyzing GMM Segments..."):
        teencode, emoji, stopwords = load_resources()
        vectorizer, tfidf_matrix, cosine_sim, df_bikes = load_recommender_system()
        
        # --- DYNAMIC CLUSTER MAPPING (GMM + ISOLATION FOREST) ---
        # 1. Load Model & Data
        cl_scaler, cl_model, cl_iso_model, df_encoded, cl_mode = load_clustering_model("GMM")
        
        cluster_map = {}
        
        if cl_model is not None and df_encoded is not None:
            try:
                # 2. Chạy dự đoán trên toàn bộ tập dữ liệu (Training Set)
                # Lý do: Ta cần biết dữ liệu phân bố thực tế thế nào để tính "Tâm cụm thực tế"
                # Hàm run_clustering_inference đã bao gồm logic Isolation Forest (nếu có)
                labels, _, outliers = run_clustering_inference(
                    df_encoded, cl_scaler, cl_model, cl_iso_model, cl_mode
                )
                
                # 3. Xử lý Outliers (Quan trọng: Loại bỏ nhiễu khỏi việc tính toán tâm cụm)
                # Nếu Isolation Forest đánh dấu là outlier (True), ta gán nhãn -1 để lọc ra
                final_labels = labels.copy()
                if outliers is not None:
                    final_labels[outliers] = -1
                
                # 4. Tính Profile (Trung bình) của từng cụm dựa trên dữ liệu thực tế
                # Sử dụng hàm calculate_cluster_profiles có sẵn trong utils
                df_profiles = calculate_cluster_profiles(df_encoded, final_labels)
                
                # 5. Loại bỏ cụm rác (-1) nếu có
                df_profiles = df_profiles[df_profiles['Cluster'] != -1]
                
                # 6. Sắp xếp các cụm theo mức giá trung bình (Tăng dần)
                # Giả định cột 'Giá' tồn tại trong df_encoded
                if 'Giá' in df_profiles.columns:
                    df_profiles = df_profiles.sort_values(by='Giá')
                    
                    # 7. Gán nhãn phân khúc dựa trên thứ tự giá
                    labels_def = [
                        ("Tiết kiệm", "#10B981"),  # Xanh lá - Giá thấp nhất
                        ("Phổ thông", "#3B82F6"),  # Xanh dương - Giá giữa
                        ("Cao cấp",   "#F59E0B")   # Cam/Vàng - Giá cao nhất
                    ]
                    
                    # Lặp qua các cụm đã sắp xếp và gán vào map
                    for rank, row in enumerate(df_profiles.itertuples()):
                        c_id = int(row.Cluster)
                        if rank < len(labels_def):
                            name, color = labels_def[rank]
                            cluster_map[c_id] = (name, color)
                        else:
                            cluster_map[c_id] = (f"Phân khúc {rank+1}", "#64748B")
            except Exception as e:
                # Fallback nếu có lỗi xảy ra
                cluster_map = {}
        else:
            cluster_map = {}
    with st.spinner("🚀 Booting up & Analyzing GMM Segments..."):
        teencode, emoji, stopwords = load_resources()
        vectorizer, tfidf_matrix, cosine_sim, df_bikes = load_recommender_system()
        
        # --- DYNAMIC CLUSTER MAPPING (GMM) - LOGIC MỚI DỰA TRÊN CENTROIDS ---
        # 1. Load GMM Model
        cl_scaler, cl_model, _, df_encoded, cl_mode = load_clustering_model("GMM")
        
        cluster_map = {}
        
        # Chỉ chạy logic này nếu load được model GMM và tìm thấy thuộc tính means_ (centroids)
        if cl_model is not None and hasattr(cl_model, 'means_'):
            try:
                # B1: Xác định vị trí cột "Giá" trong dữ liệu training
                # df_encoded trả về từ utils đã khớp cột với lúc train model
                price_index = df_encoded.columns.get_loc("Giá")
                
                # B2: Lấy giá trị tại cột "Giá" của các tâm cụm (Centroids)
                # cl_model.means_ là mảng (n_clusters, n_features) chứa toạ độ tâm
                # Kết quả list: [(Cluster_ID_0, Giá_Tâm_0), (Cluster_ID_1, Giá_Tâm_1), ...]
                clusters_info = []
                for c_id, centroid in enumerate(cl_model.means_):
                    center_price = centroid[price_index]
                    clusters_info.append((c_id, center_price))
                
                # B3: Sắp xếp các cụm dựa trên Giá Tâm từ THẤP -> CAO
                # Cụm nào có tâm giá nhỏ nhất sẽ lên đầu
                clusters_info.sort(key=lambda x: x[1])
                
                # B4: Gán nhãn theo thứ tự
                # Rank 0 (Thấp nhất) -> Tiết kiệm
                # Rank 1 (Giữa)      -> Phổ thông
                # Rank 2 (Cao nhất)  -> Cao cấp
                labels_def = [
                    ("Tiết kiệm", "#10B981"),  # Green
                    ("Phổ thông", "#3B82F6"),  # Blue
                    ("Cao cấp",   "#F59E0B")   # Orange
                ]
                
                for rank, (c_id, val) in enumerate(clusters_info):
                    if rank < len(labels_def):
                        name, color = labels_def[rank]
                        cluster_map[c_id] = (name, color)
                    else:
                        # Dự phòng cho trường hợp K > 3
                        cluster_map[c_id] = (f"Phân khúc {rank+1}", "#64748B")
                        
            except Exception as e:
                st.error(f"Lỗi khi mapping GMM Centroids: {e}")
                cluster_map = {}
        else:
            cluster_map = {}

    # --- VIEW: DETAIL PAGE ---
    # --- VIEW: DETAIL PAGE ---
    if st.session_state.selected_bike is not None:
        bike = st.session_state.selected_bike
        
        # Nút quay lại
        if st.button("⬅️ Quay lại danh sách", type="secondary"):
            st.session_state.selected_bike = None
            st.rerun()
            
        # 1. HEADER SECTION (Tiêu đề & Giá)
        st.markdown(f"""
        <div class="detail-header">
            <h2 style="color: white; margin:0;">{bike.get('Tiêu đề', 'Chi tiết xe')}</h2>
            <div style="margin-top: 10px; display: flex; align-items: center; gap: 15px;">
                <span style="background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 5px;">📍 {bike.get('Địa chỉ', '').split(',')[-1].strip()}</span>
                <span style="background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 5px;">📅 Đăng tin: 2024</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. KEY METRICS (Hàng ngang thông số chính)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Năm sản xuất", bike.get('Năm đăng ký', 'N/A'), border=True)
        m2.metric("Odometer (Km)", f"{bike.get('Số Km đã đi', 0):,}", border=True)
        m3.metric("Phân khối", bike.get('Dung tích xe', 'N/A'), border=True)
        m4.metric("Tình trạng", bike.get('Tình trạng', 'Đã sử dụng'), border=True)

        st.markdown("---")

        # 3. MAIN CONTENT (Chia 2 cột: Specs vs Price/Action)
        c_specs, c_info = st.columns([1.5, 1], gap="large")

        with c_specs:
            st.markdown("### 🛠️ Thông số kỹ thuật")
            
            # Tạo Grid hiển thị thông số bằng HTML/CSS custom
            specs_data = {
                "Thương hiệu": bike.get('Thương hiệu', 'N/A'),
                "Dòng xe": bike.get('Dòng xe', 'N/A'),
                "Loại xe": bike.get('Loại xe', 'N/A'),
                "Xuất xứ": bike.get('Xuất xứ', 'N/A'),
                "Bảo hành": bike.get('Chính sách bảo hành', 'N/A'),
                "Trọng lượng": bike.get('Trọng lượng', 'N/A')
            }
            
            # Render Grid 2 cột cho Specs
            cols = st.columns(2)
            for i, (k, v) in enumerate(specs_data.items()):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="spec-box">
                        <div class="spec-label">{k}</div>
                        <div class="spec-value">{v}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("### 📝 Mô tả người bán")
            description = bike.get('Mô tả chi tiết', bike.get('description', 'Không có mô tả.'))
            st.markdown(f"""
            <div style="background: #F8FAFC; padding: 15px; border-radius: 8px; border-left: 4px solid #3B82F6; color: #334155;">
                {description.replace('_', '<br>• ')}
            </div>
            """, unsafe_allow_html=True)

        with c_info:
            st.markdown("### 💵 Phân tích giá")
            
            current_price_str = str(bike.get('Giá', '0')).replace('.', '').replace(' đ', '')
            try:
                current_price = float(current_price_str)
            except:
                current_price = 0
                
            st.markdown(f'<div class="price-tag">{bike.get("Giá", "Liên hệ")}</div>', unsafe_allow_html=True)
            
            # Hiển thị khoảng giá thị trường
            min_p = bike.get('Khoảng giá min', 'N/A')
            max_p = bike.get('Khoảng giá max', 'N/A')
            
            st.markdown(f"""
            <div style="margin-top: 15px; padding: 15px; border: 1px dashed #CBD5E1; border-radius: 8px;">
                <div style="color: #64748B; font-size: 13px;">Khoảng giá thị trường (tham khảo):</div>
                <div style="font-weight: bold; color: #1E293B; font-size: 18px;">{min_p} - {max_p}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Logic tính toán cụm phân khúc (Cluster) - Lấy lại logic cũ
            try:
                # Tạo data input giả lập để predict
                p_val = current_price / 1e6
                p_input = {
                    "Khoảng giá min": p_val, "Khoảng giá max": p_val, "Giá": p_val,
                    "Năm đăng ký": float(bike.get('Năm đăng ký', 2019)),
                    "Số Km đã đi": float(bike.get('Số Km đã đi', 10000))
                }
                c_id = predict_new_sample(p_input, cl_scaler, cl_model, cl_mode)
                c_name, c_color = cluster_map.get(c_id, ("Không xác định", "#94A3B8"))
                
                st.markdown(f"""
                <div style="margin-top: 10px;">
                    <span style="font-size:13px; color:#64748B;">Phân khúc AI gợi ý:</span><br>
                    <span style="background-color:{c_color}; color:white; padding: 4px 12px; border-radius: 12px; font-weight:bold; font-size:14px;">
                        {c_name}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            except:
                pass

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Nút Action
            st.markdown("### 📞 Liên hệ")
            url = bike.get('Href', bike.get('url', '#'))
            st.link_button("👉 Xem tin gốc & Gọi người bán", url, type="primary", use_container_width=True)
            
            st.info("⚠️ Lưu ý: MotoAI chỉ tổng hợp thông tin. Vui lòng kiểm tra xe thực tế trước khi giao dịch.")

    # --- VIEW: LISTING GRID (Mặc định) ---
    else:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #1E3A8A;">Find Your Dream Ride</h1>
            <p style="color: #64748B;">AI-Powered Smart Search & Segmentation</p>
        </div>
        """, unsafe_allow_html=True)

        # Search Bar
        col1, col2 = st.columns([4, 1])
        with col1:
            search_query = st.text_input("", placeholder="Example: Vision red color, cheap price...", label_visibility="collapsed")
        with col2:
            if st.button("🔍 Search", type="primary", use_container_width=True):
                st.session_state.page = 0
                
        # Logic Search
        if search_query:
            # Get recommender results (Top 200 matches)
            results = out_recommend_motorbike(search_query, 200, df_bikes, vectorizer, tfidf_matrix, teencode, emoji, stopwords)
            st.session_state.search_results = results
        elif st.session_state.search_results is None:
            # Default: Use the RAW dataframe (Full List)
            st.session_state.search_results = df_bikes.copy()

        # Pagination Logic
        current_df = st.session_state.search_results
        items_per_page = 12 # 3x4 layout
        total_items = len(current_df)
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
        
        # Safety check for page index
        if st.session_state.page >= total_pages: st.session_state.page = total_pages - 1
        if st.session_state.page < 0: st.session_state.page = 0
        
        start_idx = st.session_state.page * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        page_items = current_df.iloc[start_idx:end_idx]

        st.markdown(f"**Showing {start_idx+1}-{end_idx} of {total_items} results**")

        # --- GRID DISPLAY 3x4 ---
        # Create chunks of 3 for the columns
        rows = [page_items.iloc[i:i+3] for i in range(0, len(page_items), 3)]
        
        for row_items in rows:
            cols = st.columns(3)
            for i, (idx, row) in enumerate(row_items.iterrows()):
                with cols[i]:
                    # Extract Info
                    title = row.get('title', row.get('Tiêu đề', 'Motorbike'))
                    price_str = row.get('Giá', '0')
                    loc = row.get('Tỉnh thành', 'Vietnam')
                    
                    # Predict Cluster Logic
                    try:
                        p_val = float(str(price_str).replace('.','').replace(' đ','')) / 1e6 if isinstance(price_str, str) else 0
                        p_input = {
                            "Khoảng giá min": p_val,
                            "Khoảng giá max": p_val,
                            "Giá": p_val,
                            "Năm đăng ký": float(row.get('Năm đăng ký', 2019)),
                            "Số Km đã đi": float(row.get('Số Km đã đi', 10000))
                        }
                        c_id = predict_new_sample(p_input, cl_scaler, cl_model, cl_mode)
                        seg_name, seg_color = cluster_map.get(c_id, ("N/A", "#666"))
                    except:
                        seg_name, seg_color = "Checking...", "#666"

                    # Card UI
                    st.markdown(f"""
                    <div class="rec-card-grid">
                        <div>
                            <div style="font-weight:bold; color:#1E40AF; height:45px; overflow:hidden; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;">{title}</div>
                            <div style="color:#059669; font-weight:bold; margin-top:5px; font-size:16px;">💰 {price_str}</div>
                            <div style="font-size:12px; color:#64748B;">📍 {loc}</div>
                            <span class="cluster-badge" style="background-color:{seg_color}20; color:{seg_color};">🛡️ {seg_name}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"View Details", key=f"btn_{idx}", use_container_width=True):
                        st.session_state.selected_bike = row
                        st.rerun()

        # --- PAGINATION CONTROLS ---
        st.markdown("<br>", unsafe_allow_html=True)
        c_prev, c_page, c_next = st.columns([1, 2, 1])
        with c_prev:
            if st.session_state.page > 0:
                if st.button("⬅️ Previous Page"):
                    st.session_state.page -= 1
                    st.rerun()
        with c_page:
            st.markdown(f"<div style='text-align:center; padding-top:5px;'>Page <b>{st.session_state.page + 1}</b> / {total_pages}</div>", unsafe_allow_html=True)
        with c_next:
            if st.session_state.page < total_pages - 1:
                if st.button("Next Page ➡️"):
                    st.session_state.page += 1
                    st.rerun()

# ==================================================
# MODULE 2: ADMIN MODE (CLUSTERING)
# ==================================================
elif app_mode == "Admin Mode (Clustering)":
    st.markdown("## 📊 Market Segmentation & Analytics")

    # Controls
    with st.container():
        c1, c2, c3 = st.columns([3, 2.5, 2], gap="medium") 
        
        with c1:
            model_choice = st.selectbox("Algorithm Selection:", 
                                      ["KMeans", "GMM", "K-Prototypes"])
        with c2:
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            show_outliers = st.toggle("Show Outliers (Isolation Forest)", value=False)
        with c3:
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-bottom: 5px; color: #64748B; font-size: 14px;">
                Current Engine: <span style="font-weight: 600; color: #3B82F6;">{model_choice}</span>
            </div>
            """, unsafe_allow_html=True)

    # Load Model
    with st.spinner("Processing Algorithms..."):
        scaler, model, iso_model, df_encoded, mode_type = load_clustering_model(model_choice)

    if model:
        # Run Inference
        labels, X_visual, outliers = run_clustering_inference(df_encoded, scaler, model, iso_model, mode_type)
        
        # Filter Data logic
        df_display = df_encoded.copy()
        
        if show_outliers and outliers is not None:
            final_labels = np.where(outliers, -1, labels)
            df_display['Cluster'] = final_labels
            n_outliers = sum(outliers)
        else:
            if outliers is not None:
                clean_mask = ~outliers
                df_display = df_display[clean_mask]
                final_labels = labels[clean_mask]
                df_display['Cluster'] = final_labels
                X_visual = X_visual[clean_mask]
                n_outliers = sum(outliers)
            else:
                df_display['Cluster'] = labels
                final_labels = labels
                n_outliers = 0

        # --- KPI CARDS ---
        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        
        n_samples = len(df_display)
        unique_clusters = np.unique(final_labels)
        n_clusters = len(unique_clusters[unique_clusters != -1])
        
        k1.markdown(f"""<div class="metric-box" style="border-color: #3B82F6;"><div class="metric-value">{n_samples:,}</div><div class="metric-label">Analyzed Samples</div></div>""", unsafe_allow_html=True)
        k2.markdown(f"""<div class="metric-box" style="border-color: #10B981;"><div class="metric-value">{n_clusters}</div><div class="metric-label">Active Clusters</div></div>""", unsafe_allow_html=True)
        k3.markdown(f"""<div class="metric-box" style="border-color: #F59E0B;"><div class="metric-value">{n_outliers}</div><div class="metric-label">Outliers Detected</div></div>""", unsafe_allow_html=True)
        k4.markdown(f"""<div class="metric-box" style="border-color: #8B5CF6;"><div class="metric-value">{model_choice}</div><div class="metric-label">Algorithm</div></div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # --- TABS ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Dashboard Analysis", 
            "🧬 Cluster Profiling", 
            "🎯 Prediction Simulator",
            "🔬 Model Evaluation"
        ])

        # --- TAB 1: DASHBOARD ---
        with tab1:
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("##### 📊 Cluster Sizes")
                size_counts = df_display['Cluster'].value_counts().reset_index()
                size_counts.columns = ['Cluster', 'Count']
                size_counts['Cluster'] = size_counts['Cluster'].replace({-1: 'Outlier'})
                
                fig_size = px.bar(size_counts, x='Cluster', y='Count', color='Cluster', 
                                text='Count', template='plotly_white',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_size.update_traces(textposition='outside')
                st.plotly_chart(fig_size, use_container_width=True)
                
            with c_right:
                st.markdown("##### 🌒 Silhouette Score (Estimate)")
                if st.button("Calculate Silhouette (Interactive)"):
                    with st.spinner("Calculating..."):
                        if len(np.unique(final_labels)) > 1:
                            avg_sil, sil_samples, sil_labels = calculate_silhouette_metrics(X_visual, final_labels)
                            sil_df = pd.DataFrame({'Cluster': sil_labels, 'Score': sil_samples})
                            avg_sil_per_cluster = sil_df.groupby('Cluster')['Score'].mean().reset_index()
                            
                            fig_sil = px.bar(avg_sil_per_cluster, x='Cluster', y='Score', color='Cluster',
                                             template='plotly_white', range_y=[-0.1, 1])
                            fig_sil.add_hline(y=avg_sil, line_dash="dash", annotation_text=f"Avg: {avg_sil:.3f}")
                            st.plotly_chart(fig_sil, use_container_width=True)
                        else:
                            st.warning("Not enough clusters to calculate Silhouette.")
                else:
                    st.info("Click to calculate (resource intensive).")

            c_pie, c_pca = st.columns([1, 2])
            with c_pie:
                st.markdown("##### 🥧 Distribution")
                fig_pie = px.pie(df_display, names='Cluster', hole=0.4,
                               color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with c_pca:
                st.markdown("##### 🗺️ PCA Visualization (2D)")
                pca = PCA(n_components=2)
                if len(X_visual) > 0:
                    components = pca.fit_transform(X_visual)
                    df_pca = pd.DataFrame(components, columns=['PC1', 'PC2'])
                    df_pca['Cluster'] = final_labels
                    df_pca['Cluster'] = df_pca['Cluster'].astype(str)
                    
                    fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                                       template='plotly_white', opacity=0.7,
                                       color_discrete_sequence=px.colors.qualitative.Bold)
                    st.plotly_chart(fig_pca, use_container_width=True)

        # --- TAB 2: PROFILING ---
        with tab2:
            st.markdown("##### 🔍 Cluster Characteristic Profiling")
            centroids = calculate_cluster_profiles(df_display, final_labels)
            
            st.markdown("**Feature Heatmap (Normalized)**")
            if not centroids.empty:
                heatmap_data = centroids.set_index('Cluster')
                heatmap_data = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
                fig_heat = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
                st.plotly_chart(fig_heat, use_container_width=True)
                
                st.markdown("**Detailed Data Table**")
                st.dataframe(centroids.style.background_gradient(cmap='Blues'), use_container_width=True)
            else:
                st.warning("No data to profile.")

        # --- TAB 3: SIMULATOR ---
        with tab3:
            st.markdown("##### 🔮 Classification Simulator")
            with st.form("sim_form"):
                col_inp1, col_inp2 = st.columns(2)
                input_data = {}
                
                if mode_type == 'mixed':
                    with col_inp1:
                        input_data["Thương hiệu"] = st.selectbox("Brand", ["Honda", "Yamaha", "Piaggio", "Suzuki", "SYM"])
                        input_data["Dòng xe"] = st.text_input("Model", "Vision")
                        input_data["Loại xe"] = st.selectbox("Type", ["Tay ga", "Xe số", "Tay côn"])
                        input_data["Xuất xứ"] = st.selectbox("Origin", ["Việt Nam", "Nhập khẩu"])
                    with col_inp2:
                        input_data["Khoảng giá min"] = st.number_input("Min Price Range", 0.0, 1000.0, 10.0)
                        input_data["Khoảng giá max"] = st.number_input("Max Price Range", 0.0, 1000.0, 50.0)
                        input_data["Năm đăng ký"] = st.slider("Year", 2010, 2025, 2020)
                        input_data["Số Km đã đi"] = st.number_input("Odometer", 0, 100000, 5000)
                        input_data["Giá"] = st.number_input("Price", 0.0, 1000.0, 30.0)
                else:
                    st.info("Numeric Input Mode")
                    for col in df_encoded.select_dtypes(include=np.number).columns:
                        if col != 'Cluster':
                            input_data[col] = st.number_input(col, value=0.0)

                submit_btn = st.form_submit_button("Predict")
            
            if submit_btn:
                pred = predict_new_sample(input_data, scaler, model, mode_type)
                st.success(f"Predicted Cluster: {pred}")

        # --- TAB 4: EVALUATION ---
        with tab4:
            st.markdown("##### 🔬 Deep Comparison Analysis (K=2 vs K=3)")
            st.caption("This module re-runs KMeans to generate a detailed stability report (Static Plot).")
            
            if mode_type == 'numeric':
                col_btn, col_info = st.columns([1, 3])
                with col_btn:
                    run_deep_analysis = st.button("🚀 Run Deep Analysis Report", type="primary")
                
                if run_deep_analysis:
                    fig_comparison = draw_comparison_analysis(X_visual)
                    st.pyplot(fig_comparison)
            else:
                st.warning("Deep Analysis is optimized for Numeric Models (KMeans/GMM) only.")
