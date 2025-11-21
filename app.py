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
    
    /* Recommendation Card */
    .rec-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        transition: transform 0.2s;
        margin-bottom: 10px;
    }
    .rec-card:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
    
    /* Fix màu chữ cho các text khác trong app chính nếu bị trắng */
    .stMarkdown, .stText, p {
        color: #1E293B; 
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
        ["User Mode (Recommender)", "Admin Mode (Clustering)"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    with st.expander("ℹ️ System Info"):
        st.info("Last Updated: Nov 2025\nVersion: 2.1.0 (Stable)")

# ==================================================
# MODULE 1: USER MODE (RECOMMENDER)
# ==================================================
if app_mode == "User Mode (Recommender)":
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="color: #1E3A8A; font-size: 42px;">Find Your Dream Ride</h1>
        <p style="font-size: 18px; color: #64748B;">AI-Powered Smart Search & Matching System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize
    with st.spinner("🚀 Booting up search engine..."):
        vectorizer, tfidf_matrix, cosine_sim, df_bikes = load_recommender_system()

    # Search Bar
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input("", placeholder="Example: Vision red color, cheap price, Hanoi...", label_visibility="collapsed")
    with col2:
        num_results = st.selectbox("", [5, 10, 15, 20], index=0, label_visibility="collapsed")
    
    if st.button("🔍 SEARCH MATCHES", type="primary", use_container_width=True):
        if search_query:
            results_df = out_recommend_motorbike(search_query, num_results, df_bikes, vectorizer, tfidf_matrix, teencode, emoji, stopwords)
            
            st.markdown(f"### 🎯 Found {len(results_df)} matches")
            st.markdown("<br>", unsafe_allow_html=True)
            
            for idx, row in results_df.iterrows():
                title = row.get('title', row.get('Tiêu đề', 'No Title'))
                price = row.get('Giá', 'Contact')
                desc = row.get('description', row.get('Mô tả chi tiết', 'No description'))
                score = row['cosine_score']
                
                with st.container():
                    st.markdown(f"""
                    <div class="rec-card">
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <div>
                                <h3 style="margin: 0; color: #1E40AF; font-size: 18px;">{title}</h3>
                                <div style="margin-top: 5px; color: #059669; font-weight: 700;">
                                    💰 {price} &nbsp;•&nbsp; 📍 {row.get('Tỉnh thành', 'Vietnam')}
                                </div>
                            </div>
                            <div style="background: #EFF6FF; color: #2563EB; padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 14px;">
                                {score*100:.0f}% Match
                            </div>
                        </div>
                        <p style="color: #64748B; font-size: 14px; margin-top: 10px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">
                            {desc[:200]}...
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("View Details & Similar Bikes"):
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.markdown("**Full Description:**")
                            st.write(desc)
                        with c2:
                            st.markdown("**You might also like:**")
                            sim_items = recommend(df_bikes, cosine_sim, idx, 3)
                            for _, s_row in sim_items.iterrows():
                                s_title = s_row.get('title', 'Bike')
                                st.markdown(f"• [{s_row['cosine_score']*100:.0f}%] {s_title[:30]}...")

# ==================================================
# MODULE 2: ADMIN MODE (CLUSTERING)
# ==================================================
elif app_mode == "Admin Mode (Clustering)":
    st.markdown("## 📊 Market Segmentation & Analytics")

    # Controls (Đã sửa canh lề chuẩn pixel)
    # Controls (Sửa lỗi: Xóa vertical_alignment, dùng spacer thủ công)
    with st.container():
        c1, c2, c3 = st.columns([3, 2.5, 2], gap="medium") # <-- Đã xóa vertical_alignment để tránh lỗi
        
        with c1:
            # Cột 1: Selectbox có nhãn, nên nó cao nhất
            model_choice = st.selectbox("Algorithm Selection:", 
                                      ["KMeans", "GMM", "K-Prototypes"])
        with c2:
            # Cột 2: Chèn khoảng trắng cao 28px để đẩy Toggle xuống ngang hàng Selectbox
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            show_outliers = st.toggle("Show Outliers (Isolation Forest)", value=False)
        with c3:
            # Cột 3: Tương tự cột 2
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
            # Mark outliers as -1
            final_labels = np.where(outliers, -1, labels)
            df_display['Cluster'] = final_labels
            n_outliers = sum(outliers)
            # X_visual includes everything
        else:
            # If NOT showing outliers, we might want to filter them out or treat them as normal clusters
            # Here we just hide the outlier distinction or filter them if they were detected
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
        # Count unique clusters excluding -1 (outliers)
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
            # Top Row
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("##### 📊 Cluster Sizes")
                size_counts = df_display['Cluster'].value_counts().reset_index()
                size_counts.columns = ['Cluster', 'Count']
                # Rename -1 to 'Outlier' for display
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
                        # Cannot calc silhouette with 1 cluster or only outliers
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

            # Bottom Row
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
            # Only calculate profile on valid clusters (exclude outliers if needed, or keep them to analyze)
            centroids = calculate_cluster_profiles(df_display, final_labels)
            
            st.markdown("**Feature Heatmap (Normalized)**")
            if not centroids.empty:
                heatmap_data = centroids.set_index('Cluster')
                # Normalize
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

        # --- TAB 4: EVALUATION (NEW DEEP ANALYSIS) ---
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