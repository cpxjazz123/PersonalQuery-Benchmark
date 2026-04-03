#!/usr/bin/env python3
"""
用户5D复杂度特征可视化
- 雷达图：每个用户的5D profile
- 热力图：所有用户的5D特征矩阵
- PCA降维：2D可视化用户分布
- K-Means聚类：用户风格分群
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
INPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/06_query"
OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/06_query/visualization"

# 5D维度
DIMENSIONS = ['clause_depth', 'dep_distance', 'modifier_density', 'coord_chain', 'negation_scope']
DIM_LABELS = ['Clause\nDepth', 'Dep\nDistance', 'Modifier\nDensity', 'Coord\nChain', 'Negation\nScope']

def load_all_users():
    """加载所有用户的5D特征"""
    files = glob.glob(os.path.join(INPUT_DIR, "queries_*.json"))
    users_data = []

    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)

        user_id = data.get('user_id', '')
        target = data.get('target_complexity', {})

        if target:
            features = {dim: target.get(dim, 0) for dim in DIMENSIONS}
            features['user_id'] = user_id
            features['avg_sentence_length'] = target.get('avg_sentence_length', 0)
            features['num_reviews'] = data.get('num_reviews_total', 0)
            features['num_long_sentences'] = data.get('num_long_sentences', 0)
            users_data.append(features)

    return pd.DataFrame(users_data)

def plot_radar_chart(df, output_path):
    """绘制雷达图 - 每个用户一个子图"""
    n_users = len(df)
    n_cols = 3
    n_rows = (n_users + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), subplot_kw=dict(polar=True))
    axes = axes.flatten() if n_users > 1 else [axes]

    # 雷达图的角度
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes[idx]

        values = [row[dim] for dim in DIMENSIONS]
        values += values[:1]  # 闭合

        ax.plot(angles, values, 'o-', linewidth=2, color=plt.cm.tab10(idx % 10))
        ax.fill(angles, values, alpha=0.25, color=plt.cm.tab10(idx % 10))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(DIM_LABELS, size=8)
        ax.set_ylim(0, 0.5)
        ax.set_title(f"{row['user_id'][:16]}\n({row['num_long_sentences']} sentences)", size=9, pad=10)

    # 隐藏多余的子图
    for idx in range(n_users, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('User 5D Complexity Profiles - Radar Chart', size=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"雷达图已保存: {output_path}")

def plot_heatmap(df, output_path):
    """绘制热力图 - 所有用户的5D特征"""
    fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.6)))

    # 准备数据
    plot_df = df.set_index('user_id')[DIMENSIONS].copy()
    plot_df.columns = DIM_LABELS

    # 归一化到0-1便于显示
    for col in DIM_LABELS:
        max_val = plot_df[col].max()
        if max_val > 0:
            plot_df[col] = plot_df[col] / max_val

    # 绘制热力图
    im = ax.imshow(plot_df.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # 设置刻度
    ax.set_xticks(range(len(DIM_LABELS)))
    ax.set_xticklabels(DIM_LABELS, rotation=45, ha='right', size=10)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df.index, size=8)

    # 添加数值标注
    for i in range(len(plot_df)):
        for j in range(len(DIM_LABELS)):
            val = df.iloc[i][DIMENSIONS[j]]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', size=7,
                   color='white' if plot_df.iloc[i, j] > 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='Normalized Value')
    plt.title('User 5D Complexity Features - Heatmap', size=14, pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"热力图已保存: {output_path}")

def plot_pca(df, output_path):
    """PCA降维可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 准备数据
    X = df[DIMENSIONS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 子图1: PCA散点图
    ax = axes[0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=range(len(df)), cmap='tab10', s=100, alpha=0.7)
    for i, user_id in enumerate(df['user_id']):
        ax.annotate(user_id[:12], (X_pca[i, 0], X_pca[i, 1]), fontsize=7, alpha=0.7)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', size=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', size=11)
    ax.set_title('PCA - User Distribution', size=12)
    ax.grid(True, alpha=0.3)

    # 子图2: PCA成分重要性
    ax = axes[1]
    importance = np.abs(pca.components_[0]) + np.abs(pca.components_[1])
    importance = importance / importance.sum()
    bars = ax.bar(DIM_LABELS, importance, color='steelblue', alpha=0.8)
    ax.set_ylabel('Importance', size=11)
    ax.set_title('PCA - Feature Importance', size=12)
    ax.set_xticklabels(DIM_LABELS, rotation=45, ha='right')
    for bar, imp in zip(bars, importance):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{imp:.2%}', ha='center', va='bottom', size=9)

    plt.suptitle('PCA Dimensionality Reduction', size=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PCA图已保存: {output_path}")
    print(f"  PC1+PC2解释方差: {pca.explained_variance_ratio_.sum()*100:.1f}%")

def plot_kmeans(df, output_path):
    """K-Means聚类可视化"""
    # 准备数据
    X = df[DIMENSIONS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 确定最优聚类数（2-6）
    inertias = []
    K_range = range(2, min(7, len(df)))
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # 选择聚类数（简单用3或数据量的一半）
    n_clusters = min(3, len(df) - 1)

    # 执行K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['cluster'] = clusters

    # PCA降维用于可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1: 聚类结果
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for c in range(n_clusters):
        mask = clusters == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[c]], s=100,
                  label=f'Cluster {c}', alpha=0.7, edgecolors='white')
        for i in np.where(mask)[0]:
            ax.annotate(df.iloc[i]['user_id'][:10], (X_pca[i, 0], X_pca[i, 1]),
                       fontsize=6, alpha=0.7)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', size=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', size=11)
    ax.set_title(f'K-Means Clustering (k={n_clusters})', size=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 子图2: 聚类中心雷达图
    ax = axes[1]
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]

    for c in range(n_clusters):
        center = kmeans.cluster_centers_[c]
        # 反标准化得到原始尺度的值
        center_original = center * scaler.scale_ + scaler.mean_
        # 归一化到0-1（简化显示）
        center_normalized = center_original / np.array([0.3, 0.15, 0.25, 0.2, 0.2])
        center_normalized = np.clip(center_normalized, 0, 1)
        center_normalized = center_normalized.tolist()
        center_normalized += center_normalized[:1]

        ax.plot(angles, center_normalized, 'o-', linewidth=2, color=colors[c],
               label=f'Cluster {c}')
        ax.fill(angles, center_normalized, alpha=0.15, color=colors[c])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIM_LABELS, size=9)
    ax.set_ylim(0, 1.2)
    ax.set_title('Cluster Centers - Radar Chart', size=12)
    ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('K-Means User Clustering', size=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"K-Means聚类图已保存: {output_path}")
    print(f"  聚类数: {n_clusters}")
    print(f"  各簇用户数: {dict(zip(*np.unique(clusters, return_counts=True)))}")

    return df[['user_id', 'cluster']]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("用户5D复杂度特征可视化")
    print("=" * 60)

    # 加载数据
    df = load_all_users()
    print(f"\n共加载 {len(df)} 个用户的5D特征")
    print(f"数据维度: {df[DIMENSIONS].describe().round(4)}")

    # 1. 雷达图
    print("\n[1/4] 绘制雷达图...")
    plot_radar_chart(df, os.path.join(OUTPUT_DIR, "radar_chart.png"))

    # 2. 热力图
    print("\n[2/4] 绘制热力图...")
    plot_heatmap(df, os.path.join(OUTPUT_DIR, "heatmap.png"))

    # 3. PCA降维
    print("\n[3/4] PCA降维分析...")
    plot_pca(df, os.path.join(OUTPUT_DIR, "pca.png"))

    # 4. K-Means聚类
    print("\n[4/4] K-Means聚类分析...")
    cluster_df = plot_kmeans(df, os.path.join(OUTPUT_DIR, "kmeans.png"))

    # 保存聚类结果
    cluster_df.to_csv(os.path.join(OUTPUT_DIR, "user_clusters.csv"), index=False)
    print(f"\n聚类结果已保存: {os.path.join(OUTPUT_DIR, 'user_clusters.csv')}")

    # 打印每个簇的特征均值
    print("\n" + "=" * 60)
    print("各簇5D特征均值:")
    print("=" * 60)
    df_with_cluster = df.merge(cluster_df, on='user_id')
    for c in sorted(df_with_cluster['cluster'].unique()):
        cluster_data = df_with_cluster[df_with_cluster['cluster'] == c]
        print(f"\nCluster {c} ({len(cluster_data)} 用户):")
        for dim in DIMENSIONS:
            print(f"  {dim}: {cluster_data[dim].mean():.4f}")

    print("\n" + "=" * 60)
    print(f"所有可视化结果已保存到: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()