#!/usr/bin/env python3
"""
用户5D复杂度特征分析脚本
直接读取已有的queries_*.json文件进行深度分析
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

INPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/06_query"
OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/06_query/analysis"
DIMENSIONS = [
    'subordinate_clause_freq', 'dep_distance', 'modifier_density', 'coord_chain',
    'negation_scope', 'voice_ratio', 'branching_direction',
    'advcl_freq', 'comp_clause_freq', 'fanout', 'parataxis_freq',
    'prep_density', 'appos_freq', 'avg_sentence_length'
]
DIM_LABELS = [
    'SubCla\nFreq', 'Dep\nDist', 'Mod\nDensity', 'Coord\nChain',
    'Negation\nScope', 'Voice\nRatio', 'Branch\nDir',
    'AdvCl\nFreq', 'Comp\nFreq', 'Fanout', 'Paratax\nFreq',
    'Prep\nDensity', 'Appos\nFreq', 'Avg\nSentLen'
]


def load_users():
    """加载所有用户数据"""
    files = glob.glob(os.path.join(INPUT_DIR, "queries_*.json"))
    data = []
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            d = json.load(f)
        target = d.get('target_complexity', {})
        if target:
            features = {dim: target.get(dim, 0) for dim in DIMENSIONS}
            features['user_id'] = d.get('user_id', '')
            features['num_reviews'] = d.get('num_reviews_total', 0)
            features['num_long_sentences'] = d.get('num_long_sentences', 0)
            data.append(features)
    return pd.DataFrame(data)


def analyze_dimension_stats(df):
    """各维度统计分析"""
    print("=" * 70)
    print("【各维度描述统计】")
    print("=" * 70)
    for dim in DIMENSIONS:
        vals = df[dim]
        cv = vals.std() / vals.mean() * 100 if vals.mean() > 0 else 0
        print(f"{dim:20s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}, CV={cv:.1f}%")


def analyze_correlation(df):
    """维度间相关性分析"""
    print("\n" + "=" * 70)
    print("【维度间相关性矩阵】")
    print("=" * 70)

    corr_matrix = df[DIMENSIONS].corr()
    print("\n相关系数矩阵:")
    print(corr_matrix.round(3).to_string())

    # 找出强相关对
    print("\n强相关对 (|r| > 0.3):")
    for i in range(len(DIMENSIONS)):
        for j in range(i+1, len(DIMENSIONS)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.3:
                print(f"  {DIMENSIONS[i]} <-> {DIMENSIONS[j]}: r = {r:.3f}")


def analyze_optimal_clusters(df):
    """寻找最优聚类数"""
    print("\n" + "=" * 70)
    print("【最优聚类数分析】")
    print("=" * 70)

    X = df[DIMENSIONS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    for k in range(2, min(11, len(df))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)
        inertia = kmeans.inertia_
        results.append({'k': k, 'silhouette': silhouette, 'inertia': inertia})
        print(f"  k={k}: 轮廓系数={silhouette:.4f}, Inertia={inertia:.2f}")

    # 找最优k
    best_k = max(results, key=lambda x: x['silhouette'])['k']
    print(f"\n  最优聚类数(轮廓系数): k = {best_k}")
    return best_k


def analyze_clusters(df, n_clusters=3):
    """聚类结果分析"""
    print("\n" + "=" * 70)
    print(f"【K-Means聚类分析 (k={n_clusters})】")
    print("=" * 70)

    X = df[DIMENSIONS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # 各簇特征
    print("\n各簇5D特征均值:")
    cluster_stats = df.groupby('cluster')[DIMENSIONS].mean()
    print(cluster_stats.round(4).to_string())

    # 各簇人数
    print(f"\n各簇人数:")
    counts = df['cluster'].value_counts().sort_index()
    for c, cnt in counts.items():
        print(f"  Cluster {c}: {cnt}人 ({cnt/len(df)*100:.1f}%)")

    # 簇间距离
    print("\n簇间距离(欧氏距离):")
    centers = kmeans.cluster_centers_
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            dist = np.linalg.norm(centers[i] - centers[j])
            print(f"  Cluster {i} vs Cluster {j}: {dist:.4f}")

    # 簇内紧凑度
    print("\n簇内紧凑度(簇内平均标准差):")
    for c in range(n_clusters):
        cluster_data = df[df['cluster'] == c][DIMENSIONS]
        avg_std = cluster_data.std().mean()
        print(f"  Cluster {c}: {avg_std:.4f}")

    return df


def plot_all(df, best_k):
    """绘制所有分析图"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X = df[DIMENSIONS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. 轮廓系数曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    sil_scores = []
    inertias = []
    for k in range(2, min(11, len(df))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, labels))
        inertias.append(kmeans.inertia_)

    ax.plot(range(2, 2+len(sil_scores)), sil_scores, 'bo-', label='Silhouette Score')
    ax.set_xlabel('Number of Clusters (k)', size=11)
    ax.set_ylabel('Silhouette Score', size=11)
    ax.set_title('Silhouette Score vs Number of Clusters', size=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(2, 2+len(sil_scores)))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "silhouette_scores.png"), dpi=150)
    plt.close()
    print(f"\n轮廓系数图已保存: {OUTPUT_DIR}/silhouette_scores.png")

    # 2. PCA双标图
    fig, ax = plt.subplots(figsize=(10, 8))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'],
                       cmap='tab10', s=50, alpha=0.6)

    # 绘制特征向量
    for i, (dim, label) in enumerate(zip(DIMENSIONS, DIM_LABELS)):
        loading = pca.components_[:, i] * 2  # 放大以便显示
        ax.arrow(0, 0, loading[0], loading[1], head_width=0.05,
                 head_length=0.02, fc='red', ec='red', alpha=0.7)
        ax.text(loading[0]*1.15, loading[1]*1.15, label, fontsize=10,
               ha='center', va='center', color='red')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', size=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', size=11)
    ax.set_title(f'PCA Biplot (PC1+PC2={pca.explained_variance_ratio_.sum()*100:.1f}%)', size=12)
    ax.grid(True, alpha=0.3)
    ax.legend(*scatter.legend_elements(), title='Cluster', loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_biplot.png"), dpi=150)
    plt.close()
    print(f"PCA双标图已保存: {OUTPUT_DIR}/pca_biplot.png")

    # 3. 特征分布箱线图
    n_dims = len(DIMENSIONS)
    n_cols = 4
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_dims > 1 else [axes]
    for i, (dim, label) in enumerate(zip(DIMENSIONS, DIM_LABELS)):
        df.boxplot(column=dim, by='cluster', ax=axes[i])
        axes[i].set_title(label, size=10)
        axes[i].set_xlabel('Cluster', size=9)
    for i in range(n_dims, len(axes)):
        axes[i].set_visible(False)
    plt.suptitle('Feature Distribution by Cluster', size=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_boxplots.png"), dpi=150)
    plt.close()
    print(f"箱线图已保存: {OUTPUT_DIR}/cluster_boxplots.png")

    # 4. 簇中心雷达图对比
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]

    for c in range(df['cluster'].nunique()):
        center = df[df['cluster'] == c][DIMENSIONS].mean().values
        center = center.tolist()
        center += center[:1]
        ax.plot(angles, center, 'o-', linewidth=2, label=f'Cluster {c}')
        ax.fill(angles, center, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIM_LABELS, size=7)
    ax.set_ylim(0, max(df[DIMENSIONS].max()) * 1.2)
    ax.set_title('Cluster Centers Radar Chart (14D)', size=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_radar.png"), dpi=150)
    plt.close()
    print(f"雷达图已保存: {OUTPUT_DIR}/cluster_radar.png")

    # 5. 热力图
    fig, ax = plt.subplots(figsize=(16, 8))
    cluster_means = df.groupby('cluster')[DIMENSIONS].mean()
    cluster_means.columns = DIM_LABELS

    # 归一化显示
    normalized = cluster_means.copy()
    for col in DIM_LABELS:
        max_val = cluster_means[col].max()
        if max_val > 0:
            normalized[col] = cluster_means[col] / max_val

    im = ax.imshow(normalized.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(DIM_LABELS)))
    ax.set_xticklabels(DIM_LABELS, rotation=45, ha='right', size=11)
    ax.set_yticks(range(len(cluster_means)))
    ax.set_yticklabels([f'Cluster {i}' for i in cluster_means.index], size=11)

    for i in range(len(cluster_means)):
        for j in range(len(DIM_LABELS)):
            val = cluster_means.iloc[i, j]
            norm_val = normalized.iloc[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', size=10,
                   color='white' if norm_val > 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='Normalized Value')
    plt.title('Cluster Centers Heatmap', size=14, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_heatmap.png"), dpi=150)
    plt.close()
    print(f"热力图已保存: {OUTPUT_DIR}/cluster_heatmap.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("用户5D复杂度特征深度分析 - 权限验证")
    print("=" * 70)

    # 加载数据
    df = load_users()
    print(f"\n共加载 {len(df)} 个用户的5D特征")

    # 1. 维度统计
    analyze_dimension_stats(df)

    # 2. 相关性分析
    analyze_correlation(df)

    # 3. 最优聚类数
    best_k = analyze_optimal_clusters(df)

    # 4. 聚类分析(使用k=2)
    n_clusters = 2
    df = analyze_clusters(df, n_clusters)

    # 5. 绘图
    print("\n" + "=" * 70)
    print("【生成分析图表】")
    print("=" * 70)
    plot_all(df, best_k)

    # 保存聚类结果
    df[['user_id', 'cluster']].to_csv(
        os.path.join(OUTPUT_DIR, "user_clusters.csv"), index=False)
    print(f"\n聚类结果已保存: {OUTPUT_DIR}/user_clusters.csv")

    print("\n" + "=" * 70)
    print(f"所有分析结果已保存到: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()