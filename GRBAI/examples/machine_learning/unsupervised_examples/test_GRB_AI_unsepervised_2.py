import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('../../../datasets/summary_general-2.txt', sep=r'\s*\|\s*', engine='python')

# Remove rows with NaN values
data = data.dropna()

# Select numeric columns for feature extraction
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()

# Prepare the feature matrix using all numeric features
X = data[numeric_features]

# Standardize the feature columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality Reduction with PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
clusters_iso = iso_forest.fit_predict(X_pca)

# Add anomaly labels to the dataset
data['Cluster_IsolationForest'] = np.where(clusters_iso == -1, 'Anomaly', 'Normal')

# Feature importance from PCA
feature_importance = pd.DataFrame(pca.components_, columns=numeric_features, index=['PC1', 'PC2']).T
feature_importance['PC1'] = feature_importance['PC1'].abs()
feature_importance['PC2'] = feature_importance['PC2'].abs()

# Print feature importance
print("Feature Importance in PCA:")
print(feature_importance.sort_values(by='PC1', ascending=False))
print(feature_importance.sort_values(by='PC2', ascending=False))

# Visualization: 2D Projection of PCA Components with Feature Names
plt.figure(figsize=(12, 8))
plt.scatter(X_pca[data['Cluster_IsolationForest'] == 'Normal', 0], 
            X_pca[data['Cluster_IsolationForest'] == 'Normal', 1], 
            c='blue', label='Normal', alpha=0.6)
plt.scatter(X_pca[data['Cluster_IsolationForest'] == 'Anomaly', 0], 
            X_pca[data['Cluster_IsolationForest'] == 'Anomaly', 1], 
            c='red', label='Anomaly', alpha=0.8)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D Projection of Isolation Forest Results')
plt.legend()

# Annotate feature contributions
for i, feature in enumerate(numeric_features):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
              head_width=0.1, head_length=0.1, fc='k', ec='k')
    plt.text(pca.components_[0, i]*1.2, pca.components_[1, i]*1.2, 
             feature, color='k', fontsize=10)

plt.show()

# Histogram of Anomaly Scores
anomaly_scores = iso_forest.decision_function(X_pca)

plt.figure(figsize=(10, 6))
plt.hist(anomaly_scores[clusters_iso == 1], bins=30, alpha=0.6, color='red', label='Anomaly')
plt.hist(anomaly_scores[clusters_iso == 0], bins=30, alpha=0.6, color='blue', label='Normal')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Histogram of Anomaly Scores')
plt.legend()
plt.show()