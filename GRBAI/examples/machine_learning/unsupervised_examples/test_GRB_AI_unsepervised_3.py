import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('../../../datasets/summary_general-2.txt', sep=r'\s*\|\s*', engine='python')

# Remove rows with NaN values
data = data.dropna()

# Create additional features
data['Normalized_T50'] = data['T50'] / data['T90']
data['Log_T50'] = np.log(data['T50'])
data['T50_to_T90'] = data['T50'] / data['T90']

# Prepare the feature matrix
features = ['T50', 'Normalized_T50', 'Log_T50', 'T50_to_T90']
X = data[features]

# Standardize the feature columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature Selection using mutual information
selector = SelectKBest(mutual_info_classif, k='all')  # Adjust k as needed
X_selected = selector.fit_transform(X_scaled, np.zeros(X_scaled.shape[0]))  # Dummy target variable

# Identify the selected features
selected_features = [features[i] for i in range(len(features)) if selector.get_support()[i]]
print("Selected Features:", selected_features)

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
clusters_iso = iso_forest.fit_predict(X_selected)

# Add anomaly labels to the dataset
data['Cluster_IsolationForest'] = np.where(clusters_iso == -1, 'Anomaly', 'Normal')

# Visualization: Pairplot
sns.pairplot(data, hue='Cluster_IsolationForest', vars=selected_features, palette={'Anomaly': 'red', 'Normal': 'blue'})
plt.title('Pairplot with Anomalies Highlighted')
plt.show()

# Dimensionality Reduction for visualization (if more than 2 features)
if len(selected_features) > 2:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_selected)

    plt.figure(figsize=(10, 6))
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
    plt.show()

# Histogram of Anomaly Scores
anomaly_scores = iso_forest.decision_function(X_selected)

plt.figure(figsize=(10, 6))
plt.hist(anomaly_scores[data['Cluster_IsolationForest'] == 'Normal'], bins=30, alpha=0.6, color='blue', label='Normal')
plt.hist(anomaly_scores[data['Cluster_IsolationForest'] == 'Anomaly'], bins=30, alpha=0.6, color='red', label='Anomaly')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Histogram of Anomaly Scores')
plt.legend()
plt.show()