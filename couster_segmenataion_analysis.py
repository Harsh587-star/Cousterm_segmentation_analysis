# ===========================
# Customer Segmentation with K-Means
# IDLE Compatible Version
# ===========================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ===========================
# Step 1: Load CSV
# ===========================
file_path = r"C:\Users\hr479\Downloads\ifood_df.csv"
df = pd.read_csv(file_path)
print("✅ Dataset loaded successfully!")

# ===========================
# Step 2: Explore Dataset
# ===========================
print(df.info())
print(df.describe().T)
print("\nMissing values:\n", df.isnull().sum())

# ===========================
# Step 3: Clean Data
# ===========================
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# Fill missing Income with median
if df['Income'].isnull().sum() > 0:
    df['Income'] = df['Income'].fillna(df['Income'].median())

# ===========================
# Step 4: Feature Engineering
# ===========================
df['TotalSpend'] = (
    df['MntFishProducts'] + df['MntMeatProducts'] +
    df['MntFruits'] + df['MntSweetProducts'] +
    df['MntWines'] + df['MntGoldProds']
)

df['TotalKids'] = df['Kidhome'] + df['Teenhome']

features = [
    'Income', 'TotalSpend', 'NumDealsPurchases',
    'NumWebPurchases', 'NumStorePurchases',
    'NumCatalogPurchases', 'Recency', 'TotalKids'
]

X = df[features]

# ===========================
# Step 5: Standardization
# ===========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===========================
# Step 6: Determine Optimal Clusters (Elbow Method)
# ===========================
wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(2, 11), wcss, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# ===========================
# Step 7: Apply K-Means Clustering
# ===========================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ===========================
# Step 8: PCA for Visualization
# ===========================
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = pca_result[:,0], pca_result[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2")
plt.title("Customer Segments")
plt.show()

# ===========================
# Step 9: Cluster Analysis
# ===========================
cluster_summary = df.groupby("Cluster")[features + ["TotalSpend"]].mean()
print("\nCluster Profile:\n", cluster_summary)

print("\nCustomer Count per Cluster:\n", df['Cluster'].value_counts())

# ===========================
# Step 10: Feature Visualization per Cluster
# ===========================
plt.figure(figsize=(10,6))
sns.barplot(x="Cluster", y="Income", data=df, ci=None)
plt.title("Average Income per Cluster")
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x="Cluster", y="TotalSpend", data=df, ci=None)
plt.title("Average Spending per Cluster")
plt.show()
