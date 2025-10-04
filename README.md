# iFood Customer Segmentation

## Project Overview
Segment iFood customers using **K-Means clustering** based on their income, spending patterns, and purchase behavior to identify distinct customer groups.

## Features Used
- Income
- TotalSpend (sum of all product categories)
- NumDealsPurchases, NumWebPurchases, NumStorePurchases, NumCatalogPurchases
- Recency
- TotalKids (Kidhome + Teenhome)

## Steps
1. Load and explore the dataset
2. Data cleaning and feature engineering
3. Standardize numerical features
4. Use Elbow Method to determine optimal number of clusters
5. Apply K-Means clustering
6. Visualize clusters using PCA
7. Analyze and interpret cluster profiles

## Libraries Required
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install libraries using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
