# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:24:20 2023

@author: 20181846
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet

climate_change = pd.read_csv("API_19_DS2_en_csv_v2_5361599.csv", skiprows=4)

renewable = climate_change.loc[(climate_change["Indicator Name"]
    == "Renewable energy consumption (% of total final energy consumption)")]
print(renewable.describe())

renewable2 = renewable[["1990", "2000", "2010", "2015", "2019"]]
print(renewable.describe())

# Create correlation map of renewable energy consumption
corr = renewable2.corr()
print(corr)

ct.map_corr(renewable2)
plt.title("Correlation map of renewable energy consumption for all countries")
plt.savefig("heatmap_renewable.png", dpi=300, bbox_inches="tight")
plt.show()

# Create scatter matric of renewable energy consumption
pd.plotting.scatter_matrix(renewable2, figsize=(12, 12), s=5, alpha=0.8)
plt.suptitle(
    "Scatter matrix of renewable energy consumption for all countries",
    fontsize=30)
plt.savefig("scattermatrix_renewable.png", dpi=300, bbox_inches="tight")
plt.show()

# Extract the columns that will be used for clustering
renewable3 = renewable2[["1990", "2019"]]
renewable3 = renewable3.dropna()
renewable3 = renewable3.reset_index()
renewable3 = renewable3.drop("index", axis=1)

# Normalize, store min and max, and calculate silhouette score
df_norm, df_min, df_max = ct.scaler(renewable3)
print("n score")
for ncluster in range(2, 10):
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_norm)
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    print(ncluster, skmet.silhouette_score(renewable3, labels))

# Perform normalized K-means clustering with optimal number of clusters
ncluster = 6
kmeans = cluster.KMeans(n_clusters=ncluster)
kmeans.fit(df_norm)
labels = kmeans.labels_
cen = kmeans.cluster_centers_
cen = np.array(cen)
xcen = cen[:, 0]
ycen = cen[:, 1]

plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_norm["1990"], df_norm["2019"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.title("Normalized K-means clustering of renewable energy consumption")
plt.xlabel("Renewable energy consumption (1990)")
plt.ylabel("Renewable energy consumption (2019)")
plt.savefig("cluster_renewable_norm.png", dpi=300, bbox_inches="tight")
plt.show()

# Display clustering results with original values
print(cen)
scen = ct.backscale(cen, df_min, df_max)
print()
print(scen)
xcen = scen[:, 0]
ycen = scen[:, 1]
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(renewable3["1990"], renewable3["2019"], 10, labels, marker="o",
            cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.title("K-means clustering of renewable energy consumption")
plt.xlabel("Renewable energy consumption (1990)")
plt.ylabel("Renewable energy consumption (2019)")
plt.savefig("cluster_renewable.png", dpi=300, bbox_inches="tight")
plt.show()