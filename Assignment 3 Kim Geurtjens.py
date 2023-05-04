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
import scipy.optimize as opt
import errors as err


def read_df(filename):
    """Function that reads a dataframe in World-bank format"""
    df = pd.read_csv(filename, skiprows=4)
    return df


def transposed_df(df):
    """Function that returns a transposed dataframe"""
    df_tr = pd.DataFrame.transpose(df)
    df_tr = df_tr.dropna()
    df_tr.columns = df_tr.iloc[0]
    df_tr = df_tr.iloc[4:]
    df_tr["Year"] = df_tr.index
    df_tr = df_tr.astype(float)
    return df_tr


def poly(x, a, b, c, d, e):
    """ Calulates polynominal"""
    x = x - 2005
    f = a + b*x + c*x**2 + d*x**3 + e*x**4
    return f


climate_change = read_df("API_19_DS2_en_csv_v2_5361599.csv")

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
renewable4 = renewable3.reset_index()
renewable4 = renewable4.drop("index", axis=1)

# Normalize, store min and max, and calculate silhouette score
df_norm, df_min, df_max = ct.scaler(renewable4)
print("n score")
for ncluster in range(2, 10):
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_norm)
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    print(ncluster, skmet.silhouette_score(renewable4, labels))

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
plt.scatter(renewable4["1990"], renewable4["2019"], 10, labels, marker="o",
            cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.title("K-means clustering of renewable energy consumption")
plt.xlabel("Renewable energy consumption (1990)")
plt.ylabel("Renewable energy consumption (2019)")
plt.legend()
plt.savefig("cluster_renewable.png", dpi=300, bbox_inches="tight")
plt.show()

# Perform fitting on a country in green cluster (with index 1268)
renewable_bur = renewable.loc[(renewable["Country Name"] == "Burundi")]
tr_bur = transposed_df(renewable_bur)

plt.figure()
plt.plot(tr_bur["Year"], tr_bur["Burundi"])
plt.xlim(1990, 2019)
plt.xlabel("Year")
plt.ylabel(
    "Renewable energy consumption (% of total)")
plt.title("Renewable energy consumption of Burundi over time")
plt.savefig("renewable_burundi.png")
plt.show()

# Fit polynomial
param, covar = opt.curve_fit(poly, tr_bur["Year"], tr_bur["Burundi"])

sigma = np.sqrt(np.diag(covar))
print(sigma)
year = np.arange(1990, 2030)
forecast = poly(year, *param)
low, up = err.err_ranges(year, poly, param, sigma)

tr_bur["fit"] = poly(tr_bur["Year"], *param)

plt.figure()
plt.plot(tr_bur["Year"], tr_bur["Burundi"], label="renewable")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlim(1990, 2029)
plt.ylim(0, 100)
plt.xlabel("Year")
plt.ylabel("Renewable energy consumption (% of total)")
plt.legend()
plt.title("Forecast of renewable energy consumption Burundi, including CI")
plt.savefig("forecast_burundi.png")
plt.show()

# Perform fitting on a country in pink cluster (with index 3396)
renewable_con = renewable.loc[(renewable["Country Name"] == "Congo, Rep.")]
tr_con = transposed_df(renewable_con)

plt.figure()
plt.plot(tr_con["Year"], tr_con["Congo, Rep."])
plt.xlim(1990, 2019)
plt.xlabel("Year")
plt.ylabel(
    "Renewable energy consumption (% of total)")
plt.title("Renewable energy consumption of Congo over time")
plt.savefig("renewable_congo.png")
plt.show()

# Fit polynomial
param, covar = opt.curve_fit(poly, tr_con["Year"], tr_con["Congo, Rep."])

sigma = np.sqrt(np.diag(covar))
print(sigma)
year = np.arange(1990, 2030)
forecast = poly(year, *param)
low, up = err.err_ranges(year, poly, param, sigma)

tr_con["fit"] = poly(tr_con["Year"], *param)

plt.figure()
plt.plot(tr_con["Year"], tr_con["Congo, Rep."], label="renewable")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("Year")
plt.ylabel("Renewable energy consumption (% of total)")
plt.xlim(1990, 2029)
plt.ylim(0, 100)
plt.legend()
plt.title("Forecast of renewable energy consumption Congo, including CI")
plt.savefig("forecast_congo.png")
plt.show()