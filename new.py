import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

X = pd.read_csv('CreditCard.csv')
X = X.select_dtypes(include=np.number)

# Uzpildo trukstamas reiksmes
X.fillna(method='ffill', inplace=True)

# Duomenu standartizavimas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Duomenu normalizavimas
X_normalized = normalize(X_scaled)

# Konvertuoja numpy masyva i pandas dataframe
X_normalized = pd.DataFrame(X_normalized)

# PCA implementavimas
pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']

# Dendogramos atvaizdavimas
plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward')))

ac2 = AgglomerativeClustering(n_clusters=2)

# Atvaizduojame klasterizavima
plt.figure(figsize=(6, 6))
plt.scatter(X_principal['P1'], X_principal['P2'],
            c=ac2.fit_predict(X_principal),
            cmap='rainbow')
plt.show()

ac3 = AgglomerativeClustering(n_clusters=3)

plt.figure(figsize=(6, 6))
plt.scatter(X_principal['P1'], X_principal['P2'],
            c=ac3.fit_predict(X_principal), cmap='rainbow')
plt.show()

ac4 = AgglomerativeClustering(n_clusters=4)

plt.figure(figsize=(6, 6))
plt.scatter(X_principal['P1'], X_principal['P2'],
            c=ac4.fit_predict(X_principal), cmap='rainbow')
plt.show()

ac5 = AgglomerativeClustering(n_clusters=5)

plt.figure(figsize=(6, 6))
plt.scatter(X_principal['P1'], X_principal['P2'],
            c=ac5.fit_predict(X_principal), cmap='rainbow')
plt.show()

ac6 = AgglomerativeClustering(n_clusters=6)

plt.figure(figsize=(6, 6))
plt.scatter(X_principal['P1'], X_principal['P2'],
            c=ac6.fit_predict(X_principal), cmap='rainbow')
plt.show()

k = [2, 3, 4, 5, 6]

# Sudedame silueto koeficientu reiksmes i sarasa
silhouette_scores = []
silhouette_scores.append(
    silhouette_score(X_principal, ac2.fit_predict(X_principal)))
silhouette_scores.append(
    silhouette_score(X_principal, ac3.fit_predict(X_principal)))
silhouette_scores.append(
    silhouette_score(X_principal, ac4.fit_predict(X_principal)))
silhouette_scores.append(
    silhouette_score(X_principal, ac5.fit_predict(X_principal)))
silhouette_scores.append(
    silhouette_score(X_principal, ac6.fit_predict(X_principal)))

# Atvaizduojame silueto koeficientu reiksmiu stulpeline diagrama
plt.bar(k, silhouette_scores)
plt.xlabel('Number of clusters', fontsize=10)
plt.ylabel('S(i)', fontsize=10)
plt.show()

print("Silhouette scores")
print(silhouette_scores)
