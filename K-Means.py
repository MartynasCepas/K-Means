import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn import metrics
from scipy.spatial.distance import cdist
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import StandardScaler, normalize
import seaborn as sns
import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # create dataset
    data = pd.read_csv('./data/data.csv')

    data = data.drop(columns=['default.payment.next.month'])

    data.fillna(method='ffill', inplace=True)

    # Duomenu standartizavimas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # Duomenu normalizavimas
    X_normalized = normalize(X_scaled)

    # Konvertuoja numpy masyva i pandas dataframe
    X_normalized = pd.DataFrame(X_normalized)

    # PCA implementavimas
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(X_normalized)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']

    # Visualizing the data
    plt.plot()
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title('Dataset')
    plt.scatter(X_principal['P1'], X_principal['P2'])
    plt.show()

    # Dendogramos atvaizdavimas
    plt.figure(figsize=(8, 8))
    plt.title('Visualising the data')
    Dendrogram = shc.dendrogram((shc.linkage(X_principal, method='ward')))

    kmeans1 = KMeans(n_clusters=2).fit(X_principal)
    centroids = kmeans1.cluster_centers_
    print(centroids)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_principal['P1'], X_principal['P2'],
                c=kmeans1.fit_predict(X_principal),
                cmap='rainbow')
    plt.show()

    kmeans2 = KMeans(n_clusters=3).fit(X_principal)
    centroids = kmeans1.cluster_centers_
    print(centroids)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_principal['P1'], X_principal['P2'],
                c=kmeans2.fit_predict(X_principal),
                cmap='rainbow')
    plt.show()

    kmeans3 = KMeans(n_clusters=4).fit(X_principal)
    centroids = kmeans1.cluster_centers_
    print(centroids)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_principal['P1'], X_principal['P2'],
                c=kmeans3.fit_predict(X_principal),
                cmap='rainbow')
    plt.show()

    kmeans4 = KMeans(n_clusters=5).fit(X_principal)
    centroids = kmeans1.cluster_centers_
    print(centroids)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_principal['P1'], X_principal['P2'],
                c=kmeans4.fit_predict(X_principal),
                cmap='rainbow')
    plt.show()

    kmeans5 = KMeans(n_clusters=6).fit(X_principal)
    centroids = kmeans1.cluster_centers_
    print(centroids)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_principal['P1'], X_principal['P2'],
                c=kmeans5.fit_predict(X_principal),
                cmap='rainbow')
    plt.show()

    # Sudedame silueto koeficientu reiksmes i sarasa
    silhouette_scores = []
    silhouette_scores.append(
        silhouette_score(X_principal, kmeans1.fit_predict(X_principal)))
    silhouette_scores.append(
        silhouette_score(X_principal, kmeans2.fit_predict(X_principal)))
    silhouette_scores.append(
        silhouette_score(X_principal, kmeans3.fit_predict(X_principal)))
    silhouette_scores.append(
        silhouette_score(X_principal, kmeans4.fit_predict(X_principal)))
    silhouette_scores.append(
        silhouette_score(X_principal, kmeans5.fit_predict(X_principal)))

    print("Silhouette scores")
    print(silhouette_scores)

    # Atvaizduojame silueto koeficientu reiksmiu stulpeline diagrama
    k = [2, 3, 4, 5, 6]

    plt.bar(k, silhouette_scores)
    plt.xlabel('Number of clusters', fontsize=10)
    plt.ylabel('S(i)', fontsize=10)
    plt.show()

    # Skaiciuojame Inertia

    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}

    distortions.append(sum(np.min(cdist(X_principal, kmeans1.cluster_centers_,
                                        'euclidean'), axis=1)) / X_principal.shape[0])
    inertias.append(kmeans1.inertia_)
    inertias.append(kmeans2.inertia_)
    inertias.append(kmeans3.inertia_)
    inertias.append(kmeans4.inertia_)
    inertias.append(kmeans5.inertia_)

    plt.plot(k, inertias, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.show()

    # Visual

    plt.scatter(X_principal['P1'], X_principal['P2'], c=kmeans1.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    plt.show()