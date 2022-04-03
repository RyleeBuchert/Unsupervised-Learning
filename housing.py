import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


if __name__ == "__main__":

    plt.style.use("seaborn-whitegrid")
    plt.rc("figure", autolayout=True)
    plt.rc(
        "axes",
        labelweight="bold",
        labelsize="large",
        titleweight="bold",
        titlesize=14,
        titlepad=10,
    )

    housing_data = pd.read_csv('Data\\California_Housing\\housing.csv')
    housing_data = housing_data.loc[:, ["median_income", "latitude", "longitude"]]

    k_means = KMeans(10)
    housing_data['Cluster'] = k_means.fit_predict(housing_data)
    housing_data['Cluster'] = housing_data['Cluster'].astype("category")
    print(housing_data.head())

    sns.relplot(x="longitude", y="latitude", hue="Cluster", data=housing_data, height=6)
    plt.show()



    plt.style.use("seaborn-whitegrid")
    plt.rc("figure", autolayout=True)
    plt.rc(
        "axes",
        labelweight="bold",
        labelsize="large",
        titleweight="bold",
        titlesize=14,
        titlepad=10,
    )

    housing_data = pd.read_csv('Data\\California_Housing\\housing.csv')
    housing_data = housing_data.loc[:, ["median_income", "latitude", "longitude"]]

    gmm = GaussianMixture(10)
    housing_data['Cluster'] = gmm.fit_predict(housing_data)
    housing_data['Cluster'] = housing_data['Cluster'].astype("category")
    print(housing_data.head())

    sns.relplot(x="longitude", y="latitude", hue="Cluster", data=housing_data, height=6)
    plt.show()