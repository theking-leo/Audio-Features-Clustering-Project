# Audio-Features-Clustering-Project.
Overview

This project involves the clustering and analysis of audio features in a dataset of songs. The goal is to group similar songs based on their audio characteristics using KMeans clustering. The project is implemented in Python using libraries such as pandas, scikit-learn, and visualization tools like matplotlib and seaborn.
Steps:
1. Data Loading




# Load the dataset from a CSV file
data = pd.read_csv("df_audio_features_1000")

This step uses the pandas library to load the dataset containing audio features of songs.
2. Data Exploration


data.head()

Prints the first few rows of the dataset to get an overview of the data.
3. Feature Selection


columns_selected = ['danceability', 'energy', 'key', 'loudness', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
work_columns = data[columns_selected]

Selects specific features from the dataset for clustering.
4. Feature Scaling


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(work_columns)
scaled_df = pd.DataFrame(scaled_features, columns=columns_selected, index=work_columns.index)

Scales the selected features to ensure that they are on a similar scale.
5. KMeans Clustering


from sklearn.cluster import KMeans

my_kk = KMeans(n_clusters=5)
clusters = my_kk.predict(scaled_df)

Applies KMeans clustering to group songs into clusters based on their scaled features.
6. Principal Component Analysis (PCA)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_df)

Reduces the dimensionality of the dataset using Principal Component Analysis.
7. Visualization - Scatter Plot


import matplotlib.pyplot as plt
import seaborn as sns

# Create a scatter plot using Seaborn and Matplotlib
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2',
                hue='Cluster Label', data=pca_df, palette='Set1', s=100)
plt.title('KMeans Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

Visualizes the clustered data in a 2D scatter plot.
8. 3D Visualization (Optional)



# 3D Plot using Matplotlib
# ...

# Spider Plot using Seaborn
# ...

Optional 3D visualization and spider plot for a more in-depth analysis.
9. Determining the Optimal Number of Clusters


inertia_values = []
for num_clusters in range(1, 35):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(scaled_df)
    inertia_values.append(kmeans.inertia_)

Determines the optimal number of clusters using the elbow method.
10. Silhouette Score



from sklearn.metrics import silhouette_score

# Fit KMeans with n_clusters=20
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(scaled_df)

# Calculate the Silhouette Score
silhouette_avg = silhouette_score(scaled_df, kmeans.labels_)
print("The Silhouette Score is:", silhouette_avg)

Calculates the silhouette score as an additional metric for cluster quality.
11. Davies-Bouldin Index



from sklearn.metrics import davies_bouldin_score

# Calculate Davies-Bouldin Index for different k values
max_k = 20
davies_bouldin_scores = []
for k in range(2, max_k):
    labels = KMeans(n_clusters=k).fit(scaled_df).labels_
    davies_bouldin_scores.append(davies_bouldin_score(scaled_df, labels))

Calculates the Davies-Bouldin Index to evaluate cluster separation.
12. Uploading New Data (Optional)


# Upload new data for clustering
uploaded = files.upload()
filename = list(uploaded.keys())[0]
new_data = pd.read_csv(io.StringIO(uploaded[filename].decode('utf-8')))

Allows the user to upload new data for clustering.
13. Clustering New Data (Optional)


# Perform clustering on the new data
# ...

Performs clustering on the newly uploaded data using the pre-trained model.
14. Creating Playlists


# Create playlists for each cluster
# ...

Groups songs into playlists based on their cluster labels.
Conclusion

This README provides an overview of the steps involved in the audio features clustering project. It covers data loading, exploration, feature selection, scaling, clustering, visualization, and evaluation. The optional sections demonstrate how to handle new data and create playlists based on the clusters.

Feel free to adjust the code and parameters based on your specific requirements and dataset. If you have any questions or issues, feel free to reach out!
