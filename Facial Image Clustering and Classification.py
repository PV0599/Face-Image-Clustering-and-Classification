import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Supervised learning ---> Internal maths is difficult but interpretation
# is easy

# Unsupervised learning ---> Internal maths is easy but interpretation is
# very difficult

# In most real world DS examples, the output of USL becomes the input
# of SL

# Walmart (Data Scientist) --> Create an App that a user could use to 
# take a picture of any store item and display its price, properties, 
# date of manufacture and expiry etc

# 10 family members, you want to create a classifcation and recognition
# model to identify each family member uniquely. 
# You already have a lot of pictures as your data source, BUT in order
# to apply classification logic, the pictures need to be labeled.
# Challenge ---> In DS, we know, the larger the volume of data, the
# better the recognition model would be.
# For a large volume of data, we can't assign labels manually.
# Clustering algo ---> Data dump ---> {F, F, F}, {M, M} etc
# Inspect cluster --> Assign label in one go
# Run the classification model

# Project: Facial Image Clustering and Classification
# Data: Human faces (Olivetti Faces Dataset) AT & T Lab in US

# Loading the dataset

from sklearn.datasets import fetch_olivetti_faces
olivetti = fetch_olivetti_faces()
images = olivetti.images

#EDA
plt.imshow(images[0],cmap='gray')
plt.axis('off')
plt.show

for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i],cmap='gray')
plt.tight_layout()
plt.show()

#Insight : We have the passport size (64 * 64) images of 40 different
# people as our input data


data = olivetti.data
target = olivetti.target
from sklearn.model_selection import StratifiedShuffleSplit
strat_split = StratifiedShuffleSplit(n_splits=1,test_size=0.2)
train_valid_idx, test_idx = next(strat_split.split(data,target))

X_train_valid = data[train_valid_idx]
y_train_valid = target[train_valid_idx]
X_test = data[test_idx]
y_test = target[test_idx]

strat_split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2)
train_idx, valid_idx = next(strat_split.split(X_train_valid, y_train_valid))

X_train = X_train_valid[train_idx]
y_train = y_train_valid[train_idx]
X_valid = X_train_valid[valid_idx]
y_valid = y_train_valid[valid_idx]

X_train.shape

# Applying dimensionality reduction

from sklearn.decomposition import PCA
pca = PCA(n_components = 0.99)
X_train_pca = pca.fit_transform(X_train)
pca.n_components_

# Insight: We could preserve 99% of variance in the data by just
# taking 4% of the features

X_test_pca = pca.transform(X_test)

# Applying clustering techniques to group similar observations
# together

from sklearn.cluster import KMeans

k_range = range(5, 150, 5)
kmeans_per_k = []

for k in k_range:
    print("k = {}".format(k))
    kmeans = KMeans(n_clusters = k).fit(X_train_pca)
    kmeans_per_k.append(kmeans)

# Computing optimal 'k' value by silhouette analysis
    
from sklearn.metrics import silhouette_score
silhouette_scores = [silhouette_score(X_train_pca, model.labels_) for model in kmeans_per_k]

# Average: 0.09
# Your Silhouette Score: 0.23 (highest) (no oultiers)
# Interpretation: Not the optimal case of clustering

# Closer to +1 ---> Optimal case of clustering
# Closer to 0 ----> Clusters are overlapping || No clear separation among clusters
# Closer to -1 ----> Incorrect cluster assignment

best_index = np.argmax(silhouette_scores)
best_k = k_range[best_index]
best_score = silhouette_scores[best_index]

# High overlapping amonog observations encountered

plt.plot(k_range, silhouette_scores)
plt.xlabel('K (No of clusters)')
plt.ylabel('Silhouette Score')
plt.title('Clustering EDA')
plt.show()

# Elbow plot analysis

inertias = [model.inertia_ for model in kmeans_per_k]
best_inertia = inertias[best_index]

plt.plot(k_range, inertias)
plt.xlabel('K (No of clusters)')
plt.ylabel('WCV (Within Cluster Variation)')
plt.title('Clustering EDA')
plt.show()

# best_k = 125
# We would be applying the clusterin algorithm onto the dataset
# with an optimal cluster value of 125

best_model = kmeans_per_k[best_index]
best_model

best_model.fit(X_train_pca)
y_labels = best_model.labels_

# Interpretation

cluster_frequencies = pd.Series(y_labels).value_counts()
cluster_4 = X_train[y_labels == 4]

for i in range(5):
    plt.subplot(3, 2, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cluster_4[i].reshape(64, 64), cmap = 'gray')
plt.show()


cluster_31 = X_train[y_labels == 31]

for i in range(6):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cluster_31[i].reshape(64, 64), cmap = 'gray')
plt.show()

# Interpretation via Automation
# Logic :-> Create a function to plot facial images
# Logic :-> Call the aboe function for all the unique cluster values

def plot_facial_images(faces, labels, n_cols = 5):
    faces = faces.reshape(-1, 64, 64)
    n_rows = (len(faces) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows * 1.5))
    for index, (face, label) in enumerate(zip(faces, labels)):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(face, cmap = 'gray')
        plt.axis('off')
        plt.title(label)
    plt.show()

for cluster_id in np.unique(best_model.labels_):
    print("Cluster", cluster_id)
    in_cluster = best_model.labels_ == cluster_id
    faces = X_train[in_cluster]
    labels = y_train[in_cluster]
    plot_facial_images(faces, labels)


# Scenario: Out of 4096 features, we have used only 185 features for
# clustering the images together. 185 is 4% of the entire data but
# still preserves 99% of the variance.
    
# Reconstruct facial images back from PCA and apply EDA to check
# Whether the faces are still distinguishable
    
# Output: 25 images from the actual data
# 25 reconstructed images
    
X_reconstructed = pca.inverse_transform(X_train_pca)

for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_train[i].reshape(64, 64), cmap = 'gray')
plt.show()

for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_reconstructed[i].reshape(64, 64), cmap = 'gray')
plt.show()























