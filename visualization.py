import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D  


embedding_file_path = r'C:\Users\HUAWEI\OneDrive\Desktop\iGEM\embeddings.txt'
embedding_data = np.loadtxt(embedding_file_path)  


dimensions = int(input("input dimensions(2/3)："))


tsne = TSNE(n_components=dimensions, random_state=42)
reduced_data = tsne.fit_transform(embedding_data)


from sklearn.metrics.pairwise import euclidean_distances


distances = euclidean_distances(reduced_data)
min_distances = np.min(distances + np.eye(distances.shape[0]) * np.max(distances), axis=1)
scaler = MinMaxScaler()
min_distances_scaled = scaler.fit_transform(min_distances.reshape(-1, 1)).flatten()


plt.figure(figsize=(12, 10))  

if dimensions == 2:
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=min_distances_scaled, cmap='viridis', marker='o')
    plt.title("t-SNE visualization of embeddings (2D)")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
elif dimensions == 3:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=min_distances_scaled, cmap='viridis', marker='o')
    ax.set_title("t-SNE visualization of embeddings (3D)")
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    ax.set_zlabel("t-SNE component 3")



plt.colorbar(scatter, label='Distance to Nearest Neighbor') 
plt.show()
