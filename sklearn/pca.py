import numpy as np
from sklearn.decomposition import PCA

data = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

def pca(X, red_dim):
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:,i]) for i in range(n_features)])
    norm_X = X-mean
    scatter_X = np.dot(np.transpose(norm_X), norm_X)
    eig_val, eig_vec = np.linalg.eig(scatter_X)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    eig_pairs.sort(key= lambda x:x[0], reverse=True)
    feature = np.array([ele[1] for ele in eig_pairs[:red_dim]])
    return np.dot(norm_X, np.transpose(feature))

sk_pca = PCA(n_components=1).fit(data)
print(sk_pca.transform(data))
print(pca(data, 1))