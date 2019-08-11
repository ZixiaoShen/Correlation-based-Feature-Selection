import scipy.io
from CFSmethod import CFS

mat = scipy.io.loadmat('../Datasets/madelon.mat')
X = mat['X']
X = X.astype(float)
y = mat['Y']
y = y[:, 0]
n_samples, n_features = X.shape  # number of samples and number of features

idx = CFS.cfs(X, y)
print(idx)
