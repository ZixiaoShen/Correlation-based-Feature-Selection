import scipy.io


mat = scipy.io.loadmat('../Datasets/colon.mat')
X = mat['X']
X = X.astype(float)
y = mat['Y']
y = y[:, 0]
n_samples, n_features = X.shape  # number of samples and number of features



