# Author: Xin Wang
# Email: xinwangmath@gmail.com
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mllib as ml
#%matplotlib inline

np.random.seed(1)
# 1. load in data
wine_raw = pd.read_csv('wine.csv', header = None)
mnist_raw_train = pd.read_csv('train.csv', header = None).transpose()
mnist_raw_test = pd.read_csv('test.csv', header = None).transpose()

# 2. data preprocessing
# 2.1. for the wine data set
def sample_func(index, sample_size):
    return np.random.choice(index.size, sample_size)


label_1_index = np.where(wine_raw[0] == 1)[0]
label_2_index = np.where(wine_raw[0] == 2)[0]

# shuffle and get train and test set: train set size is 5
np.random.shuffle(label_1_index)
np.random.shuffle(label_2_index)

wine_5_train_index_c1 = label_1_index[:5]
wine_5_test_index_c1 = label_1_index[5:]
wine_5_train_index_c2 = label_2_index[:5]
wine_5_test_index_c2 = label_2_index[5:]

wine_5_train = pd.concat([wine_raw.ix[wine_5_train_index_c1], wine_raw.ix[wine_5_train_index_c2]])
wine_5_test = pd.concat([wine_raw.ix[wine_5_test_index_c1], wine_raw.ix[wine_5_test_index_c2]])

# shuffle again and get train and test set: train set size is 50
np.random.shuffle(label_1_index)
np.random.shuffle(label_2_index)

wine_50_train_index_c1 = label_1_index[:50]
wine_50_test_index_c1 = label_1_index[50:]
wine_50_train_index_c2 = label_2_index[:50]
wine_50_test_index_c2 = label_2_index[50:]

wine_50_train = pd.concat([wine_raw.ix[wine_50_train_index_c1], wine_raw.ix[wine_50_train_index_c2]])
wine_50_test = pd.concat([wine_raw.ix[wine_50_test_index_c1], wine_raw.ix[wine_50_test_index_c2]])

wine_5_train_X = wine_5_train.as_matrix(columns = range(1,14))
wine_5_train_Y = wine_5_train[0].as_matrix()

wine_5_test_X = wine_5_test.as_matrix(columns = range(1,14))
wine_5_test_Y = wine_5_test[0].as_matrix()

wine_50_train_X = wine_50_train.as_matrix(columns = range(1,14))
wine_50_train_Y = wine_50_train[0].as_matrix()

wine_50_test_X = wine_50_test.as_matrix(columns = range(1,14))
wine_50_test_Y = wine_50_test[0].as_matrix()

# construct a shuffled list of index for the purpose of 10-fold cv
label_12_index = np.concatenate((label_1_index, label_2_index))
np.random.shuffle(label_12_index)



#2.2. For the MNIST data set
#mnist_raw_train = pd.read_csv('data/train.csv', header = None).transpose()
#mnist_raw_test = pd.read_csv('data/test.csv', header = None).transpose()

# convert the class label type to int
mnist_raw_train[784] = mnist_raw_train[784].astype(int)
mnist_raw_test[784] = mnist_raw_test[784].astype(int)

def mnist_extract_specified_labels(dataset, label_1, label_2):
    mnist_label_1_index = np.where(dataset[784] == label_1)[0]
    mnist_label_2_index = np.where(dataset[784] == label_2)[0]
    mnist_label_12_index = np.concatenate((mnist_label_1_index, mnist_label_2_index))
    return dataset.ix[mnist_label_12_index]

mnist_01_train = mnist_extract_specified_labels(mnist_raw_train, 0, 1)
mnist_01_test = mnist_extract_specified_labels(mnist_raw_test, 0, 1)

mnist_35_train = mnist_extract_specified_labels(mnist_raw_train, 3, 5)
mnist_35_test = mnist_extract_specified_labels(mnist_raw_test, 3, 5)

mnist_01_train_X = mnist_01_train.as_matrix(columns = range(784))
mnist_01_train_Y = mnist_01_train[784].as_matrix()

mnist_01_test_X = mnist_01_test.as_matrix(columns = range(784))
mnist_01_test_Y = mnist_01_test[784].as_matrix()

mnist_35_train_X = mnist_35_train.as_matrix(columns = range(784))
mnist_35_train_Y = mnist_35_train[784].as_matrix()

mnist_35_test_X = mnist_35_test.as_matrix(columns = range(784))
mnist_35_test_Y = mnist_35_test[784].as_matrix()

# validation set pending



# 3. Data analysis
# 3.1. PCA for the wine data set
# 3.1.1. with training set size = 5
wine_5_pca = ml.PCA(wine_5_train_X)
wine_5_pca_eigenvalues = wine_5_pca.get_eigenvalues()

plt.plot(np.cumsum(wine_5_pca_eigenvalues)/np.sum(wine_5_pca_eigenvalues), 'bo-', linewidth = 2)
plt.ylim([0.1, 1.1])
plt.xlabel('index of eigenvalues( starting from 0)', fontsize = 16)
plt.ylabel('cummulative sum of eigenvalues', fontsize = 16)
plt.title('training set size = 5 * 2', fontsize = 16)
plt.show()

plt.semilogy(wine_5_pca_eigenvalues/np.sum(wine_5_pca_eigenvalues), 'bo-', linewidth = 2)
plt.xlabel('index of eigenvalues( starting from 0)', fontsize = 16)
plt.ylabel('eigenvalues/sum of eigenvalues', fontsize = 16)
plt.title('training set size = 5 * 2', fontsize = 16)
plt.show()


wine_50_pca = ml.PCA(wine_50_train_X)
wine_50_pca_eigenvalues = wine_50_pca.get_eigenvalues()

plt.plot(np.cumsum(wine_50_pca_eigenvalues)/np.sum(wine_50_pca_eigenvalues), 'bo-', linewidth = 2)
plt.ylim([0.1, 1.1])
plt.xlabel('index of eigenvalues( starting from 0)', fontsize = 16)
plt.ylabel('cummulative sum of eigenvalues', fontsize = 16)
plt.title('training set size = 50 * 2', fontsize = 16)
plt.show()

plt.semilogy(wine_50_pca_eigenvalues/np.sum(wine_50_pca_eigenvalues), 'bo-', linewidth = 2)
plt.xlabel('index of eigenvalues( starting from 0)', fontsize = 16)
plt.ylabel('eigenvalues/sum of eigenvalues', fontsize = 16)
plt.title('training set size = 50 * 2', fontsize= 16)
plt.show()


# reconstruction for wine data set
print wine_5_test_X[0]
wine_5_V_7 = np.dot(wine_5_pca.get_components(7), np.dot(wine_5_pca.get_components(7).T, wine_5_test_X[0]))
print wine_5_V_7
print np.linalg.norm(wine_5_V_7 - wine_5_test_X[0] )

print wine_50_test_X[0]
wine_50_V_7 = np.dot(wine_50_pca.get_components(7), np.dot(wine_50_pca.get_components(7).T, wine_50_test_X[0]))
print wine_50_V_7
print np.linalg.norm(wine_50_V_7 - wine_50_test_X[0] )


# 3.2. PCA for the mnist data set
# 3.2.1. cummulative sum of eigenvalues
mnist_01_pca = ml.PCA(mnist_01_train_X)
mnist_01_pca_eigenvalues = mnist_01_pca.get_eigenvalues()

plt.plot(np.cumsum(mnist_01_pca_eigenvalues)/np.sum(mnist_01_pca_eigenvalues), 'bo-', linewidth = 1)
#plt.ylim([0.1, 1.1])
plt.xlabel('index of eigenvalues( starting from 0)', fontsize = 16)
plt.ylabel('cummulative sum of eigenvalues', fontsize = 16)
plt.title('MNIST with label 0 and 1', fontsize = 16)
plt.show()

mnist_35_pca = ml.PCA(mnist_35_train_X)
mnist_35_pca_eigenvalues = mnist_35_pca.get_eigenvalues()

plt.plot(np.cumsum(mnist_35_pca_eigenvalues)/np.sum(mnist_35_pca_eigenvalues), 'bo-', linewidth = 1)
#plt.ylim([0.1, 1.1])
plt.xlabel('index of eigenvalues( starting from 0)', fontsize = 16)
plt.ylabel('cummulative sum of eigenvalues', fontsize = 16)
plt.title('MNIST with label 3 and 5', fontsize = 16)
plt.show()

# most important eigenvetors and 20th eigenvetor
mnist_01_pca_eigenvectors = mnist_01_pca.get_components(20)
eigenv_1_01 = mnist_01_pca_eigenvectors[:, 0]
eigenv_20_01 = mnist_01_pca_eigenvectors[:, 19]
plt.imshow(eigenv_1_01.reshape((28, 28)).T, cmap = 'Greys')
plt.title('the most important eigenvector for MNIST with class label 0 and 1')
plt.show()
plt.imshow(eigenv_20_01.reshape((28, 28)).T, cmap = 'Greys')
plt.title('the 20th eigenvector for MNIST with class label 0 and 1')
plt.show()

mnist_35_pca_eigenvectors = mnist_35_pca.get_components(20)
eigenv_1_35 = mnist_35_pca_eigenvectors[:, 0]
eigenv_20_35 = mnist_35_pca_eigenvectors[:, 19]
plt.imshow(eigenv_1_35.reshape((28, 28)).T, cmap = 'Greys')
plt.title('the most important eigenvector for MNIST with class label 3 and 5')
plt.show()
plt.imshow(eigenv_20_35.reshape((28, 28)).T, cmap = 'Greys')
plt.title('the 20th eigenvector for MNIST with class label 3 and 5')
plt.show()

# reconstruction with the first 50 eigenvectors
mnist_01_V_50 = mnist_01_pca.get_components(50)

mnist_01_test_img = mnist_01_test_X[0].reshape((28, 28)).T
plt.imshow(mnist_01_test_img, cmap = 'Greys')
plt.title('MNIST 0 1 random test point')
plt.show()

mnist_01_reconst_vec = np.dot(mnist_01_V_50, np.dot(mnist_01_V_50.T, mnist_01_test_X[0]))
mnist_01_reconst_img = mnist_01_reconst_vec.reshape((28, 28)).T
plt.imshow(mnist_01_reconst_img, cmap = 'Greys')
plt.title('MNIST 0 1 reconstructed with the first 50 eigenvectors')
plt.show()

mnist_01_error_vec = mnist_01_test_X[0] - mnist_01_reconst_vec
mnist_01_error_img = mnist_01_error_vec.reshape((28, 28)).T
plt.imshow(mnist_01_error_img, cmap = 'Greys')
plt.title('MNIST 0 1 reconstruction error with the first 50 eigenvectors')
plt.show()



mnist_35_V_50 = mnist_35_pca.get_components(50)

mnist_35_test_img = mnist_35_test_X[0].reshape((28, 28)).T
plt.imshow(mnist_35_test_img, cmap = 'Greys')
plt.title('MNIST 3 5 random test point')
plt.show()

mnist_35_reconst_vec = np.dot(mnist_35_V_50, np.dot(mnist_35_V_50.T, mnist_35_test_X[0]))
mnist_35_reconst_img = mnist_35_reconst_vec.reshape((28, 28)).T
plt.imshow(mnist_35_reconst_img, cmap = 'Greys')
plt.title('MNIST 3 5 reconstructed with the first 50 eigenvectors')
plt.show()

mnist_35_error_vec = mnist_35_test_X[0] - mnist_35_reconst_vec
mnist_35_error_img = mnist_35_error_vec.reshape((28, 28)).T
plt.imshow(mnist_35_error_img, cmap = 'Greys')
plt.title('MNIST 3 5 reconstruction error with the first 50 eigenvectors')
plt.show()

# 3.3. LDA for the wine data set

# 3.3.1. size 5 * 2 training set: singularity problem. See the report.
wine_5_train_X_reduced = wine_5_pca.get_projection(7)
wine_5_pca_bases = wine_5_pca.get_components(7)
wine_5_lda = ml.BinaryLDA(wine_5_train_X_reduced, wine_5_train_Y)
wine_5_lda_eigenvalues = wine_5_lda.get_eigenvalues()

plt.plot(np.cumsum(wine_5_lda_eigenvalues)/np.sum(wine_5_lda_eigenvalues), 'bo-', linewidth = 2)
#plt.ylim([0.1, 1.1])
plt.xlabel('index of eigenvalues( starting from 0)', fontsize = 16)
plt.ylabel('cummulative sum of eigenvalues', fontsize = 16)
plt.title('training set size = 5 * 2', fontsize = 16)
plt.show()

print wine_5_lda_eigenvalues


lda_reconstruct_index = np.random.choice(wine_5_test_Y.size, 1)
lda_reconstruct_point = wine_5_test_X[lda_reconstruct_index, ]
print lda_reconstruct_point

lda_reconstruction = wine_5_lda.get_transformed_data(1, np.dot(lda_reconstruct_point, wine_5_pca_bases))
print np.dot(wine_5_pca_bases, lda_reconstruction.T).T

lda_reconstruction = wine_5_lda.get_transformed_data(1, np.dot(lda_reconstruct_point, wine_5_pca_bases))
print np.dot(wine_5_pca_bases, lda_reconstruction.T).T

print (np.linalg.norm( (np.dot(wine_5_pca_bases, lda_reconstruction.T).T - lda_reconstruct_point) )
/np.linalg.norm(lda_reconstruct_point))

# 3.3.2. size 50 * 2 training set.

wine_50_lda = ml.BinaryLDA(wine_50_train_X, wine_50_train_Y)
wine_50_lda_eigenvalues = wine_50_lda.get_eigenvalues()

plt.plot(np.cumsum(wine_50_lda_eigenvalues)/np.sum(wine_50_lda_eigenvalues), 'bo-', linewidth = 2)
#plt.ylim([0.1, 1.1])
plt.xlabel('index of eigenvalues( starting from 0)', fontsize = 16)
plt.ylabel('cummulative sum of eigenvalues', fontsize = 16)
plt.title('training set size = 50 * 2', fontsize = 16)
plt.show()

print wine_5_lda_eigenvalues

# reconstruction:
lda_reconstruct_index = np.random.choice(wine_50_test_Y.size, 1)
lda_reconstruct_point = wine_50_test_X[lda_reconstruct_index, ]
print lda_reconstruct_point

lda_reconstruction = wine_50_lda.get_transformed_data(1, lda_reconstruct_point)

print np.linalg.norm(lda_reconstruction - lda_reconstruct_point )
print np.linalg.norm(lda_reconstruction - lda_reconstruct_point )/np.linalg.norm(lda_reconstruct_point)


# 3.4. LDA for MNIST
# 3.4.1. cummulative sum of eigenvalues

mnist_01_train_X_reduced = mnist_01_pca.get_projection(50)
mnist_01_bases = mnist_01_pca.get_components(50)
mnist_01_lda = ml.BinaryLDA(mnist_01_train_X_reduced, mnist_01_train_Y)
mnist_01_lda_eigenvalues = mnist_01_lda.get_eigenvalues()

plt.plot(np.cumsum(mnist_01_lda_eigenvalues)/np.sum(mnist_01_lda_eigenvalues), 'bo-', linewidth = 1)
#plt.ylim([0.1, 1.1])
plt.xlabel('index of eigenvalues( starting from 0)', fontsize = 16)
plt.ylabel('cummulative sum of eigenvalues', fontsize = 16)
plt.title('MNIST with label 0 and 1', fontsize = 16)
plt.show()

print mnist_01_lda_eigenvalues


mnist_35_train_X_reduced = mnist_35_pca.get_projection(50)
mnist_35_bases = mnist_35_pca.get_components(50)
mnist_35_lda = ml.BinaryLDA(mnist_35_train_X_reduced, mnist_35_train_Y)
mnist_35_lda_eigenvalues = mnist_35_lda.get_eigenvalues()

plt.plot(np.cumsum(mnist_35_lda_eigenvalues)/np.sum(mnist_35_lda_eigenvalues), 'bo-', linewidth = 1)
#plt.ylim([0.1, 1.1])
plt.xlabel('index of eigenvalues( starting from 0)', fontsize = 16)
plt.ylabel('cummulative sum of eigenvalues', fontsize = 16)
plt.title('MNIST with label 3 and 5', fontsize = 16)
plt.show()

print mnist_35_lda_eigenvalues

# 3.4.2. the most important eigenvector and the 20th eigenvetor

mnist_01_lda_eigenvectors = mnist_01_lda.get_components(20)
eigenv_1_01 = np.dot(mnist_01_bases, mnist_01_lda_eigenvectors[:, 0]).T
eigenv_20_01 = np.dot(mnist_01_bases, mnist_01_lda_eigenvectors[:, 19]).T
plt.imshow(eigenv_1_01.reshape((28, 28)).T, cmap = 'Greys')
plt.title('the most important eigenvector for MNIST with class label 0 and 1')
plt.show()
plt.imshow(eigenv_20_01.reshape((28, 28)).T, cmap = 'Greys')
plt.title('the 20th eigenvector for MNIST with class label 0 and 1')
plt.show()

mnist_35_lda_eigenvectors = mnist_35_lda.get_components(20)
eigenv_1_35 = np.dot(mnist_35_bases, mnist_35_lda_eigenvectors[:, 0]).T
eigenv_20_35 = np.dot(mnist_35_bases, mnist_35_lda_eigenvectors[:, 19]).T
plt.imshow(eigenv_1_35.reshape((28, 28)).T, cmap = 'Greys')
plt.title('the most important eigenvector for MNIST with class label 3 and 5')
plt.show()
plt.imshow(eigenv_20_35.reshape((28, 28)).T, cmap = 'Greys')
plt.title('the 20th eigenvector for MNIST with class label 3 and 5')
plt.show()

# 3.4.3. reconstruction

# reconstruction with the first eigenvector

temp = np.random.choice(mnist_01_test_Y.size, 1)
mnist_01_test_img = mnist_01_test_X[temp].reshape((28, 28)).T
plt.imshow(mnist_01_test_img, cmap = 'Greys')
plt.title('MNIST 0 1 random test point')
plt.show()

mnist_01_reconst_vec = np.dot(mnist_01_test_X[temp], eigenv_1_01) * eigenv_1_01
mnist_01_reconst_img = mnist_01_reconst_vec.reshape((28, 28)).T
plt.imshow(mnist_01_reconst_img, cmap = 'Greys')
plt.title('MNIST 0 1 reconstructed with the first eigenvector')
plt.show()

mnist_01_error_vec = mnist_01_test_X[temp] - mnist_01_reconst_vec
mnist_01_error_img = mnist_01_error_vec.reshape((28, 28)).T
plt.imshow(mnist_01_error_img, cmap = 'Greys')
plt.title('MNIST 0 1 reconstruction error with the first eigenvector')
plt.show()



temp = np.random.choice(mnist_35_test_Y.size, 1)
mnist_35_test_img = mnist_35_test_X[temp].reshape((28, 28)).T
plt.imshow(mnist_35_test_img, cmap = 'Greys')
plt.title('MNIST 3 5 random test point')
plt.show()

mnist_35_reconst_vec = np.dot(mnist_35_test_X[temp], eigenv_1_35) * eigenv_1_35
mnist_35_reconst_img = mnist_35_reconst_vec.reshape((28, 28)).T
plt.imshow(mnist_35_reconst_img, cmap = 'Greys')
plt.title('MNIST 3 5 reconstructed with the first eigenvector')
plt.show()

mnist_35_error_vec = mnist_35_test_X[temp] - mnist_35_reconst_vec
mnist_35_error_img = mnist_35_error_vec.reshape((28, 28)).T
plt.imshow(mnist_35_error_img, cmap = 'Greys')
plt.title('MNIST 3 5 reconstruction error with the first eigenvector')
plt.show()

# 3.5. Classification
# 3.5.1. wine_5
wine_5_pca_bases = wine_5_pca.get_components(3)
wine_5_train_X_reduced = np.dot(wine_5_train_X, wine_5_pca_bases)
wine_5_test_X_reduced = np.dot(wine_5_test_X, wine_5_pca_bases)
wine_5_lda = ml.BinaryLDA(wine_5_train_X_reduced, wine_5_train_Y)
wine_5_lda_predicted_Y = wine_5_lda.predict_label(wine_5_test_X_reduced)

wine_5_nb = ml.BinaryGaussianNB(wine_5_train_X_reduced, wine_5_train_Y)
wine_5_nb_predicted_Y = wine_5_nb.predict_label(wine_5_test_X_reduced)
#print wine_5_test_Y
#print wine_5_lda_predicted_Y
#print wine_5_nb_predicted_Y

def confusion(real_Y, pred_Y, c0, c1):
    a_00 = np.sum(pred_Y[real_Y == c0] == c0)
    a_01 = np.sum(pred_Y[real_Y == c0] == c1)
    a_10 = np.sum(pred_Y[real_Y == c1] == c0)
    a_11 = np.sum(pred_Y[real_Y == c1] == c1)

    mat = np.array([[a_00, a_01], [a_10, a_11]])
    rate = float(a_00 + a_11)/real_Y.size

    return mat, rate

wine_5_mat_lda, wine_5_rate_lda = confusion(wine_5_test_Y, wine_5_lda_predicted_Y, 1, 2)
wine_5_mat_nb, wine_5_rate_nb = confusion(wine_5_test_Y, wine_5_nb_predicted_Y, 1, 2)

print wine_5_mat_lda
print wine_5_mat_nb
print wine_5_rate_lda
print wine_5_rate_nb


#wine_50_pca_bases = wine_50_pca.get_components(7)
#wine_50_train_X_reduced = np.dot(wine_50_train_X, wine_50_pca_bases)
#wine_50_test_X_reduced = np.dot(wine_50_test_X, wine_50_pca_bases)
wine_50_lda = ml.BinaryLDA(wine_50_train_X, wine_50_train_Y)
wine_50_lda_predicted_Y = wine_50_lda.predict_label(wine_50_test_X)

wine_50_nb = ml.BinaryGaussianNB(wine_50_train_X, wine_50_train_Y)
wine_50_nb_predicted_Y = wine_50_nb.predict_label(wine_50_test_X)
#print wine_50_test_Y
#print wine_50_lda_predicted_Y
#print wine_50_nb_predicted_Y

wine_50_mat_lda, wine_50_rate_lda = confusion(wine_50_test_Y, wine_50_lda_predicted_Y, 1, 2)
wine_50_mat_nb, wine_50_rate_nb = confusion(wine_50_test_Y, wine_50_nb_predicted_Y, 1, 2)

print wine_50_mat_lda
print wine_50_mat_nb
print wine_50_rate_lda
print wine_50_rate_nb

# wine cv
wine_cv_mat_lda = np.zeros((2, 2))
wine_cv_rate_lda = 0.0
wine_cv_mat_nb = np.zeros((2, 2))
wine_cv_rate_nb = 0.0
one_tenth = label_12_index.size/10
for i in range(10):
    wine_cv_test_index = label_12_index[i * one_tenth: (i+1)*one_tenth]
    mask = np.ones(label_12_index.shape, dtype = bool)
    mask[wine_cv_test_index] = False
    wine_cv_train_X = wine_raw.ix[mask].as_matrix(columns = range(1,14))
    wine_cv_train_Y = wine_raw.ix[mask].as_matrix(columns = [0]).T.flatten()
    wine_cv_test_X = wine_raw.ix[wine_cv_test_index].as_matrix(columns = range(1,14))
    wine_cv_test_Y = wine_raw.ix[wine_cv_test_index].as_matrix(columns = [0]).T.flatten()

    #print wine_cv_train_Y == 1

    wine_cv_lda = ml.BinaryLDA(wine_cv_train_X, wine_cv_train_Y)
    wine_cv_lda_predicted_Y = wine_cv_lda.predict_label(wine_50_test_X)

    wine_cv_nb = ml.BinaryGaussianNB(wine_cv_train_X, wine_cv_train_Y)
    wine_cv_nb_predicted_Y = wine_cv_nb.predict_label(wine_50_test_X)

    wine_cv_mat_lda_n, wine_cv_rate_lda_n = confusion(wine_cv_test_Y, wine_cv_lda_predicted_Y, 1, 2)
    wine_cv_mat_nb_n, wine_cv_rate_nb_n = confusion(wine_cv_test_Y, wine_cv_nb_predicted_Y, 1, 2)

    wine_cv_mat_lda  = wine_cv_mat_lda_n/10.0 + wine_cv_mat_lda
    wine_cv_mat_nb  = wine_cv_mat_nb_n/10.0 + wine_cv_mat_nb

    wine_cv_rate_lda  = wine_cv_rate_lda_n/10.0 + wine_cv_rate_lda
    wine_cv_rate_nb  = wine_cv_rate_nb_n/10.0 + wine_cv_rate_nb

print wine_cv_mat_lda
print wine_cv_mat_nb
print wine_cv_rate_lda
print wine_cv_rate_nb

mnist_01_test_X_reduced = np.dot(mnist_01_test_X, mnist_01_bases)
mnist_01_lda_predicted_Y = mnist_01_lda.predict_label(mnist_01_test_X_reduced)

mnist_01_train_X_reduced = np.dot(mnist_01_train_X, mnist_01_bases)
mnist_01_nb = ml.BinaryGaussianNB(mnist_01_train_X_reduced, mnist_01_train_Y)
mnist_01_nb_predicted_Y = mnist_01_nb.predict_label(mnist_01_test_X_reduced)
#print mnist_01_lda_predicted_Y
#print mnist_01_nb_predicted_Y
#print mnist_01_test_Y

mnist_01_mat_lda, mnist_01_rate_lda = confusion(mnist_01_test_Y, mnist_01_lda_predicted_Y, 0, 1)
mnist_01_mat_nb, mnist_01_rate_nb = confusion(mnist_01_test_Y, mnist_01_nb_predicted_Y, 0, 1)

print mnist_01_mat_lda
print mnist_01_mat_nb
print mnist_01_rate_lda
print mnist_01_rate_nb

mnist_35_test_X_reduced = np.dot(mnist_35_test_X, mnist_35_bases)
mnist_35_lda_predicted_Y = mnist_35_lda.predict_label(mnist_35_test_X_reduced)

mnist_35_train_X_reduced = np.dot(mnist_35_train_X, mnist_35_bases)
mnist_35_nb = ml.BinaryGaussianNB(mnist_35_train_X_reduced, mnist_35_train_Y)
mnist_35_nb_predicted_Y = mnist_35_nb.predict_label(mnist_35_test_X_reduced)
#print mnist_35_lda_predicted_Y
#print mnist_35_nb_predicted_Y
#print mnist_35_test_Y

mnist_35_mat_lda, mnist_35_rate_lda = confusion(mnist_35_test_Y, mnist_35_lda_predicted_Y, 3, 5)
mnist_35_mat_nb, mnist_35_rate_nb = confusion(mnist_35_test_Y, mnist_35_nb_predicted_Y, 3, 5)

print mnist_35_mat_lda
print mnist_35_mat_nb
print mnist_35_rate_lda
print mnist_35_rate_nb
