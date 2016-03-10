import numpy as np
import scipy as sp
import scipy.linalg

class PCA:
    """Principal component analysis: argument: X -- a numpy ndarray of shape (n, p)"""

    def __init__(self, X):
        self.data = X
        self.rank = min(self.data.shape)
        self.u, self.s, self.v = np.linalg.svd(X, full_matrices = False)

    def get_components(self, q):
        """return the first q eigenvectors, if q > min(n, p), then return the first min(n, p) eigenvectors"""
        q = min(q, self.rank)
        return self.v.T[:, :q]

    def get_transformed_data(self, q):
        """return the dimension reduced data (reduced using first q eigenvectors)"""
        q = min(q, self.rank)
        return np.dot(np.dot(self.v.T[:, :q], self.v[:q, :]), self.data.T).T

    def get_projection(self, q):
        """return the projection of the data set onto the first q eigenvectors"""
        q = min(q, self.rank)
        return np.dot(self.v[:q, :], self.data.T).T

    def get_singular_values(self):
        """return the singular values of the original data matrix"""
        return self.s

    def get_eigenvalues(self):
        """return the eigenvalues of the scaled covariance matrix X^T X"""
        return self.s * self.s

    def get_transformed_new_data(self, q, new_data):
        """return the dimension reduced new data (reduced using first q eigenvectors)"""
        q = min(q, self.rank)
        return np.dot(np.dot(self.v.T[:, :q], self.v[:q, :]), new_data.T).T





class BinaryGaussianNB:
    """Naive Bayes classifier for a binary class response variable"""

    def __init__(self, X, Y):
        """Constructor. input: X -- data matrix, a (N by p) 2d array, Y -- response vector, a N-vector """
        self.data = X
        self.response_vector = Y
        self.labels = np.unique(Y)
        self.N = X.shape[0]
        self.N_1 = Y[ Y == self.labels[1] ].size
        # if more than 2 labels are presented in Y, throw an error
        if self.labels.size != 2:
            raise ValueError("more than two classes in BinaryGaussianNB")

        # estimated parameters
        pi = float(self.N_1) / self.N
        #print pi
        mu_0 = np.mean(X[ Y == self.labels[0], :], axis = 0, dtype = np.float64)
        #print mu_0
        mu_1 = np.mean(X[ Y == self.labels[1], :], axis = 0, dtype = np.float64)
        #print mu_1

        X0 = X[ Y == self.labels[0] ] - mu_0
        #print X0
        X1 = X[ Y == self.labels[1] ] - mu_1
        #print X1
        sigma_sq = (np.sum(X0 * X0, axis = 0, dtype = np.float64)
        + np.sum(X1 * X1, axis = 0, dtype = np.float64)) / (self.N - 2)
        #print sigma_sq

        # parameters used in the a posterior probability function
        self.w_0 = np.log( (1 - pi)/pi )
        + sum( (mu_1 * mu_1 - mu_0 * mu_0) / ( 2 * sigma_sq) )
        self.w = ( mu_0 - mu_1 )/sigma_sq

    def predict_prob(self, new_data, label = None):
        """give Naive Bayes a posterior probability for points in new_data to have the specified label
           arguments: new_data -- a (M by p) new data matrix
                      label -- specify a label, if no label is specified, probability for one of the label is output"""
        if label == None:
            label = self.labels[1]
        if len(new_data.shape) == 2:
            new_data = new_data.T
        prob_1 = float(1)/( 1 + np.exp(self.w_0 + np.dot(self.w, new_data)) )
        if label == self.labels[1]:
            return prob_1
        elif label == self.labels[0]:
            return 1 - prob_1
        else:
            raise ValueError("incorrect input class label in BinaryGaussianNB.predict_prob")

    def predict_label(self, new_data, specified_label = None, threshold = 0.5):
        """give Naive Bayes classification result, using the threshold"""
        if specified_label == None:
            specified_label = self.labels[1]

        if specified_label == self.labels[0]:
            threshold = 1 - threshold
        elif specified_label == self.labels[1]:
            pass
        else:
            raise ValueError("incorrect label name in BinaryGaussianNB.predict_label")

        number = 1
        if len(new_data.shape) == 2:
            number = new_data.shape[0]
        if number == 1:
            return self.labels[0] if self.predict_prob(new_data) < threshold else self.labels[1]

        new_label = np.repeat(self.labels[1], number)
        new_label[self.predict_prob(new_data) < threshold] = self.labels[0]

        return new_label


class BinaryLDA:
    """Linear Discriminant Analysis classifier for a binary class response variable"""

    def __init__(self, X, Y):
        """Constructor. arguments: X -- data matrix, a (N by p) 2d array, Y -- response vector, a N-vector """
        self.data = X
        self.response_vector = Y
        self.labels = np.unique(Y)
        self.N = X.shape[0]
        self.N_1 = Y[ Y == self.labels[1] ].size
        # if more than 2 labels are presented in Y, throw an error
        if self.labels.size != 2:
            raise ValueError("more than two classes in BinaryGaussianNB")

        # estimated parameters
        self.pi_1 = float(self.N_1) / self.N
        self.pi_0 = 1 - self.pi_1
        self.mu_0 = np.mean(X[ Y == self.labels[0], :], axis = 0, dtype = np.float64)
        self.mu_1 = np.mean(X[ Y == self.labels[1], :], axis = 0, dtype = np.float64)

        X0 = X[ Y == self.labels[0] ] - self.mu_0
        X1 = X[ Y == self.labels[1] ] - self.mu_1
        #sigma_sq = (np.sum(X0 * X0, axis = 0, dtype = np.float64)
        #+ np.sum(X1 * X1, axis = 0, dtype = np.float64)) / (self.N - 2)
        self.cov_matrix = (np.dot(X0.T, X0) + np.dot(X1.T, X1))/(self.N - 2)

        sw = self.cov_matrix * (self.N - 2)
        difference = self.mu_0 - self.mu_1
        sb = np.dot( difference.reshape((self.mu_0.size, 1)), difference.reshape( (1, self.mu_0.size)) )

        self.w, self.v = scipy.linalg.eigh(sb, sw)
        self.w = np.flipud(self.w)
        self.v = np.fliplr(self.v)

        #self.w, self.v = np.linalg.eig(np.dot(np.linalg.inv(sw), sb))

    def _log_ratio(self, new_data):
        """returns the log_ratio function. Argument: new_data -- a (M by p) new data matrix"""
        if len(new_data.shape) == 2 and new_data.shape[1] != self.mu_0.size:
            new_data = new_data.T

        inv_cov_mul_mu_diff = np.dot( np.linalg.inv(self.cov_matrix), (self.mu_1 - self.mu_0))

        ratio = np.dot(new_data, inv_cov_mul_mu_diff) + np.log(self.pi_1 /self.pi_0)
        - 0.5 * np.dot((self.mu_1 + self.mu_0), inv_cov_mul_mu_diff)

        return ratio

    def predict_label(self, new_data):
        """returns the predicted labels. Argument: new_data -- a (M by p) new data matrix"""

        number = new_data.size / self.mu_0.size

        if number == 1:
            return self.labels[1] if self._log_ratio(new_data) >= 0 else self.labels[0]

        new_label = np.repeat(self.labels[1], number)
        new_label[self._log_ratio(new_data) < 0] = self.labels[0]

        return new_label

    def get_components(self, q):
        """return the first q eigenvectors, if q > min(n, p), then return the first min(n, p) eigenvectors"""
        return self.v[:, :q]

    def get_eigenvalues(self):
        return self.w

    def get_transformed_data(self, q, new_data):
        return np.dot(self.v[:, :q] , np.dot(self.v[:, :q].T, new_data.T)).T
