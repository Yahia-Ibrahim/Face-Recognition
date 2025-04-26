import sklearn
import numpy as np

class GaussianMixture():

    def __init__(self, n_components=2, max_iter=100, seed=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.seed = seed
        self.posteriors, self.means, self.covs = None, None, None

    
    def _gaussian_dist_pdf(self, x, mean, cov):
        x = np.asarray(x)
        mean = np.asarray(mean)
        cov = np.asarray(cov)

        n = mean.shape[0]
        diff = x - mean
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)

        norm_const = 1 / np.sqrt((2 * np.pi) ** n * det_cov)
        exponent = -0.5 * diff.T @ inv_cov @ diff

        return norm_const * np.exp(exponent)
    
    def _responsability(self, x, k_means, k_covs):
        k = k_means.shape[0]
        return np.array([self._gaussian_dist_pdf(x, k_means[n, :], k_covs[n, :, :]) for n in range(k)])
    

    def _e_step(self, X, k_pies, k_means, k_covs):
        responsabilities = np.apply_along_axis(func1d=lambda x : self._responsability(x, k_means=k_means, k_covs=k_covs), axis=1, arr=X)
        N_k_pies = np.tile(A=k_pies, reps=(X.shape[0], 1))
        denomerator = np.sum(a=responsabilities * N_k_pies, axis=1) # Normalization constant
        N_denomerator = np.tile(A=denomerator, reps=(k_means.shape[0], 1)).T
        responsabilities = responsabilities * N_k_pies / N_denomerator
        return responsabilities.T


    
    def _m_step(self, X, responsabilities):
        responsabilities_sum = np.sum(a=responsabilities, axis=1)
        Posteriors = responsabilities_sum / X.shape[0]
        means = responsabilities @ X / responsabilities_sum
        covs = []
        for k in range(responsabilities.shape[0]):
            X_centred = X - means[k, :]
            X_weighted = X_centred * np.tile(responsabilities[k, :], reps=(X.shape[1],1)).T
            covs.append(X_weighted.T@X_centred/np.sum(responsabilities[k, :]))
        covs = np.array(covs)
        print(Posteriors.shape, means.shape, covs.shape)
        return Posteriors, means, covs
    
    def fit(self, X):
        np.random.seed(self.seed)
        means = np.random.randn(self.n_components, X.shape[1])
        covs = np.array([np.eye(X.shape[1]) for _ in range(self.n_components)])
        posteriors = np.ones(self.n_components)/self.n_components

        for i in range(self.max_iter):
            responsabilities = self._e_step(X, posteriors, means, covs)
            covs_prev = covs
            posteriors, means, covs = self._m_step(X, responsabilities)
            if np.min(abs(covs - covs_prev) < 0.1):
                break
        self.posteriors, self.means, self.covs = posteriors, means, covs
        return self
    
    def predict_proba(self, X):
        pred = self._e_step(X, self.posteriors, self.means, self.covs)
        return pred.T

    def predict(self, X):
        pred = self._e_step(X, self.posteriors, self.means, self.covs)
        return np.argmax(pred.T, axis=1)


if __name__ == "__main__":

    # generate a dataset
    mean1, mean2 = np.array([0, 0]), np.array([10, 20])
    sigma1, sigma2 = np.array([[1, 0], [0, 1]]), np.array([[5, -5], [-5, 10]])
    np.random.seed(42)
    X1 = np.random.multivariate_normal(mean1, sigma1, 1000)
    np.random.seed(42)
    X2 = np.random.multivariate_normal(mean2, sigma2, 200)
    X = np.vstack([X1, X2])
    gmm = GaussianMixture()
    # responsabilities = gmm.e_step(X=X, k_pies=np.array([0.5, 0.5]), k_means=np.array([mean1, mean2]), k_covs=np.array([sigma1, sigma2]))
    # gmm._m_step(X, responsabilities=responsabilities)
