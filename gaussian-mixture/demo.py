import os
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__),'old_faithful.txt')
xs = np.loadtxt(path)

phis = np.array([0.5, 0.5])
mus = np.array([[0.0, 50.0],[0.0, 100.0]])
covs = np.array([np.eye(2),np.eye(2)])
print(covs)

K = len(phis)
N = len(xs)

max_iterate = 1000
threshold = 1e-4

def multivariate_normal(x,mu,cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    d = len(x)
    return np.exp(-(x - mu).T @ inv @ (x-mu) / 2) / np.sqrt((2 * np.pi) ** d * det)

def gmm(x,phis,mus,covs):
    K = len(phis)
    res = 0
    for k in range(K):
        res += phis[k]*multivariate_normal(x,mus[k],covs[k])
    return res

# 対数尤度を求める
def likelihood(xs,phis,mus,covs):
    eps = 1e-8
    res = 0
    N = len(xs)
    for x in xs:
        y = gmm(x,phis,mus,covs)
        res += np.log(y+eps)
    res /= N
    return res

current_likelihood = likelihood(xs,phis,mus,covs)

for iter in range(max_iterate):
    # E-step
    qs = np.zeros((N,K))
    for n in range(N):
        x = xs[n]
        for k in range(K):
            qs[n,k] = phis[k]*multivariate_normal(x,mus[k],covs[k])
        qs[n] /= gmm(x,phis,mus,covs)
    
    # M-step
    qs_sum = qs.sum(axis=0)
    for k in range(K):
        phis[k] = qs_sum[k] / N

        tmp = 0
        for n in range(N):
            tmp += qs[n,k]*xs[n]
        mus[k] = tmp / qs_sum[k]
        
        tmp = 0
        for n in range(N):
            z = xs[n] - mus[k]
            z = z[:,np.newaxis] # (k) -> (k,1)
            tmp += qs[n,k] * z @ z.T
        covs[k] = tmp / qs_sum[k]
    
    next_likelihood = likelihood(xs,phis,mus,covs)
    diff = np.abs(next_likelihood-current_likelihood)
    if diff < threshold:
        break
    current_likelihood = next_likelihood

def plot_contour(w, mus, covs):
    x = np.arange(1, 6, 0.1)
    y = np.arange(40, 100, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])

            for k in range(len(mus)):
                mu, cov = mus[k], covs[k]
                Z[i, j] += w[k] * multivariate_normal(x, mu, cov)
    plt.contour(X, Y, Z)


plt.scatter(xs[:,0], xs[:,1],c="red",label="original")
plot_contour(phis,mus,covs)

data = np.zeros((N,2))
for n in range(N):
    k = np.random.choice(2,p=phis)
    mu, cov = mus[k], covs[k]
    data[n] = np.random.multivariate_normal(mu, cov)

plt.scatter(data[:,0],data[:,1],c="blue",label="generated")
plt.legend()
plt.show()