import numpy as np


def algo1(X, y, d, n, k):
    u, gamma, v = np.linalg.svd(X, full_matrices=False)
    Vk = v[:, :k]
    z = np.dot(X, Vk)
    E1 = 0.0
    for i in xrange(n):
        # Remove (zn, yn)
        Zn = np.delete(z, [i], axis=0)
        yn = np.delete(y, [i], axis=0)
        # Calcula wn
        wn = np.linalg.lstsq(Zn, yn)[0]
        # Calcula o erro
        en = (np.dot(X[i, :], np.dot(Vk, wn)) - y[i])**2
        E1 += en[0]
    return E1/n


def algo2(X, y, d, n, k):
    E2 = 0.0
    for i in xrange(n):
        # Remove (Xn, yn)
        Xn = np.delete(X, [i], axis=0)
        yn = np.delete(y, [i], axis=0)
        u, gamma, v = np.linalg.svd(Xn, full_matrices=False)
        Vk = v[:, :k]
        Zn = np.dot(Xn, Vk)
        # Calcula wn
        wn = np.linalg.lstsq(Zn, yn)[0]
        # Calcula o erro
        en = (np.dot(X[i, :], np.dot(Vk, wn)) - y[i])**2
        E2 += en[0]
    return E2/n


def gera_experimento(d, N):
    w = np.random.normal(size=(d, 1))
    X = np.random.normal(size=(N, d))
    epsilon = np.random.normal(scale=0.5, size=(N, 1))
    y = np.dot(X, w) + epsilon
    return w, X, y


def calcula_eout(X, y, k, w):
    u, gamma, v = np.linalg.svd(X, full_matrices=False)
    Vk = v[:, :k]
    Z = np.dot(X, Vk) # n x k
    wp = np.dot(Vk, np.linalg.lstsq(Z, y)[0])
    werr = w - wp
    return 0.5 + np.dot(np.transpose(werr), werr)[0][0]

E1_out = 0
E2_out = 0
E_out = 0
testes = 10**4
for i in xrange(testes):
    w, X, y = gera_experimento(5, 40)
    E1_out += algo1(X, y, 5, 40, 3)/testes
    E2_out += algo2(X, y, 5, 40, 3)/testes
    E_out += calcula_eout(X, y, 3, w)/testes

print 1, E1_out
print 2, E2_out
print "real", E_out
