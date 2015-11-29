import numpy as np
import sklearn.linear_model


absorp = np.genfromtxt("absorp.csv", delimiter=",")
train = absorp[:200, :]
test = absorp[200:, :]
media = train.mean(axis=0)
train = (train - media)
abs_max = np.amax(np.abs(train))
train = train/abs_max
test = (test - media)/abs_max

y = np.genfromtxt("endpoints.csv", delimiter=",")[:, 1]
y_train = y[:200]
y_test = y[200:]


# Varios modelos
print y_test
# Ridge
ridge = sklearn.linear_model.Ridge(alpha=0.01)
ridge.fit(train, y_train)

previsto = ridge.predict(test)
print "Ridge"
print previsto
print np.sum(np.square(previsto - y_test))/len(previsto)

# Lasso
lasso = sklearn.linear_model.Lasso(alpha=0.01, max_iter=10000)
lasso.fit(train, y_train)

previsto = lasso.predict(test)
print "Lasso"
print previsto
print np.sum(np.square(previsto - y_test))/len(previsto)


####
u, gamma, v = np.linalg.svd(train, full_matrices=False)


def pca(data, m):
    new_gamma = np.copy(gamma)
    for i in xrange(m, len(gamma)):
        new_gamma[i] = 0
    x_chapeu = np.dot(u, np.dot(np.diag(new_gamma), v))
    erro = np.sum(np.square(data - x_chapeu))
    return x_chapeu, erro


alvo = 0.01*np.sum(np.square(train))
lo = 0
hi = 100

while lo != hi:
    mid = (lo+hi)/2
    _, error = pca(train, mid)

    if error < alvo:
        hi = mid
    else:
        lo = mid + 1

dimensions = lo
