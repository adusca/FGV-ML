import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

# Carregando a imagem
img = Image.open("Elvis-nixon.jpg")
data = np.array(img.getdata()).reshape(img.size[1], img.size[0])
data = np.transpose(data)

media = data.mean(axis=0)
# Centralizando e normalizando
data = (data - media)/128

# Calculando a matriz de covariancia
matriz_de_covariancia = np.cov(data)


def save_to_file(ndarr, filename):
    newarr = 128*ndarr + media
    newarr = np.transpose(newarr).reshape(ndarr.shape[0] * ndarr.shape[1])
    image = Image.new('L', (ndarr.shape[0], ndarr.shape[1]))
    image.putdata(newarr)
    image.save(filename + '.png', "PNG")
    image.save(filename + '.jpg', "JPEG")


def plot():
    u, gamma, v = np.linalg.svd(data, full_matrices=False)
    save_to_file(v, "v")
    exit()


u, gamma, v = np.linalg.svd(data, full_matrices=False)


def pca(data, m):
    new_gamma = np.copy(gamma)
    for i in xrange(m, len(gamma)):
        new_gamma[i] = 0
    x_chapeu = np.dot(u, np.dot(np.diag(new_gamma), v))
    erro = np.sum(np.square(data - x_chapeu))
    return x_chapeu, erro


alvo = 0.05*np.sum(np.square(data))
lo = 0
hi = 4000
while lo != hi:
    mid = (lo+hi)/2
    _, error = pca(data, mid)

    if error < alvo:
        hi = mid
    else:
        lo = mid + 1
print lo
img, error = pca(data, lo)
save_to_file(img, "nixon_compressed3")
