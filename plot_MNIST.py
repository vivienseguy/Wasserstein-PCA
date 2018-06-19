import mnist
import numpy as np
from Wasserstein_PCA import Wasserstein_PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.cbook as cbook


index = 1
fig = plt.figure(1, figsize=(24.,12.))
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

n=20
L = 28

direction = [1, -1, -1, 1, 1, -1, 1, 1, -1, 1]


def geodesicSample(barycenter_locations, v1, v2, t, sigma):
    gt = barycenter_locations - v1 + t*(v1 + v2)
    x = np.linspace(0, L, L + 1)
    y = np.linspace(0, L, L + 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((L+1, L+1))
    p = gt.shape[0]
    for i in xrange(p):
        Z += gaussian_kernel(X, Y, gt[i,0], gt[i,1], sigma)
    Z = Z/p
    return Z


def gaussian_kernel(x,y, mean_x, mean_y, sigma):
    return 1./(np.sqrt(2*3.14159)*sigma)*np.exp(-((x-mean_x)*(x-mean_x)+(y-mean_y)*(y-mean_y))/(2*sigma*sigma))


for label in xrange(10):

    barycenter_data = np.load('barycenter_data_%d.npz' % (label,))
    barycenter_locations = barycenter_data['locations']
    barycenter_weights = barycenter_data['weights']

    geodesic_data = np.load('principal_geodesic_data_%d.npz' % (label,))
    v1 = geodesic_data['v1']
    v2 = geodesic_data['v2']

    for i in xrange(1,n+1):

        t = float(i-1)/float(n-1) if direction[label] == 1 else 1.-float(i-1)/float(n-1)
        Z = geodesicSample(barycenter_locations, v1, v2, t, 1.15)
        plt.subplot(10, n, index)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.imshow(1-Z, cmap='hot', origin='upper')
        index += 1

plt.show()
fig.savefig("fig_2.pdf", bbox_inches='tight')









