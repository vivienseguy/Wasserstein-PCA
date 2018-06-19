from matplotlib import use
use('TkAgg')
import mnist
import numpy as np
from Wasserstein_PCA import Wasserstein_PCA
import matplotlib.pyplot as plt



n = 60000
images, labels = mnist.read_MNIST_dataset(size=n, one_hot=False)
images = np.squeeze(images)
images = np.reshape(images, (n,-1))
L = 28
N = 10
geodesic_number = 2


for label in xrange(10):

    I = np.where(labels == label)[0]
    images_tmp = images[I,:]
    images_tmp = images_tmp[:N,:]

    x = np.linspace(0, L-1, L)
    y = np.linspace(0, L-1, L)
    xv, yv = np.meshgrid(x, y, indexing='xy')
    pixels_locations = np.concatenate((np.reshape(np.ravel(xv),(-1,1)), np.reshape(np.ravel(yv),(-1,1))), 1)

    data_locations = []
    data_weights = []
    for i in xrange(N):
        I = np.nonzero(images_tmp[i,:])
        S = np.sum(images_tmp[i, :])
        data_locations.append(pixels_locations[I[0],:])
        weights = images_tmp[i, I[0]]
        weights = weights.astype(np.float64)/S
        data_weights.append(weights)


    if False:

        plt.xlim(0,L)
        plt.ylim(0, L)
        plt.ion()
        plt.show()

        x = np.linspace(0, L - 1, 10)
        y = np.linspace(0, L - 1, 10)
        xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
        X_init = np.concatenate((np.reshape(np.ravel(xv), (-1, 1)), np.reshape(np.ravel(yv), (-1, 1))), 1)

        p = X_init.shape[0]
        X_weights = np.ones(p)/p
        WPCAObject = Wasserstein_PCA(data_locations, data_weights, X_init, X_weights)
        barycenter_locations = WPCAObject.compute_uniform_free_support_barycenter(maxIter=80, X_init=X_init)

        plt.clf()
        plt.scatter(barycenter_locations[:, 0], L - barycenter_locations[:, 1], s=X_weights * 3000, marker='o')
        plt.draw()
        plt.pause(1.)
        file = np.savez('barycenter_data_%d.npz' % (label,), locations=barycenter_locations, weights=X_weights)

    else:

        data = np.load('barycenter_data_%d.npz' % (label,))
        barycenter_locations = data['locations']
        barycenter_weights = data['weights']
        WPCAObject = Wasserstein_PCA(data_locations, data_weights, barycenter_locations, barycenter_weights)

    log_file = open("'principal_geodesic_log_%d.txt" % (label,), "w")
    previous_v1s = []
    previous_v2s = []
    for geodesic_index in xrange(geodesic_number):
        v1, v2 = WPCAObject.compute_principal_component(previous_v1s=previous_v1s, previous_v2s=previous_v2s, maxIter=300, step_size_t=0.05, step_size_v=1., decrease_rate=0.015, plot_fig=True, v1_0=None, v2_0=None, log_file=log_file)
        previous_v1s.append(v1)
        previous_v2s.append(v2)
        log_file.close()
        file = np.savez('principal_geodesic_data_%d_%d.npz' % (label, geodesic_index), v1=v1, v2=v2)







