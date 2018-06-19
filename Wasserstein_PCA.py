import matplotlib.pyplot as plt
import numpy as np
import ot
import mnist
from sympy import Matrix

class Wasserstein_PCA(object):

    def __init__(self, data_positions, data_weights, barycenter_positions, barycenter_weights):

        self.data_positions = data_positions
        self.data_weights = data_weights
        self.barycenter_positions = barycenter_positions
        self.barycenter_weights= barycenter_weights


    def compute_uniform_free_support_barycenter(self, maxIter, X_init=None):

        iter_count = 0

        d = self.barycenter_positions.shape[1]
        p = self.barycenter_weights.size
        N = len(self.data_positions)

        X = X_init
        X_weights = np.ones(p)/p

        while ( iter_count < maxIter ):

            T_sum = np.zeros((p, d))

            for i in xrange(N):

                T_sum += self.barycentric_projection(X, self.barycenter_weights, self.data_positions[i], self.data_weights[i])

                # M = ot.dist(X, self.data_positions[i])
                # T = ot.emd(X_weights, self.data_weights[i], M)
                # T_sum += np.reshape(1./self.barycenter_weights, (-1,1))*np.matmul(T, self.data_positions[i])

            X = T_sum/N
            iter_count += 1

        return X


    def barycentric_projection(self, source_locations, source_weights, target_locations, target_weights):

        M = ot.dist(source_locations, target_locations)
        T = ot.emd(source_weights, target_weights, M)

        return np.reshape(1./source_weights, (-1,1))*np.matmul(T, target_locations)


    def compute_principal_component(self, previous_v1s=None, previous_v2s=None, require_geodesicity=False, maxIter=20, step_size_t=0.1, step_size_v=0.1, decrease_rate=0., plot_fig=False, v1_0=None, v2_0=None, log_file=None):

        d = self.barycenter_positions.shape[1]
        p = self.barycenter_weights.size
        N = len(self.data_positions)

        t = 0.5*np.ones(N)

        if not v1_0 or not v2_0:
            v1 = 0.25*np.random.randn(p, d)
            v2 = v1
        else:
            v1 = v1_0
            v2 = v2_0

        if previous_v1s:
            previous_vs_aligned = np.zeros(p*d, len(previous_v1s))
            for i in xrange(len(previous_v1s)):
                v = previous_v1s[i]+previous_v2s[i]
                previous_vs_aligned[i,:] =v.ravel()
            A = Matrix(previous_vs_aligned).nullspace()
            orth_dim = A.shape[1]
            Q = []
            for i in xrange(orth_dim):
                Q.append(np.reshape(A[:,i], (p, d)))


        iter_count = 0

        while ( iter_count < maxIter ):

            objective_function = 0

            grad_v1 = np.zeros((p, d))
            grad_v2 = np.zeros((p, d))
            grad_t = np.zeros(N)

            for i in xrange(N):

                geodesic_position_t = self.barycenter_positions - v1 + t[i]*(v1+v2)

                M = ot.dist(geodesic_position_t, self.data_positions[i])
                T = ot.emd(self.barycenter_weights, self.data_weights[i], M)
                objective_function += np.sum(T*M)

                if 1:
                    tmp = (geodesic_position_t - np.reshape(1./self.barycenter_weights, (-1,1))*np.matmul(T, self.data_positions[i]))
                else:
                    tmp = (np.reshape(self.barycenter_weights, (-1,1))*geodesic_position_t - np.matmul(T, self.data_positions[i]))

                grad_v1_tmp = 2*(t[i]-1.)*tmp
                grad_v2_tmp = 2*t[i]*tmp

                grad_v1 += grad_v1_tmp
                grad_v2 += grad_v2_tmp

                grad_t[i] = 2*np.sum(T*np.sum((v1+v2)*geodesic_position_t,1, keepdims=True))-2*np.sum(T*np.matmul(v1+v2, np.transpose(self.data_positions[i])))


            print('Wasserstein PCA: iteration %3d, objective: %f\n' % (iter_count, objective_function))
            iter_count += 1

            t = t - (1./(1.+decrease_rate*iter_count))*step_size_t*grad_t
            t = np.minimum(np.maximum(t, 0.), 1.)

            v1 = v1 - (1./(1.+decrease_rate*iter_count))*step_size_v*grad_v1/N
            v2 = v2 - (1./(1.+decrease_rate*iter_count))*step_size_v*grad_v2/N

            if previous_v1s:
                v_1_p = np.zeros(p, d);
                v_2_p = np.zeros(p, d);
                for k in xrange(orth_dim):
                    v_1_p = v_1_p + np.sum(np.diag(Q[k]*v_1))*Q[k];
                    v_2_p = v_2_p + np.sum(np.diag(Q[k]*v_1))*Q[k];
                v_1 = v_1_p;
                v_2 = v_2_p;

            if require_geodesicity: # barycentric projection to project each v1 and v2 on the optimal velocity fields set
                v1 = self.barycentric_projection(self.barycenter_positions, self.barycenter_weights, self.barycenter_positions - v1, self.barycenter_weights) - self.barycenter_positions
                v2 = self.barycentric_projection(self.barycenter_positions, self.barycenter_weights, self.barycenter_positions + v2, self.barycenter_weights) - self.barycenter_positions

            if log_file:
                log_file.write('iter=%d, loss = %f\n' % (iter_count, objective_function))

            if plot_fig:

                plt.figure(num=1, figsize=(6,6))
                plt.clf()
                axes = plt.gca()
                axes.set_xlim([0, 28])
                axes.set_ylim([0, 28])
                plt.ion()
                plt.scatter(self.barycenter_positions[:,0], self.barycenter_positions[:,1], s=self.barycenter_weights*3000, marker='o')

                for s in xrange(p):
                    plt.plot([self.barycenter_positions[s, 0], self.barycenter_positions[s,0] - v1[s,0]],
                             [self.barycenter_positions[s, 1], self.barycenter_positions[s, 1] - v1[s, 1]])

                for s in xrange(p):
                    plt.plot([self.barycenter_positions[s, 0], self.barycenter_positions[s,0] + v2[s,0]],
                             [self.barycenter_positions[s, 1], self.barycenter_positions[s, 1] + v2[s, 1]])

                plt.title(str(objective_function)+'  (iter=%d)' % (iter_count,))

                plt.draw()
                # plt.show()
                plt.pause(0.1)

        return v1, v2
