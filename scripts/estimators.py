'''
 * This file is part of the Gas-Leak-Estimation.
 * *
 * @author Kalvik Jakkala
 * @contact kjakkala@uncc.edu
 * Repository: https://github.com/UNCCharlotte-CS-Robotics/Gas-Leak-Estimation
 *
 * Copyright (C) 2020--2022 Kalvik Jakkala.
 * The Gas-Leak-Estimation repo is owned by Kalvik Jakkala and is protected by United States copyright laws and applicable international treaties and/or conventions.
 *
 * The Gas-Leak-Estimation repo is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * DISCLAIMER OF WARRANTIES: THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON-INFRINGEMENT. YOU BEAR ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE OR HARDWARE.
 *
 * SUPPORT AND MAINTENANCE: No support, installation, or training is provided.
 *
 * You should have received a copy of the GNU General Public License along with Gas-Leak-Estimation repo. If not, see <https://www.gnu.org/licenses/>.
'''

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, genextreme
from dispersion_model import GaussianPlume


'''
Implementation of the GEV prior based leak rate estimator presented in "A Mobile Sensing 
Approach for Regional Surveillance of Fugitive Methane Emissions in Oil and Gas Production"
'''
class GEVEstimator:
    '''
    Args:
        s: true leak rate (g/s) of oil well
        location: x, y, z (meters) location of oil well
    '''
    def __init__(self, s, location=[0., 0., 0.]):
        self.model = GaussianPlume(location=location)

        self.S_W = genextreme(-0.10161380555209218,
                              .10243008255436033,
                              0.06567245306678968).pdf
        self.sigma_e = 0.01
        self.int_lim = 10.0
        self.s = s
        self.s_w_entropy = self.S_W_entropy()

    '''
    Method to compute EER at given location
    '''
    def __call__(self, X, Y, Z):
        A = self.dispersion(X, Y, Z)
        s_maw_entropy = self.S_MAW_entropy(A)
        eer = self.s_w_entropy-s_maw_entropy
        return eer

    '''
    Entropy of prior leak rate 
    '''
    def S_W_entropy(self):
        value, error = quad(self.S_W, 0, self.int_lim)
        return -1.0*np.log2(value)

    '''
    Second term in Eq 11
    m: Methane concentraton at location used to generate A
    A: Simulated methane concentration from dispersion model with source rate 1.0    '''
    def S_MAW(self, m, A):
        num = lambda s: np.square(self.M_SAW(m, s, A)*self.S_W(s))
        num_value, error = quad(num, 0, self.int_lim)
        den = lambda s: self.M_SAW(m, s, A)*self.S_W(s)
        den_value, error = quad(den, 0, self.int_lim)
        return np.log2((num_value/(den_value+1e-6)+1e-6))

    def S_MAW_entropy(self, A):
        result = []
        for a in A:
            result.append(self.S_MAW(self.s*a, a))
        return np.sum(result)

    '''
    Eq 13
    m: Methane concentraton at location used to generate A
    S: Source Rate
    A: Simulated methane concentration from dispersion model with source rate 1.0
    '''
    def M_SAW(self, m, s, A):
        return norm.pdf(s*A, m, self.sigma_e)

    '''
    Method to compute leak rate at given location (absolute coordinates)
    the method transforms the location to be relative to the oil well location
    '''
    def dispersion(self, X, Y, Z):
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)
        Z = np.array(Z, dtype=float)
        return self.model(1.0, X, Y, Z)

    '''
    Method to compute the posterior leak rate of oil well given the 
    gas concentration data (M) collected at locations (X, Y, Z)
    '''
    def leak_rate(self, M, X, Y, Z, mode=True, lim=(0, 10)):
        M = np.array(M, dtype=float)
        A = self.dispersion(X, Y, Z)
        x_lim = np.linspace(*lim, 100)
        prob = []
        for s in x_lim:
            P_S_M = self.S_W(s)
            for a, m in zip(A, M):
                den = lambda s1: self.M_SAW(m, s1, a)*self.S_W(s1)
                #den_value, error = quad(den, 0, self.int_lim)
                P_S_M *= self.M_SAW(m, s, a)#/den_value
            prob.append(P_S_M)
        if mode:
            return x_lim[np.argmax(prob)]
        return x_lim, prob


'''
Implementation of the Gaussian prior based leak rate estimator presented in 
"Probabilistic Gas Leak Rate Estimation using Submodular Function Maximization 
with Routing Constraints"
'''
class GaussEstimator:
    '''
    Args:
        s: true leak rate (g/s) of oil well
        location: x, y, z (meters) location of oil well
    '''
    def __init__(self, s, location=[0., 0., 0.]):
        self.model = GaussianPlume(location=location)
        self.mu_s = 0.14768811799978182
        self.sigma_s = 0.65
        self.sigma_e = 0.03
        self.location = np.array(location)
        self.s = s
        self.entropy_s = -np.log2(1/(2*self.sigma_s*np.sqrt(np.pi)))
        self.entropy_s_m_const = np.log2(np.exp(1)/(2*np.sqrt(2)*np.pi))

    '''
    Method to compute EER at given location
    '''
    def __call__(self, X, Y, Z):
        A = self.dispersion(X, Y, Z)
        eer = self._eer(A)
        return eer

    '''
    Method to compute leak rate at given location (absolute coordinates)
    the method transforms the location to be relative to the oil well location
    '''
    def dispersion(self, X, Y, Z):
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)
        Z = np.array(Z, dtype=float)
        return self.model(1.0, X, Y, Z)

    '''
    Method to compute EER at given the A terms and simulated true leak rate 
    '''
    def _eer(self, A):
        entropy_s_m_int = (A*(self.mu_s-self.s))**2
        entropy_s_m_int /= 2*(np.square(A*self.sigma_s)+(self.sigma_e**2))
        entropy_s_m_int = np.sum(entropy_s_m_int)
        entropy_s_m_int *= np.log2(np.exp(1))
        return self.entropy_s + self.entropy_s_m_const + entropy_s_m_int

    '''
    Method to compute the posterior leak rate of oil well given the 
    gas concentration data (M) collected at locations (X, Y, Z)
    '''
    def leak_rate(self, M, X, Y, Z, sigma=False):
        A = self.dispersion(X, Y, Z)
        M = np.array(M, dtype=float)
        mu = ((M@A*self.sigma_s**2) + (self.mu_s*self.sigma_e**2)) / \
             ((A@A*self.sigma_s**2)+self.sigma_e**2)
        if sigma:
            sigma = (2*(self.sigma_e**2)*(self.sigma_s**2)) / \
                    ((A@A*self.sigma_s**2)+self.sigma_e**2)
            return mu, sigma
        return mu

'''
Helper method to compute posterior leak rate from data collected 
along given the path and compute MSE against the true leak rate.
Args:
    path: list of tuples, each tuple has the index of the start node 
          and end node of an edge in the path. 
    locs: Numpy array of coordinates of nodes in the graph 
    leak_estimators: list of gas leak estimator objects, one for each oil well
    num_pts: Number of data samples simulated along of each edge 
'''
def compute_leak_rate_mse(path, locs, leak_estimators, num_pts=10):
    path = set([tuple(sorted(edge)) for edge in set(path)])
    x, y, z = [], [], []
    for edge in path:
        x.extend(np.linspace(locs[edge[0]][1],
                             locs[edge[1]][1],
                             num_pts).tolist())
        y.extend(np.linspace(locs[edge[0]][0],
                             locs[edge[1]][0],
                             num_pts).tolist())
    path = x, y, np.zeros_like(x)

    errors = []
    for well in leak_estimators:
        rate = well.s
        measurements = well.dispersion(*path)*rate
        mu = well.leak_rate(measurements, *path)
        errors.append(np.square(mu-rate))
    error = np.mean(errors)
    return error

'''
Helper method to compute eer of given path
Args:
    path: list of tuples, each tuple has the index of the start node 
          and end node of an edge in the path. 
    locs: Numpy array of coordinates of nodes in the graph 
    leak_estimators: list of gas leak estimator objects, one for each oil well
    num_pts: Number of data samples simulated along of each edge 
'''
def compute_eer(path, locs, leak_estimators, num_pts=10):
    path = set([tuple(sorted(edge)) for edge in set(path)])
    x, y, z = [], [], []
    for edge in path:
        x.extend(np.linspace(locs[edge[0]][1],
                             locs[edge[1]][1],
                             num_pts).tolist())
        y.extend(np.linspace(locs[edge[0]][0],
                             locs[edge[1]][0],
                             num_pts).tolist())
    path = x, y, np.zeros_like(x)
    eer = 0
    for well in leak_estimators:
        eer += well(*path)
    return eer


if __name__=='__main__':
    pass
