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
from functools import partial
from scipy.optimize import root
from scipy.special import gamma
from scipy.stats import multivariate_normal

'''
Simplified gas dispersion model that simulates the leak as a 3D Gaussian. 
Used for testing only, not an accurate gas dispersion simulator. 
'''
class SimplifiedDispersionModel:
    def __init__(self, mean=np.zeros(3), cov=np.eye(3)):
        self.dist = multivariate_normal(mean, cov)

    def __call__(self, S, x, y, z):
        return S*self.dist.pdf(np.dstack([x, y, z]))


'''
Gas dispersion model presented in "Estimation of point source fugitive 
emission rates from a single sensor time series: A conditionally-sampled 
Gaussian plume reconstruction"
'''
class GaussianDispersionModel:
    '''
    Args:
        U:          Wind speed at measurment height z
        z_s:        Height of source
        theta:      Direction of wind
        theta_p:    Direction of peak concentration
        uw:         Product of the Mean covariance of the horizontal and vertical wind components
        air_temp:   Mean air temperature
        sonic_temp: Sonic air temperature
        sigma_y:    Plume width
        wt:         Turbulent Intensity
    '''
    def __init__(self,
                 U=1.26,
                 z_s=1.5,
                 theta=337.4,
                 theta_p=155,
                 uw=0.1,
                 air_temp=16.25,
                 sonic_temp=16.25,
                 sigma_y=0.53,
                 wt=0.069425843):
        self.U = U
        self.z_s = z_s
        self.theta = theta
        self.theta_p = theta_p
        self.uw = uw
        self.air_temp = air_temp
        self.sonic_temp = sonic_temp
        self.sigma_y = sigma_y
        self.wt = wt

        # Constants
        self.c = 0.6
        self.a_1 = 16
        self.a_2 = 16
        self.b_1 = 5
        self.b_2 = 5
        self.k = 0.41
        self.p = 1.55
        self.g = 9.81

        self.u_star = np.sqrt(self.uw)
        self.L = self.obukhov_length()

    def obukhov_length(self):
        return -((self.u_star**3)*self.air_temp)/(self.k*self.g*self.wt)

    def calc_psi(self, z):
        if self.L < 0:
            return ((1-(self.a_2*z/self.L))**(1/4))-1
        else:
            return -(self.b_2*z/self.L)

    def calc_z_o(self, z):
        return z/np.exp((self.U*self.k/self.u_star) + self.calc_psi(z))

    def calc_U_bar(self, z, z_o):
        if self.L < 0:
            z = self.c*z
            return (self.u_star/self.k)*(np.log(z/z_o)-self.calc_psi(z))
        else:
            return (self.u_star/self.k)*(np.log(z/z_o)-self.calc_psi(z))

    def calc_Lx_x_o(self, z_o, z_bar):
        z_o += 1e-6
        if self.L < 0:
            return (z_bar/(self.k**2))*(np.log(self.c*z_bar/z_o)-\
                                        self.calc_psi(self.c*z_bar))*\
                    (1-(self.p*self.a_1*z_bar)/(4*self.L))**(-1/2)
        else:
            return (z_bar/(self.k**2))*((np.log(self.c*z_bar/z_o)+\
                                         ((2*self.b_2*self.p*z_bar)/\
                                          (3*self.L)))*\
                                        (1+(self.b_1*self.p*z_bar)/\
                                         (2*self.L))+(self.b_1/4-self.b_2/6)*\
                                        self.p*z_bar/self.L)

    def calc_z_bar(self, z_o, Lx_x_o, z_bar):
        if np.any(np.isnan(z_bar)) or np.any(z_bar<=0):
            return np.ones_like(z_bar)*100
        return self.calc_Lx_x_o(z_o, z_bar)-Lx_x_o


    def calc_s(self, z_bar, z_o):
        if self.L < 0:
            return ((1-((self.a_1*self.c*z_bar)/(2*self.L)))/(1-((self.a_1*self.c*z_bar)/self.L))) \
                    + (((1-((self.a_2*self.c*z_bar)/self.L))**(-1/4))/\
                    (np.log(self.c*z_bar/z_o)-self.calc_psi(self.c*z_bar)))
        else:
            return (1+((2*self.b_1*self.c*z_bar)/self.L))/(1+((self.b_1*self.c*z_bar)/self.L)) \
                    + (1+((self.b_2*self.c*z_bar)/self.L))/\
                    (np.log(self.c*z_bar/z_o)+self.calc_psi(z_bar))


    def calc_A_B(self, s):
        return s*gamma(2/s)*(gamma(1/s)**2), gamma(2/s)*gamma(1/s)

    def calc_D_z(self, x, z):
        L_eff = x*np.cos(np.degrees(self.theta-self.theta_p))
        z_o = self.calc_z_o(z)
        x_o = self.calc_Lx_x_o(z_o, self.z_s)
        calc_z_bar_partial = partial(self.calc_z_bar, z_o, L_eff+x_o)
        z_bar = root(calc_z_bar_partial, np.ones_like(z)).x
        s = np.abs(self.calc_s(z_bar, z_o))
        A, B = self.calc_A_B(s)
        U_bar = self.calc_U_bar(z_bar, z_o)
        A[A == np.inf] = 0
        B[B == np.inf] = 0
        return (A/z_bar)*np.exp(-np.power(B*z/z_bar, s))/U_bar

    def calc_D_y(self, x, y):
        return (1/(np.sqrt(2*np.pi)*self.sigma_y))*np.exp(-0.5*(y/self.sigma_y)**2)

    '''
    Args:
        S: Stource leak rate
        x: x coordinate (meters)
        y: y coordinate (meters)
        z: z coordinate (meters)
    '''
    def __call__(self, S, x, y, z):
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        z = np.array(z, dtype=float)

        return S*self.calc_D_y(x, y)*self.calc_D_z(x, z)


if __name__=='__main__':
    pass
