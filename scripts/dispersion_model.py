'''
 * This file is part of the Gas-Leak-Estimation.
 * *
 * @author Kalvik Jakkala
 * @contact kjakkala@uncc.edu
 * Repository: https://github.com/UNCCharlotte-CS-Robotics/Gas-Leak-Estimation
 *
 * DISCLAIMER OF WARRANTIES: THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON-INFRINGEMENT. YOU BEAR ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE OR HARDWARE.
 *
 * SUPPORT AND MAINTENANCE: No support, installation, or training is provided.
 *
'''

"""Copyright 2016 National Technology & Engineering 
    Solutions of Sandia, LLC (NTESS). Under the terms of Contract 
	DE-NA0003525 with NTESS, the U.S. Government retains certain rights 
	in this software.
    
    Revised BSD License

    Original code repo: https://github.com/sandialabs/chama
"""

"""
The simulation module contains methods to run Gaussian air dispersion models. 
Chama can also integrate simulations from third party software for additional
sensor placement applications.
"""

import numpy as np


def _calculate_sigma(x, stability_class):
    """
    Calculates sigmay and sigmaz as a function of grid points in the 
    direction of travel (x) for stability class A through F.

    Parameters
    ---------------
    x: numpy array
        Grid points in the direction of travel (m)
    stability_class : string
        Stability class, A through F
        
    Returns
    ---------
    sigmay: numpy array
        Standard deviation of the Gaussian distribution in the horizontal
        (crosswind) direction (m)
    sigmaz: numpy array
        Standard deviation of the Gaussian distribution in the vertical
        direction (m)
    """
    if stability_class == 'A':
        k = [0.250, 927, 0.189, 0.1020, -1.918]
    elif stability_class == 'B':
        k = [0.202, 370, 0.162, 0.0962, -0.101]
    elif stability_class == 'C':
        k = [0.134, 283, 0.134, 0.0722, 0.102]
    elif stability_class == 'D':
        k = [0.0787, 707, 0.135, 0.0475, 0.465]
    elif stability_class == 'E':
        k = [0.0566, 1070, 0.137, 0.0335, 0.624]
    elif stability_class == 'F':
        k = [0.0370, 1170, 0.134, 0.0220, 0.700]
    else:
        return

    sigmay = k[0] * x / (1 + x / k[1]) ** k[2]
    sigmaz = k[3] * x / (1 + x / k[1]) ** k[4]

    return sigmay, sigmaz


class GaussianPlume:
    
    def __init__(self, 
                 location=(0, 0, 0),
                 wind_direction=0,
                 wind_speed=1.0,
                 stability_class='A'):
        """
        Defines the Gaussian plume model.
        
        Parameters
        ---------------
        location: tuple (x, y, z)
            x, y, z location of the source (m)
        wind_direction: float
            Wind direction (degrees)
        wind_speed: float
            Wind speed (m/s)
        stability_class: string
            Stability class, A through F            
        """

        self.loc = location
        self.wind_direction = wind_direction
        self.wind_speed = wind_speed
        self.stability_class = stability_class


    def _modify_grid(self, X, Y):
        """
        Rotates grid to account for wind direction.
        Translates grid to account for source location.

        Returns
        ---------
        X: numpy array
            x values in the field (m)
        Y: numpy array
            y values in the field (m)
        """

        angle_rad = self.wind_direction / 180.0 * np.pi
        Xloc = (X - self.loc[0]) * np.cos(angle_rad) \
                + (Y - self.loc[1]) * np.sin(angle_rad)
        Yloc = - (X - self.loc[0]) * np.sin(angle_rad) \
                + (Y - self.loc[1]) * np.cos(angle_rad)

        Xloc[Xloc < 0] = 0
            
        return Xloc, Yloc


    def __call__(self, rate, X, Y, Z):
        """
        Computes the concentrations of a Gaussian plume.

        Parameters
        ---------------
        rate: float
            source leak rate (kg/s)
        X: numpy array
            x values (absolute coordinates) in the field (m)
        Y: numpy array
            y values (absolute coordinates) in the field (m)
        Z: numpy array
            z values (absolute coordinates) in the field (m)
        """

        X2, Y2 = self._modify_grid(X, Y)
        sigmay, sigmaz = _calculate_sigma(X2, self.stability_class)
        
        a = np.zeros(X2.shape)
        b = np.zeros(X2.shape)
        c = np.zeros(X2.shape)

        a[X2 > 0] = rate / \
                (2 * np.pi * self.wind_speed * sigmay[X2 > 0] * sigmaz[X2 > 0])
        b[X2 > 0] = np.exp(-Y2[X2 > 0] ** 2 / (2 * sigmay[X2 > 0] ** 2))
        c[X2 > 0] = np.exp(-(Z[X2 > 0] - self.loc[2]) ** 2 /
                            (2 * sigmaz[X2 > 0] ** 2)) \
                    + np.exp(-(Z[X2 > 0] + self.loc[2]) ** 2 /
                            (2 * sigmaz[X2 > 0] ** 2))
        
        conc = a * b * c
        conc[np.isnan(conc)] = 0
        
        return conc


if __name__=='__main__':
    pass
