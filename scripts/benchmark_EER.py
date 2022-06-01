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

import time
import numpy as np
from collections import defaultdict
from estimators import GEVEstimator, GaussEstimator

if __name__=='__main__':
    gev_time = defaultdict(lambda: [])
    gauss_time = defaultdict(lambda: [])
    gev_eer = defaultdict(lambda: [])
    gauss_eer = defaultdict(lambda: [])

    model_gev = GEVEstimator(1.0)
    model_gau = GaussEstimator(1.0)

    y = np.linspace(-1, 1, 100)
    x = np.ones_like(y)
    z = np.ones_like(x)

    for dist in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        for _ in range(1):
            start_gev = time.time()
            eer_gev = model_gev(x*dist, y, z)
            end_gev = time.time()

            start_gau = time.time()
            eer_gau = model_gau(x*dist, y, z)
            end_gau = time.time()

            gev_time[dist].append(end_gev-start_gev)
            gev_eer[dist].append(eer_gev)
            gauss_time[dist].append(end_gau-start_gau)
            gauss_eer[dist].append(eer_gau)

        print("{} & {:.6E} & {:.6E} & {:.5f} & {:.5f} \\\\".format(dist,
                                                                 np.mean(gev_eer[dist]),
                                                                 np.mean(gauss_eer[dist]),
                                                                 np.mean(gev_time[dist]),
                                                                 np.mean(gauss_time[dist])))

