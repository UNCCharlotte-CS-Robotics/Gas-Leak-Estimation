import time
import numpy as np
from collections import defaultdict
from estimators import GEVEstimator, GaussEstimator

if __name__=='__main__':
    gev_time = defaultdict(lambda: [])
    gauss_time = defaultdict(lambda: [])
    gev_eer = defaultdict(lambda: [])
    gauss_eer = defaultdict(lambda: [])

    model_gev = GEVEstimator(0.6)
    model_gau = GaussEstimator(0.6)

    y = np.linspace(-1, 1, 100)
    x = np.ones_like(y)
    z = np.ones_like(x)

    for dist in [0.2, 0.8, 1.4, 2.4, 3.0, 5.0]:
        for _ in range(10):
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

        print("{} & {:.4E} & {:.4E} & {:.5f} & {:.5f} \\".format(dist,
                                                                 np.mean(gev_eer[dist]),
                                                                 np.mean(gauss_eer[dist]),
                                                                 np.mean(gev_time[dist]),
                                                                 np.mean(gauss_time[dist])))
