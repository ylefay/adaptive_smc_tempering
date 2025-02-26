import pickle

from adaptive_smc.experiments.concat import concat_my_pickles

res = concat_my_pickles(["./exp_rw_vs_ar_gaussian_to_gaussian.py_02022425144047.pkl",
                         "./exp_rw_vs_ar_gaussian_to_gaussian.py_02022425144234.pkl",
                         "./exp_rw_vs_ar_gaussian_to_gaussian.py_02022425144422.pkl",
                         "./exp_rw_vs_ar_gaussian_to_gaussian.py_02022425144607.pkl",
                         "./exp_rw_vs_ar_gaussian_to_gaussian.py_02022425144756.pkl"],
                        [2, 5, 6, 7, 8])
pickle.dump(res, open("./concatenated_AR.pkl", "wb"))

res = concat_my_pickles(["./exp_rw_vs_ar_gaussian_to_gaussian.py_02022425144142.pkl",
                         "./exp_rw_vs_ar_gaussian_to_gaussian.py_02022425144328.pkl",
                         "./exp_rw_vs_ar_gaussian_to_gaussian.py_02022425144516.pkl",
                         "./exp_rw_vs_ar_gaussian_to_gaussian.py_02022425144705.pkl",
                         "./exp_rw_vs_ar_gaussian_to_gaussian.py_02022425144853.pkl"],
                        [2, 5, 6, 7, 8])
pickle.dump(res, open("./concatenated_RW.pkl", "wb"))
