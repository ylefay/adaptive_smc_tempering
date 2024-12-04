import pickle

import matplotlib.pyplot as plt
import numpy as np


def save(res, output_path=""):
    with open(output_path, 'wb') as handle:
        pickle.dump({'res': res}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    plt.plot(res[0][:, -1].mean(axis=[1, 2]).T)
    plt.savefig(output_path.replace(".pkl", ".png"))
