import pickle

import matplotlib.pyplot as plt
import numpy as np


def save(res, config, title_keys, output_path="", ):
    with open(output_path, 'wb') as handle:
        pickle.dump({'config': config, 'res': res}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    n_particles = res[0].shape[2] * res[0].shape[3]
    errs_std = np.std(res[0][:, -1].reshape((*res[0][:, -1].shape[:1], n_particles, res[0].shape[-1])), axis=1)
    means = res[0][:, -1].mean(axis=[1, 2])
    for idx in range(means.shape[0]):
        plt.errorbar(x=np.arange(1, 1 + res[0].shape[-1]), y=means[idx], yerr=errs_std[idx])
    if title_keys:
        title = ' '.join([str(config[k]) for k in title_keys])
        plt.title(title)
    plt.savefig(output_path.replace(".pkl", ".png"))
