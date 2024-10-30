import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt

from src.utils.stats import *


def default_title():
    now = datetime.now()

    def my_fun():
        output_path = f"{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
        return output_path

    return my_fun


def save(chain, title, config, output_path=""):
    if output_path == "":
        output_path = default_title()
    to_save = {'title': title, 'config': config,
               'acceptance_rate': chain[1].update_info.acceptance_rate.mean(axis=-1).mean(axis=-1).T,
               'acceptance_ratio': chain[1].update_info.acceptance_rate,
               'esjd': esjd(chain),
               'logZ_logW': logZ_logW(chain),
               'lmbda': chain[0][0].lmbda,
               'paths': chain[0][0][0][0]}

    with open(output_path, 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot(chain, title):
    paths = chain[0][0][0][0][:, -1]
    N_chains = len(paths)
    plt.clf()
    plt.plot(chain[1].update_info.acceptance_rate.mean(axis=-1).mean(axis=-1).T)
    plt.title("Acceptance rate")
    plt.xlabel("Tempering step")
    plt.savefig(f"{title}_acceptance_rate.png")
    plt.clf()
    plt.plot(paths.mean(axis=1).mean(axis=0))
    plt.title("Mean over chains and particles")
    plt.xlabel("Component")
    plt.savefig(f"{title}_mean_over_chains_particles.png")
    plt.clf()
    for idx in range(N_chains):
        plt.plot(paths[idx].mean(axis=0))
    plt.title("Mean over particles")
    plt.xlabel("Component")
    plt.savefig(f"{title}_all_chains.png")
    plt.clf()
    for idx in range(N_chains):
        plt.plot(chain[0][0].lmbda[idx])
    plt.xlabel("Tempering step")
    plt.title("Temperature")
    plt.savefig(f"{title}_lmbda_all_chains.png")
    plt.clf()
    esjds = esjd(chain)
    for idx in range(N_chains):
        plt.plot(esjds[idx])
    plt.xlabel("Tempering step")
    plt.title("ESJD")
    plt.savefig(f"{title}_esjd_all_chains.png")
    plt.clf()
    logZ, _ = logZ_logW(chain)
    plt.plot(logZ[:, -1])
    plt.title("Log Z")
    plt.xlabel("Tempering step")
    plt.savefig(f"{title}_logZ_over_chains.png")
    plt.clf()
    plt.plot(logZ.var(axis=0))
    plt.title("Log Z variance computed over the chains")
    plt.xlabel("Tempering step")
    plt.savefig(f"{title}_varlogZ_over_chains.png")
    plt.clf()
