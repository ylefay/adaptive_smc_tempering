import pickle


def save(res, config, title_keys, additional_title_elements, output_path=""):
    """
    Saving in a PKL file the config dictionnary and the output of the SMC sampler.
    In addition, a plot of the means with error bars for each tempered distribution is saved in a PNG file.
    """
    with open(output_path, 'wb') as handle:
        pickle.dump({'config': config, 'res': res}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """n_particles = res[0].shape[2] * res[0].shape[3]
    errs_std = np.std(res[0][:, -1].reshape((*res[0][:, -1].shape[:1], n_particles, res[0].shape[-1])), axis=1)
    means = res[0][:, -1].mean(axis=[1, 2])
    for idx in range(means.shape[0]):
        plt.errorbar(x=np.arange(1, 1 + res[0].shape[-1]), y=means[idx], yerr=errs_std[idx])

    title = None
    if title_keys:
        title = ' '.join([str(config[k]) for k in title_keys])
    if additional_title_elements:
        title = title + ' '.join([str(k) for k in additional_title_elements])
    if title:
        plt.title(title)

    plt.savefig(output_path.replace(".pkl", ".png"))
    plt.clf()"""
