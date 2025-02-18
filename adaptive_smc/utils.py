import pickle


def save(res, config, title_keys, additional_title_elements, output_path=""):
    """
    Saving in a PKL file the config dictionnary and the output of the SMC sampler.
    In addition, a plot of the means with error bars for each tempered distribution is saved in a PNG file.
    """
    with open(output_path, 'wb') as handle:
        pickle.dump({'config': config, 'res': res}, handle, protocol=pickle.HIGHEST_PROTOCOL)
