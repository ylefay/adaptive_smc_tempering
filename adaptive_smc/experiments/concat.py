import pickle
from typing import List

import jax.numpy as jnp


def concat_my_pickles(list_of_path_of_pickles: List[str], list_idxs: List[int]):
    """
    Loading successively each pickle and concatenating the selected fields
    Assuming compatible shapes
    Assuming identical configurations.
    """
    first_pickle = pickle.load(open(list_of_path_of_pickles[0], "rb"))
    config = first_pickle['config']
    res = first_pickle['res']
    number_of_pickles = len(list_of_path_of_pickles)
    my_concat = {idx: jnp.zeros(shape=(res[idx].shape[0] * number_of_pickles,) + (res[idx].shape[1:]),
                                dtype=res[idx].dtype)
                 for
                 idx in list_idxs}
    for idx_pickle, path in enumerate(list_of_path_of_pickles):
        _f = open(path, "rb")
        f = pickle.load(_f)
        for idx in list_idxs:
            my_concat[idx] = my_concat[idx].at[
                             idx_pickle * res[idx].shape[0]: (idx_pickle + 1) * res[idx].shape[
                                 0]].set(
                f['res'][idx])
        _f.close()

    return {'res': my_concat, 'config': config}
