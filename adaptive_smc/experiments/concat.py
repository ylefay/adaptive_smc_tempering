import pickle
from typing import List

import jax.numpy as jnp


def concat_my_pickles(list_of_path_of_pickles: str, list_idxs: List[int]):
    """
    Loading successively each pickle and concatenating the selected fields
    Assuming compatible shapes
    """
    first_pickle = pickle.load(open(list_of_path_of_pickles[0], "rb"))['res']
    number_of_pickles = len(list_of_path_of_pickles)
    my_concat = {idx: jnp.zeros(shape=(first_pickle[idx].shape[0] * number_of_pickles) + (first_pickle[idx].shape[1:]))
                 for
                 idx in list_idxs}
    for idx_pickle, path in enumerate(list_of_path_of_pickles):
        with pickle.load(open(path, "rb")) as f:
            for idx in list_idxs:
                my_concat[idx] = my_concat[idx].at[
                                 idx_pickle * first_pickle[idx].shape[0]: (idx_pickle + 1) * first_pickle[idx].shape[
                                     0]].set(
                    f['res'][idx])
    return {'res': my_concat}
