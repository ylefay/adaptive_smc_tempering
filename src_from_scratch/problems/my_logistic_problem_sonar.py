from problems.logistic import get_dataset
from problems.logistic import get_log_likelihood


def get_loglikelihood_fn(dim):
    """
    Defining the loglikelihood function for the Sonar logistic regression,
    restricted to the first dim features.
    """

    flipped_predictors = get_dataset(dataset="Sonar")[:, :dim]
    loglikelihood_fn = get_log_likelihood(flipped_predictors)

    return loglikelihood_fn
