"""Collection of sampling strategies used in experiments."""

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss


def get_samplers(random_state: int = 42):
    """Return a mapping of sampler name -> sampler object (or None).

    Parameters
    ----------
    random_state : int, optional
        Random state to use for samplers that accept it.
    """
    return {
        "No Sampling": None,
        "Random Under": RandomUnderSampler(random_state=random_state),
        "Random Over": RandomOverSampler(random_state=random_state),
        "SMOTE": SMOTE(random_state=random_state),
        "NearMiss": NearMiss()
    }
