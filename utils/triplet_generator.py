"""Methods to generate triplets based on year classification. Negative samples should have a larger year distance from anchor than positive samples."""

import numpy as np
from numpy.random import multinomial
from scipy.stats import norm


def pick_index_of_positive_sample(anchor_index: int, years: np.ndarray, epsilon: float = 1e-1) -> int:
    """Given the index of an anchor sample, select a positive sample from the same batch to be used in triplet loss.

    Sampling strategy -- Pick the car in the batch whose year is closest to a draw from some narrow Gaussian distribution X ~ (anchor_year, stdev),
    where stdev is chosen as follows:

        - Find the min and max year in the batch
        - Divide the closest distance between anchor year and either min/max year by 5
    
    This process ensures a distribution that is centered at anchor_year and almost never goes outside the 
    temporal boundaries of the batch's years (when it does, we round to the nearest boundary).

    Args:
        anchor_index (int): Index of anchor sample to find a positive sample for.
        years (np.ndarray): Array containing the years of all cars in the batch.
        epsilon (float): Specifies standard deviation when the above formula returns 0.

    Returns:
        positive_index: Index of the positive sample selected
    """
    anchor_year = years[anchor_index]

    max_year, min_year = years.max(), years.min()
    stdev = (min(abs(max_year - anchor_year), abs(min_year - anchor_year)) / 5) + epsilon     # Standard deviation of 0 can occur when anchor year is on boundary

    probabilities = norm.pdf(years, loc=anchor_year, scale=stdev)
    probabilities[anchor_index] = 0

    """
    We take a sample uniformly from the closest year if our normal densities are zero
    everywhere (this happens when the anchor is in an outlier year relative to the batch)
    """
    if np.allclose(probabilities, np.zeros_like(probabilities)):

        year_dists = np.abs(years - anchor_year, dtype=np.float64)
        year_dists[anchor_index] = np.inf
        closest_year = years[np.argmin(year_dists)]

        probabilities[years == closest_year] = 1
        probabilities[anchor_index] = 0

    draw = np.random.multinomial(1, probabilities / probabilities.sum())
    positive_index = draw.tolist().index(1)

    return positive_index

def pick_index_of_negative_sample(anchor_index: int, years: np.ndarray) -> int:
    """Given the index of an anchor sample, select a negative sample from the same batch to be used in triplet loss.

    Sampling strategy-- Find the car in the batch whose year is drawn from X ~ Categorical(d) on all years present in the batch,
    where d is computed as follows:

        - Find the squared distance from each year in the batch to the anchor_year
        - Divide the array of absolute distances by its sum to get d. Years further away from anchor_year will now
          be assigned much higher probabilities than years close to anchor_year for the categorical draw.

    Args:
        anchor_index (int): Index of anchor sample to find a negative sample for.
        years (np.ndarray): Array containing the years of all cars in the batch.

    Returns:
        negative_index: Index of the negative sample selected
    """
    anchor_year = years[anchor_index]

    squared_distances = (years - anchor_year) ** 2
    probabilities = squared_distances / squared_distances.sum()
    print(probabilities.dtype)
    draw = multinomial(1, probabilities)
    negative_index = draw.tolist().index(1)

    return negative_index
    