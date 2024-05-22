import numpy as np


def create_test_mask(old_mask, max_prop: float, mode="horiz"):
    """
    Expects a single 2D mask
    """
    old_mask[old_mask != 0] = 1  # Enforce non-zero values to be 1
    old_mask_inv = 1 - old_mask  # Inverted mask shows locations of data points
    nPoints = np.count_nonzero(old_mask_inv)

    test_mask = old_mask.copy()

    # Iteratively find largest mask that doesn't exceed max_prop
    sum_new_gaps = 0
    n = 1
    if mode == "horiz":
        while sum_new_gaps < nPoints * max_prop:
            test_mask[:n, :] = 1
            test_mask[-n:, :] = 1

            sum_new_gaps = np.count_nonzero(test_mask) - np.count_nonzero(old_mask)

            n += 1

        final_n = n - 2
        test_mask = old_mask.copy()
        test_mask[:final_n, :] = 1
        test_mask[-final_n:, :] = 1

    elif mode == "vert":
        while sum_new_gaps < nPoints * max_prop:
            test_mask[:, :n] = 1
            test_mask[:, -n:] = 1

            sum_new_gaps = np.count_nonzero(test_mask) - np.count_nonzero(old_mask)

            n += 1

        final_n = n - 2
        test_mask = old_mask.copy()
        test_mask[:, :final_n] = 1
        test_mask[:, -final_n:] = 1

    else:
        raise ValueError(f"Mask type {mode} not implemented")

    sum_new_gaps = np.count_nonzero(test_mask) - np.count_nonzero(old_mask)
    print(f"Proportion of test gaps added: {sum_new_gaps/nPoints:.2f}")

    return test_mask, final_n
