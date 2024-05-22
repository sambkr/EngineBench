import numpy as np
from matplotlib.path import Path


class GapHandler:
    """
    Abstract base class for different methods for adding gaps to the data
    """

    def add_gaps(self, u, v, initial_mask):
        raise NotImplementedError


class DefaultHandler(GapHandler):
    """
    Generates square shaped gaps of a fixed size and location for the input snapshot.
    """

    def __init__(self, seed=None):
        self.seed = seed

    def add_gaps(self, u, v, initial_mask):
        u[25:35, 25:35] = 0
        v[25:35, 25:35] = 0
        new_mask = np.zeros_like(u, dtype=np.uint8)
        new_mask[u == 0] = 1
        return u, v, new_mask


class SheetGapHandler(GapHandler):
    """
    Generates randomly shaped block gaps for the input snapshot.
    """

    def __init__(self, seed=None, max_removal_fraction=0.25, central_sheet=True):
        self.seed = seed  # Optional random seed
        self.max_removal_fraction = (
            max_removal_fraction  # Maximum fraction of pixels that can be removed
        )
        self.central_sheet = central_sheet

    def add_gaps(self, inputu, inputv, initial_mask):
        if self.seed is not None:
            np.random.seed(self.seed)

        u, v = inputu.copy(), inputv.copy()
        height, width = u.shape[:2]

        initial_mask[initial_mask != 0] = 1  # Enforce non-zero values to be 1
        initial_mask_inv = (
            1 - initial_mask
        )  # Inverted mask shows locations of data points

        n_data_points = np.sum(initial_mask_inv)

        # Function to create random points on the edges
        def create_edge_points():
            top_points = [(x, 0) for x in np.random.choice(width, 2, replace=False)]
            bottom_points = [
                (x, height - 1) for x in np.random.choice(width, 2, replace=False)
            ]
            left_points = [(0, y) for y in np.random.choice(height, 2, replace=False)]
            right_points = [
                (width - 1, y) for y in np.random.choice(height, 2, replace=False)
            ]
            return top_points + bottom_points + left_points + right_points

        acceptable_mask = False
        while not acceptable_mask:
            all_points = create_edge_points()

            # Randomly select four points to form a polygon
            selected_indices = np.random.choice(len(all_points), 4, replace=False)
            points = np.array([all_points[i] for i in selected_indices])

            # Create a Path object from the polygon points
            polygon_path = Path(points)

            # Create a grid of points representing the image
            y_grid, x_grid = np.mgrid[:height, :width]
            grid_points = np.vstack((x_grid.flatten(), y_grid.flatten())).T

            # Use the path to create a mask
            polygon_mask = polygon_path.contains_points(grid_points).reshape(
                (height, width)
            )

            # Calculate masks for the number of new gaps to be added inside/outside the polygon
            inside_mask = np.logical_and(initial_mask_inv, polygon_mask)
            outside_mask = np.logical_and(
                initial_mask_inv, np.logical_not(polygon_mask)
            )

            if self.central_sheet:
                removal_mask = inside_mask
            else:
                removal_mask = outside_mask

            # Check if the removal is within the acceptable threshold
            if np.sum(removal_mask) <= self.max_removal_fraction * n_data_points:
                acceptable_mask = True

        # Apply the mask to the input data
        u[removal_mask] = 0
        v[removal_mask] = 0
        new_mask = removal_mask.astype(np.uint8)

        return u, v, new_mask


class ClusteredDropoutHandler(GapHandler):
    def add_gaps(self, u, v, initial_mask):
        # Implementation for clustered dropouts
        pass
