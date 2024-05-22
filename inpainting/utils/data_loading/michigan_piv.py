import numpy as np
import h5py
import torch
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from .gap_handler import GapHandler


class MichiganPIV(VisionDataset):
    """
    Data loading class
    """

    def __init__(
        self,
        data_path: str,
        grp_dict: dict,
        cad_dict: dict,
        grp_cad_idx: tuple,
        target_img_shape: tuple,
        gap_handler: GapHandler,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            data_path, transform=transform, target_transform=target_transform
        )
        self.train = train
        self.data_path = data_path
        self.grp_dict = grp_dict
        self.cad_dict = cad_dict
        self.grp_cad_idx = grp_cad_idx
        self.target_img_shape = target_img_shape
        self.gap_handler = gap_handler

        self.metadata = self._load_metadata()

    def _load_metadata(self):
        with h5py.File(self.data_path, "r") as f:
            grp = f[self.grp_dict[self.grp_cad_idx[0]]]  # Load test point
            data = grp[self.cad_dict[self.grp_cad_idx[1]]]  # Load crank angle

            # Generate the original x y grid
            x = data[0, :, 0]
            y = data[0, :, 1]
            unique_x = np.unique(x)
            unique_y = np.unique(y)
            gridx, gridy = np.meshgrid(unique_x, unique_y)
            img_shape = gridx.shape

            # Create the target_img_shape grid and padding for later use
            paddedx, paddedy, pad_x, pad_y = MichiganPIV.create_target_grid(
                unique_x, unique_y, self.target_img_shape
            )

            metadata = {
                "data_shape": data.shape,
                "nSnaps": int(len(data)),
                "mean_u": data.attrs["mean_u"],
                "std_u": data.attrs["std_u"],
                "mean_v": data.attrs["mean_v"],
                "std_v": data.attrs["std_v"],
                "img_shape": img_shape,
                "paddedx": paddedx,
                "paddedy": paddedy,
                "pad_x": pad_x,
                "pad_y": pad_y,
            }

            print(
                f"Loaded test point: {self.grp_dict[self.grp_cad_idx[0]]}; CAD: {self.cad_dict[self.grp_cad_idx[1]]}"
            )

        return metadata

    def __len__(self) -> int:
        return self.metadata["nSnaps"]

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        idx = idx
        u, v, initial_mask, scales = self._load_data(idx)

        # Scale data
        meta = self.metadata
        scaledu = (u - scales["mean_u"]) / scales["std_u"]
        scaledv = (v - scales["mean_v"]) / scales["std_v"]

        # Apply gap handling strategy
        blockedu, blockedv, new_mask = self.gap_handler.add_gaps(
            scaledu, scaledv, initial_mask
        )

        # Ensure that gaps are 0s
        scaledu[(initial_mask != 0)] = 0
        scaledv[(initial_mask != 0)] = 0
        blockedu[(new_mask != 0) | (initial_mask != 0)] = 0
        blockedv[(new_mask != 0) | (initial_mask != 0)] = 0

        # Add padding to create target shape
        paddedu = MichiganPIV.add_padding(scaledu, meta["pad_x"], meta["pad_y"])
        paddedv = MichiganPIV.add_padding(scaledv, meta["pad_x"], meta["pad_y"])
        padded_blocku = MichiganPIV.add_padding(blockedu, meta["pad_x"], meta["pad_y"])
        padded_blockv = MichiganPIV.add_padding(blockedv, meta["pad_x"], meta["pad_y"])
        padded_newgaps_mask = MichiganPIV.add_padding(
            new_mask, meta["pad_x"], meta["pad_y"]
        )
        padded_initial_mask = MichiganPIV.add_padding(
            initial_mask, meta["pad_x"], meta["pad_y"], pad_values=1
        )

        A = np.stack((paddedu, paddedv), axis=-1)  # Original snapshots
        B = np.stack((padded_blocku, padded_blockv), axis=-1)  # Gappy snapshots
        C = np.stack(
            (padded_newgaps_mask, padded_initial_mask), axis=-1
        )  # Save masks without entering training process

        if self.transform:
            A = self.transform(A)
            B = self.transform(B)
            C = self.transform(C)

        A = A.to(dtype=torch.float32)
        B = B.to(dtype=torch.float32)

        return A, B, C, scales

    def _load_data(self, idx):
        # Load the h5 slice, reshape and format.
        with h5py.File(self.data_path, "r") as f:
            grp = f[self.grp_dict[self.grp_cad_idx[0]]]  # Load test point
            dataset = grp[self.cad_dict[self.grp_cad_idx[1]]]  # Load crank angle

            # Get scales
            scales = {}
            for attr in dataset.attrs:
                scales[attr] = dataset.attrs[attr]

            snap = dataset[idx, ...]  # Load one snapshot
            u = snap[..., 2]
            v = snap[..., 3]

            gridu = u.reshape(
                self.metadata["img_shape"][0], self.metadata["img_shape"][1]
            )
            gridv = v.reshape(
                self.metadata["img_shape"][0], self.metadata["img_shape"][1]
            )

            # Binary mask to locate the bounds of the unmodified data (0 indicates data, 1 indicates gap)
            initial_mask = np.zeros((gridu.shape[0], gridu.shape[1]), dtype=np.uint8)
            initial_mask[(gridu == 0) | (gridv == 0)] = 1

        return gridu, gridv, initial_mask, scales

    @staticmethod
    def create_target_grid(x_values, y_values, new_shape):
        nx, ny = len(x_values), len(y_values)
        target_nx, target_ny = new_shape

        add_nx = target_nx - nx
        add_ny = target_ny - ny

        add_nx_l = add_nx // 2
        add_nx_r = add_nx - add_nx_l
        add_ny_t = add_ny // 2
        add_ny_b = add_ny - add_ny_t

        dx = np.round(np.mean(np.diff(x_values)), 4)
        dy = np.round(np.mean(np.diff(y_values)), 4)

        new_xs_l = sorted(np.round(x_values[0] - np.arange(1, add_nx_l + 1) * dx, 4))
        new_xs_r = np.round(x_values[-1] + np.arange(1, add_nx_r + 1) * dx, 4)
        new_ys_t = sorted(np.round(y_values[0] - np.arange(1, add_ny_t + 1) * dy, 4))
        new_ys_b = np.round(y_values[-1] + np.arange(1, add_ny_b + 1) * dy, 4)

        new_xs = np.concatenate((new_xs_l, x_values, new_xs_r))
        new_ys = np.concatenate((new_ys_t, y_values, new_ys_b))
        gridx, gridy = np.meshgrid(new_xs, new_ys)

        pad_x = ((0, 0), (add_nx_l, add_nx_r))
        pad_y = ((add_ny_t, add_ny_b), (0, 0))

        return gridx, gridy, pad_x, pad_y

    @staticmethod
    def add_padding(matrix, pad_x, pad_y, pad_values=0):
        matrix = np.pad(matrix, pad_x, mode="constant", constant_values=pad_values)
        matrix = np.pad(matrix, pad_y, mode="constant", constant_values=pad_values)
        return matrix
