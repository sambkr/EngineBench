import numpy as np
import json
import os
import torch
import argparse
import logging
import yaml

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.evaluation import calc_metrics_def
from utils.data_loading import MichiganPIV, SheetGapHandler, DefaultHandler, custom_collate_fn
from models.unet import build_unet
from models.unetr import build_unetr
from models.gan import build_gan


def parse_args():
    parser = argparse.ArgumentParser(description="Set hyperparameters")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()
    return args


def setup_logging(output_dir):
    logging.basicConfig(
        filename=os.path.join(output_dir, "evaluation.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main():
    args = parse_args()

    # Load config file
    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise Exception(f"Configuration file {args.config} not found.")

    case_name = f"{config['model']}_{config['lossfn']}_{int(config['gapsize']*100)}_{config['perm']}"
    output_dir = os.path.join(config["outputpath"], case_name)
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(output_dir)
    logging.info(f"Case name: {case_name}")
    logging.info(f"Output directory: {output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")

    grp_dict = {0: "r1300_p40", 1: "r1300_p95", 2: "r0800_p95"}  # Select test point

    cad_dict = {
        0: "cad090",
        1: "cad135",
        2: "cad180",
        3: "cad225",
        4: "cad270",
    }  # Select crank angle

    tr_val_te_order_dict = {
        "A": [0, 1, 3, 4, 2],
        "B": [1, 3, 4, 0, 2],
        "C": [0, 3, 4, 1, 2],
        "D": [0, 1, 4, 3, 2],
    }  # Select permutation

    # Select handler
    if config["gaptype"] == 'edge':
        gap_handler = SheetGapHandler(
            seed=None, max_removal_fraction=config["gapsize"], central_sheet=False
        )
    elif config["gaptype"] == 'default':
        gap_handler = DefaultHandler()
        print(f'Gap type {config["gaptype"]}')
    else:
        raise Exception(
            f"Gap type {config['gaptype']} not implemented. Try 'default' or 'edge'."
        )

    tens_transform = transforms.ToTensor()

    # Load datasets
    datasets = {}
    for idx, i in enumerate(tr_val_te_order_dict[config["perm"]]):
        grp_cad_idx = (0, i)
        datasets[f"ds{idx+1}"] = MichiganPIV(
            data_path=config["filepath"],
            grp_dict=grp_dict,
            cad_dict=cad_dict,
            grp_cad_idx=grp_cad_idx,
            gap_handler=gap_handler,
            target_img_shape=config["target_imgshape"],
            train=True,
            transform=tens_transform,
        )
        logging.info(f'ds{idx+1} length: {len(datasets[f"ds{idx+1}"])}')

    test_ds = datasets["ds5"]
    logging.info(f"\nDataset lengths - Test: {len(test_ds)} \n")

    BATCH_SIZE = config["batchsize"]
    NUM_WORKERS = config["workers"]
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    nChannels = len(test_ds[0][0])

    # Load model
    if config["model"] == "unet":
        model = build_unet(nChannels, device)
        model.load_state_dict(torch.load(os.path.join(output_dir, f"{case_name}.pth")))
    elif config["model"] == "UNETR":
        model = build_unetr(nChannels, device, config["target_imgshape"])
        model.load_state_dict(torch.load(os.path.join(output_dir, f"{case_name}.pth")))
    elif config["model"] == "gan":

        class GANOpts:
            def __init__(self):
                self.ngf = 64  # N generator filters
                self.ndf = 64  # N discriminator filters
                self.nc = 2  # N channels in first conv layer
                self.Ddrop = 0.5  # Discriminator dropout
                self.nBottleneck = 4000  # Dim of encoder bottleneck
                self.nef = 64  # N encoder filters
                self.ngpu = torch.cuda.device_count()
                self.wtl2 = 0.99  # L2 error weight

            def to_dict(self):
                return vars(self)

        opts = GANOpts()
        _, model = build_gan(opts)
        save_state = torch.load(
            os.path.join(output_dir, f"{case_name}.pth"),
            map_location=torch.device(device),
        )
        model.load_state_dict(save_state["netG_state_dict"])
    else:
        raise Exception(
            f"Model {config['model']} not implemented. Try 'unet', 'UNETR' or 'gan'."
        )

    model.to(device)
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model.eval()

    # Centre and edge metrics
    metrics_c = np.empty(shape=(len(test_ds), 3))  # RI, MI, L2
    metrics_e = np.empty(shape=(len(test_ds), 3))  # RI, MI, L2

    spectral_energy = torch.empty(
        size=(
            len(test_ds),
            4,
            nChannels,
            config["target_imgshape"][0],
            config["target_imgshape"][1],
        )
    )  # (batch, idx, channel, height, width). idx: tr_c, pr_d, tr_e, pr_e

    count = 0
    logging.info("Calculating metrics:")

    with torch.no_grad():
        for test_snaps, _, test_masks, test_scales in tqdm(test_loader, total=len(test_loader)):
            test_snaps = test_snaps.to(device)

            test_masks = test_masks.bool()

            # Fit the masks to the batch length
            new_gaps, old_gaps = torch.unsqueeze(test_masks[:,0],dim=1), torch.unsqueeze(test_masks[:,1],dim=1)
            new_gaps, old_gaps = torch.cat([new_gaps, new_gaps],dim=1), torch.cat([old_gaps, old_gaps],dim=1) # shape 64,2,128,128
            batch_len = len(test_snaps)
            true_locations = (new_gaps == 0) & (old_gaps == 0) # pixel locations of true data

            test_gaps = torch.clone(test_snaps)
            test_gaps[true_locations == False] = 0

            recons = model(test_gaps)

            for _, (
                test_snap,
                test_recon,
                test_scale,
                true_mask,
                pred_mask,
            ) in enumerate(
                zip(
                    test_snaps.clone(), recons.clone(), test_scales, true_locations, new_gaps
                )
            ):
                # Rescale the snapshots
                test_snap[0] = test_snap[0] * test_scale["std_u"] + test_scale["mean_u"]
                test_snap[1] = test_snap[1] * test_scale["std_v"] + test_scale["mean_v"]
                test_recon[0] = (
                    test_recon[0] * test_scale["std_u"] + test_scale["mean_u"]
                )
                test_recon[1] = (
                    test_recon[1] * test_scale["std_v"] + test_scale["mean_v"]
                )

                # Calculate and store metrics
                m_c, m_e, energy = calc_metrics_def(
                    test_snap,
                    test_recon,
                    true_mask, # Locations of true data
                    pred_mask, # Locations of predicted data
                    nChannels,
                    spectral_energy,
                )
                metrics_c[count] = m_c
                metrics_e[count] = m_e
                spectral_energy[count] = energy

                count += 1

    avRIc = np.mean(metrics_c[:, 0])
    avMIc = np.mean(metrics_c[:, 1])
    avL2c = np.mean(metrics_c[:, 2])

    avRIe = np.mean(metrics_e[:, 0])
    avMIe = np.mean(metrics_e[:, 1])
    avL2e = np.mean(metrics_e[:, 2])

    logging.info(f"Mean RI centre: {avRIc:.3f}")
    logging.info(f"Mean MI centre: {avMIc:.3f}")
    logging.info(f"Mean L2 centre: {avL2c:.3f}")

    logging.info(f"Mean RI edge: {avRIe:.3f}")
    logging.info(f"Mean MI edge: {avMIe:.3f}")
    logging.info(f"Mean L2 edge: {avL2e:.3f}")

    av_metrics = {
        "avRIc": avRIc,
        "avMIc": avMIc,
        "avL2c": avL2c,
        "avRIe": avRIe,
        "avMIe": avMIe,
        "avL2e": avL2e,
    }

    np.save(os.path.join(output_dir, "metrics_c.npy"), metrics_c)
    np.save(os.path.join(output_dir, "metrics_e.npy"), metrics_e)

    with open(os.path.join(output_dir, "av_metrics.json"), "w") as f:
        json.dump(av_metrics, f)

    torch.save(
        spectral_energy,
        os.path.join(output_dir, "spectral_energy" + config["perm"] + ".pt"),
    )

    logging.info("Evaluation complete")


if __name__ == "__main__":
    main()
