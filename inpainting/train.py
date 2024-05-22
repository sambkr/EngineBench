import argparse
import json
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from models.unet import build_unet
from models.unetr import build_unetr
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import EarlyStopping
from utils.data_loading import (
    CustomConcatDataset,
    MichiganPIV,
    SheetGapHandler,
    custom_collate_fn,
)


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
        filename=os.path.join(output_dir, "training.log"),
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

    # Setup this case
    case_name = f"{config['model']}_{config['lossfn']}_{int(config['gapsize']*100)}_{config['perm']}"
    output_dir = os.path.join(config["outputpath"], case_name)
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(output_dir)
    logging.info(f"Case name: {case_name}")
    logging.info(f"Output directory: {output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")

    # Dictionaries for different groups (test points), cads (crank angles), and train/val/test congifurations
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

    # Load datasets
    gap_handler = SheetGapHandler(
        seed=None, max_removal_fraction=config["gapsize"], central_sheet=False
    )
    tens_transform = transforms.ToTensor()

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

    # Take first three datasets for the train loader
    train_ds = CustomConcatDataset([datasets["ds1"], datasets["ds2"], datasets["ds3"]])
    val_ds = datasets["ds4"]
    logging.info(
        f"\nDataset lengths - Train: {len(train_ds)}; Validation: {len(val_ds)} \n"
    )

    BATCH_SIZE = config["batchsize"]
    NUM_WORKERS = config["workers"]
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    nChannels = len(val_ds[0][0])

    # Set up model
    if config["model"] == "unet":
        model = build_unet(nChannels, device)
    elif config["model"] == "UNETR":
        model = build_unetr(nChannels, device, config["target_imgshape"])
    else:
        raise Exception(
            f"Model {config['model']} not implemented. Try 'unet' or 'UNETR'."
        )

    model.to(device)
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    if config["lossfn"] == "MSE":
        loss_fn = torch.nn.MSELoss()
    elif config["lossfn"] == "huber":
        loss_fn = torch.nn.HuberLoss(reduction="mean", delta=1)
    else:
        raise Exception(
            f"Loss function {config['lossfn']} not implemented. Try 'MSE' or 'huber'."
        )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=0.001
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    early_stopping = EarlyStopping(patience=1000, delta=1e-6)

    results_array = np.zeros(
        (config["epochs"], 4)
    )  # cols: epoch, LR, train loss, val loss

    start_time = time.time()
    for epoch in tqdm(range(config["epochs"]), total=config["epochs"]):
        losses_this_ep_train = 0
        losses_this_ep_val = 0

        model.train()
        for (snap, gap, _, _) in tqdm(
            train_loader, total=len(train_loader), leave=False
        ):
            snap, gap = snap.to(device), gap.to(device)
            recon = model(gap)
            loss = loss_fn(recon, snap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_this_ep_train += loss.cpu().detach().numpy()

        avg_train_loss = losses_this_ep_train / len(train_loader)
        results_array[epoch, 2] = avg_train_loss

        model.eval()
        with torch.no_grad():
            for (snap, gap, _, _) in tqdm(
                val_loader, total=len(val_loader), leave=False
            ):
                snap, gap = snap.to(device), gap.to(device)
                recon = model(gap)
                loss = loss_fn(recon, snap)
                losses_this_ep_val += loss.cpu().detach().numpy()

            avg_val_loss = losses_this_ep_val / len(val_loader)
            results_array[epoch, 3] = avg_val_loss

        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]

        results_array[epoch, 0] = epoch + 1
        results_array[epoch, 1] = current_lr

        logging.info(
            f"Epoch: {epoch+1} | train_loss: {avg_train_loss:.6f} | val_loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}"
        )
        np.save(os.path.join(output_dir, "training_results.npy"), results_array)

        if (epoch + 1) % config["ckpt_freq"] == 0:
            np.save(
                os.path.join(output_dir, f"training_results_ep{epoch+1}.npy"),
                results_array,
            )
            if isinstance(model, torch.nn.DataParallel):
                torch.save(
                    model.module.state_dict(),
                    os.path.join(output_dir, f"{case_name}_ep{epoch+1}.pth"),
                )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{case_name}_ep{epoch+1}.pth"),
                )

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered")
            break

        scheduler.step()

    total_time = time.time() - start_time
    duration_minutes = total_time / 60
    logging.info(f"Total training time: {duration_minutes:.2f} minutes")

    model_filename = os.path.join(output_dir, case_name + ".pth")
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), model_filename)
    else:
        torch.save(model.state_dict(), model_filename)

    with open(os.path.join(output_dir, "mins.json"), "w") as f:
        json.dump(duration_minutes, f)

    # Plot train and val loss curves
    last_ep = epoch + 1
    _, axs = plt.subplots(1, 2)
    axs[0].plot(results_array[:last_ep, 2])
    axs[1].plot(results_array[:last_ep, 3])
    axs[0].title.set_text(f"Training Loss: {results_array[last_ep-1, 2]:.2e}")
    axs[1].title.set_text(f"Validation Loss: {results_array[last_ep-1, 3]:.2e}")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Epoch")
    axs[1].set_xlabel("Epoch")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "loss_curves.png"))
    plt.close()

    logging.info("Training complete")


if __name__ == "__main__":
    main()
