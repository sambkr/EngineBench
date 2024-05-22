import argparse
import json
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from models.gan import build_gan
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

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
        default="configs/default_config_GAN.yaml",
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

    # Set up model
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

    if config["model"] == "gan":
        netD, netG = build_gan(opts)
    else:
        raise Exception(f"Model {config['model']} not implemented. Try 'gan'.")

    netG.to(device)
    netD.to(device)

    if opts.ngpu > 1:
        logging.info(f"Using {opts.ngpu} GPUs")
        netG = torch.nn.DataParallel(netG)
        netD = torch.nn.DataParallel(netD)

    criterion = torch.nn.BCELoss()
    criterionMSE = torch.nn.MSELoss()

    optimizerD = torch.optim.Adam(
        netD.parameters(), lr=(config["lr"]) / 2, betas=(0.5, 0.999), weight_decay=1e-3
    )
    optimizerG = torch.optim.Adam(
        netG.parameters(), lr=config["lr"], betas=(0.5, 0.999), weight_decay=1e-3
    )

    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=50, gamma=0.75)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=50, gamma=0.75)

    results_array = np.zeros(
        (config["epochs"], 5)
    )  # cols: epoch, LR, train loss D, train loss G, val loss G

    start_time = time.time()
    for epoch in tqdm(range(config["epochs"]), total=config["epochs"]):
        # Accumulators
        losses_this_ep_train_D = 0
        losses_this_ep_train_G = 0
        losses_this_ep_val_G = 0

        # Train
        netD.train()
        netG.train()
        for (snap, gap, masks, _) in tqdm(
            train_loader, total=len(train_loader), leave=False
        ):
            snap, gap, masks = snap.to(device), gap.to(device), masks.to(device)

            # Train on real data
            real_label = torch.ones(len(snap), device=device)
            real_label = real_label.unsqueeze(1)
            netD.zero_grad()

            output = netD(snap)

            errD_real = criterion(output, real_label)
            errD_real.backward()

            # Train on fake data
            fake_label = torch.zeros(len(snap), device=device)
            fake_label = fake_label.unsqueeze(1)

            fake = netG(gap)
            output = netD(fake.detach())

            errD_fake = criterion(output, fake_label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update generator
            netG.zero_grad()

            output = netD(fake)

            errG_D = criterion(
                output, real_label
            )  # want netD to make wrong classification
            errG_l2 = criterionMSE(fake, snap)
            errG = (1 - opts.wtl2) * errG_D + opts.wtl2 * errG_l2
            errG.backward()
            optimizerG.step()

            losses_this_ep_train_D += errD.item() / len(snap)
            losses_this_ep_train_G += errG.item() / len(snap)

        avg_train_loss_D = losses_this_ep_train_D / len(train_loader)
        avg_train_loss_G = losses_this_ep_train_G / len(train_loader)

        results_array[epoch, 2] = avg_train_loss_D
        results_array[epoch, 3] = avg_train_loss_G

        for param_group in optimizerG.param_groups:
            current_lr = param_group["lr"]

        # Validate
        netD.eval()
        netG.eval()
        with torch.no_grad():
            for (snap, gap, _, _) in tqdm(
                val_loader, total=len(val_loader), leave=False
            ):
                snap, gap = snap.to(device), gap.to(device)

                fake = netG(gap)
                output = netD(fake).detach()

                real_label = torch.ones(len(snap), device=device)
                real_label = real_label.unsqueeze(1)
                errG_D = criterion(
                    output, real_label
                )  # want netD to make wrong classification
                errG_l2 = criterionMSE(fake, snap)
                errG = (1 - opts.wtl2) * errG_D + opts.wtl2 * errG_l2

                losses_this_ep_val_G += errG.item() / len(snap)

        avg_val_loss_G = losses_this_ep_val_G / len(val_loader)
        results_array[epoch, 4] = avg_val_loss_G

        results_array[epoch, 0] = epoch + 1
        results_array[epoch, 1] = current_lr

        logging.info(
            f"Epoch: {epoch+1} | "
            f"train_loss_D: {avg_train_loss_D:.6f} | "
            f"train_loss_G: {avg_train_loss_G:.6f} | "
            f"val_loss_G: {avg_val_loss_G:.6f} | "
            f"LR: {current_lr:.6f} | "
            f"Last errG_D: {errG_D:.6f} |"
            f"Last errG_l2: {errG_l2:.6f}"
        )
        np.save(os.path.join(output_dir, "training_results.npy"), results_array)

        # Save checkpoint
        if (epoch + 1) % config["ckpt_freq"] == 0:
            checkpoint = {
                "epoch": epoch,
                "results_array": results_array,
                "optimizerG_state_dict": optimizerG.state_dict(),
                "optimizerD_state_dict": optimizerD.state_dict(),
            }

            if isinstance(
                netG, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
            ):
                checkpoint["netG_state_dict"] = netG.module.state_dict()
            else:
                checkpoint["netG_state_dict"] = netG.state_dict()

            if isinstance(
                netD, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
            ):
                checkpoint["netD_state_dict"] = netD.module.state_dict()
            else:
                checkpoint["netD_state_dict"] = netD.state_dict()

            torch.save(
                checkpoint, os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            )

        schedulerD.step()
        schedulerG.step()

    total_time = time.time() - start_time
    duration_minutes = total_time / 60
    logging.info(f"Total training time: {duration_minutes:.2f} minutes")

    save_state = {
        "optimizerG_state_dict": optimizerG.state_dict(),
        "optimizerD_state_dict": optimizerD.state_dict(),
    }

    if isinstance(
        netG, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    ):
        save_state["netG_state_dict"] = netG.module.state_dict()
    else:
        save_state["netG_state_dict"] = netG.state_dict()
    if isinstance(
        netD, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    ):
        save_state["netD_state_dict"] = netD.module.state_dict()
    else:
        save_state["netD_state_dict"] = netD.state_dict()
    torch.save(save_state, os.path.join(output_dir, case_name + f".pth"))

    with open(os.path.join(output_dir, "mins.json"), "w") as f:
        json.dump(duration_minutes, f)

    # Plot train and val loss curves
    last_ep = epoch + 1
    _, axs = plt.subplots(1, 2)
    axs[0].plot(results_array[:last_ep, 3])  # avg_train_loss_G
    axs[0].plot(results_array[:last_ep, 2])  # avg_train_loss_D
    axs[1].plot(results_array[:last_ep, 4])  # avg_val_loss_G
    axs[0].title.set_text(f"Training Loss: {results_array[last_ep-1,3]:.2e}")
    axs[1].title.set_text(f"Validaton Loss: {results_array[last_ep-1,4]:.2e}")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Epoch")
    axs[1].set_xlabel("Epoch")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "loss_curves.png"))
    plt.close()

    logging.info("Training complete")


if __name__ == "__main__":
    main()
