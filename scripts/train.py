import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
import logging
import random
import torch
import json
import sys
import os

sys.path.append("..")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fairmotion.tasks.motion_prediction.test import test_model
from motion_prediction import utils, loops
from tqdm import tqdm

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def set_seeds():
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(args):
    return (
        args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    )


def train(args):
    # Choose device and random seed.
    device = get_device(args)
    set_seeds()

    # Create model directory and save config.
    if not os.path.exists(f"../models/{args.architecture}"):
        os.mkdir(f"../models/{args.architecture}")
    utils.log_config(f"../models/{args.architecture}", args)

    logging.info("Preparing dataset...")
    dataset, mean, std = utils.prepare_dataset(
        "../data/proc/train.pt",
        "../data/proc/test.pt",
        "../data/proc/val.pt",
        batch_size=args.batch_size,
        device=device,
        shuffle=args.shuffle,
    )

    logging.info("Preparing model...")
    criterion = nn.MSELoss()
    model = utils.prepare_model(
        input_dim=next(iter(dataset["train"]))[1].size(2),
        hidden_dim=args.hidden_dim,
        device=device,
        num_layers=args.num_layers,
        architecture=args.architecture,
    )

    logging.info("Running initialization loops...")
    train_loss = loops.init(model, criterion, dataset, args.batch_size)
    val_loss = loops.eval(model, criterion, dataset, args.batch_size)

    logging.info("Training model...")
    opt = utils.prepare_optimizer(model, args.optimizer, args.lr)

    losses = {"train": [], "val": []}

    iterator = tqdm(range(args.epochs))
    for epoch in iterator:
        if iterator.postfix is not None:
            cur_postfix = dict(
                [tuple(s.split("=")) for s in iterator.postfix.split(", ")]
            )
        else:
            cur_postfix = dict()
        cur_postfix.update({"Training Loss": train_loss, "Validation Loss": val_loss})
        iterator.set_postfix(cur_postfix)

        # Run training epoch.
        model, opt, train_loss = loops.train(
            model,
            criterion,
            opt,
            dataset,
            args.batch_size,
            max(0, 1 - 2 * epoch / args.epochs),
            args.architecture,
            iterator=iterator,
        )

        # Get validation loss.
        val_loss = loops.eval(
            model, criterion, dataset, args.batch_size, iterator=iterator
        )
        opt.epoch_step(val_loss=val_loss)

        # Add losses.
        losses["train"].append(train_loss)
        losses["val"].append(val_loss)

        if val_loss == min(losses["val"]):
            torch.save(model.state_dict(), f"../models/{args.architecture}/best.model")

    # Get test results.
    maes = {k: {} for k in dataset.keys()}

    for k in dataset.keys():
        maes[k] = test_model(
            model=model,
            dataset=dataset[k],
            rep="aa",
            device=device,
            mean=mean.numpy(),
            std=std.numpy(),
        )[1]

    return losses


def plot_loss_curves(args, losses):
    for k, v in losses.items():
        plt.plot(range(len(v)), v, label=k)

    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.savefig(f"../models/{args.architecture}/loss.png", format="png")


def main(args):
    losses, maes = train(args)

    # Plot loss curves.
    plot_loss_curves(args, losses)

    # Save ending error metrics.
    with open(f"../models/{args.architecture}/maes.json", "w+") as f:
        json.dump(maes, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequence to sequence motion prediction training."
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for training", default=64
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Use this option to enable shuffling",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="Hidden size of LSTM units in encoder/decoder",
        default=1024,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers of LSTM/Transformer in encoder/decoder",
        default=1,
    )
    parser.add_argument(
        "--save-model-frequency",
        type=int,
        help="Frequency (in terms of number of epochs) at which model is " "saved",
        default=5,
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=200
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Training device",
        default=None,
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="Seq2Seq archtiecture to be used",
        default=None,
        choices=[
            "seq2seq",
            "transformer",
            "transformer_encoder",
        ],
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
        default=None,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Torch optimizer",
        default="sgd",
        choices=["adam", "sgd", "noamopt"],
    )
    args = parser.parse_args()
    main(args)
