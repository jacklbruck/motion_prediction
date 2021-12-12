import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
import logging
import random
import torch
import sys
import os

sys.path.append("..")

from fairmotion.tasks.motion_prediction import test
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
    utils.log_config(args.save_model_path, args)

    logging.info("Preparing dataset...")
    dataset, mean, std = utils.prepare_dataset(
        *[
            os.path.join(args.preprocessed_path, f"{split}.pt")
            for split in ["train", "test", "val"]
        ],
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

    train_losses, val_losses = [], []
    iterator = tqdm(range(args.epochs))
    for epoch in iterator:
        if iterator.postfix is not None:
            cur_postfix = dict([tuple(s.split("=")) for s in iterator.postfix.split(", ")])
        else:
            cur_postfix = dict()
        cur_postfix.update({"Training Loss": train_loss, "Validation Loss": val_loss})

        # Run training epoch.
        model, opt, train_loss = loops.train(
            model,
            criterion,
            opt,
            dataset,
            args.batch_size,
            max(0, 1 - 2 * epoch / args.epochs),
            args.architecture,
            iterator=iterator
        )

        # Get validation loss.
        val_loss = loops.eval(model, criterion, dataset, args.batch_size, iterator=iterator)
        opt.epoch_step(val_loss=val_loss)

        # Add losses.
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # if epoch % args.save_model_frequency == 0:
        #     _, rep = os.path.split(args.preprocessed_path.strip("/"))
        #     _, mae = test.test_model(
        #         model=model,
        #         dataset=dataset["validation"],
        #         rep=rep,
        #         device=device,
        #         mean=mean,
        #         std=std,
        #         max_len=L,
        #     )
        #     logging.info(f"Validation MAE: {mae}")
        #     torch.save(model.state_dict(), f"{args.save_model_path}/{epoch}.model")
        #     if len(tracker["val_losses"]) == 0 or val_loss <= min(tracker["val_losses"]):
        #         torch.save(model.state_dict(), f"{args.save_model_path}/best.model")

    return train_losses, val_losses


def plot_curves(args, training_losses, val_losses):
    plt.plot(range(len(training_losses)), training_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"{args.save_model_path}/loss.svg", format="svg")


def main(args):
    train_losses, val_losses = train(args)

    plot_curves(args, train_losses, val_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequence to sequence motion prediction training."
    )
    parser.add_argument(
        "--preprocessed-path",
        type=str,
        help="Path to folder with pickled " "files",
        required=True,
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
        "--save-model-path",
        type=str,
        help="Path to store saved models",
        required=True,
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
        default="seq2seq",
        choices=[
            "seq2seq",
            "tied_seq2seq",
            "transformer",
            "transformer_encoder",
            "rnn",
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
