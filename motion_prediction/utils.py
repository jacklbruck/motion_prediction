# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import os
import torch
from functools import partial
from multiprocessing import Pool

from fairmotion.models import (
    decoders,
    encoders,
    optimizer,
    rnn,
    seq2seq,
    transformer,
)
from fairmotion.utils import constants
from fairmotion.ops import conversions
from .dataset import get_loader
from .models import gnn, grnn


def apply_ops(input, ops):
    """
    Apply series of operations in order on input. `ops` is a list of methods
    that takes single argument as input (single argument functions, partial
    functions). The methods are called in the same order provided.
    """
    output = input
    for op in ops:
        output = op(output)
    return output


def unflatten_angles(arr, rep):
    """
    Unflatten from (batch_size, num_frames, num_joints*ndim) to
    (batch_size, num_frames, num_joints, ndim) for each angle format
    """
    if rep == "aa":
        return arr.reshape(arr.shape[:-1] + (-1, 3))
    elif rep == "quat":
        return arr.reshape(arr.shape[:-1] + (-1, 4))
    elif rep == "rotmat":
        return arr.reshape(arr.shape[:-1] + (-1, 3, 3))


def flatten_angles(arr, rep):
    """
    Unflatten from (batch_size, num_frames, num_joints, ndim) to
    (batch_size, num_frames, num_joints*ndim) for each angle format
    """
    if rep == "aa":
        return arr.reshape(arr.shape[:-2] + (-1))
    elif rep == "quat":
        return arr.reshape(arr.shape[:-2] + (-1))
    elif rep == "rotmat":
        # original dimension is (batch_size, num_frames, num_joints, 3, 3)
        return arr.reshape(arr.shape[:-3] + (-1))


def multiprocess_convert(arr, convert_fn):
    pool = Pool(40)
    result = list(pool.map(convert_fn, arr))
    return result


def convert_fn_to_R(rep):
    ops = [partial(unflatten_angles, rep=rep)]
    if rep == "aa":
        ops.append(partial(multiprocess_convert, convert_fn=conversions.A2R))
    elif rep == "quat":
        ops.append(partial(multiprocess_convert, convert_fn=conversions.Q2R))
    elif rep == "rotmat":
        ops.append(lambda x: x)
    ops.append(np.array)
    return ops


def identity(x):
    return x


def convert_fn_from_R(rep):
    if rep == "aa":
        convert_fn = conversions.R2A
    elif rep == "quat":
        convert_fn = conversions.R2Q
    elif rep == "rotmat":
        convert_fn = identity
    return convert_fn


def unnormalize(arr, mean, std):
    return arr * (std + constants.EPSILON) + mean


def prepare_dataset(
    train_path,
    valid_path,
    test_path,
    batch_size,
    device,
    shuffle=False,
):
    dataset = {}
    for split, split_path in zip(
        ["train", "test", "validation"], [train_path, valid_path, test_path]
    ):
        mean, std = None, None
        if split in ["test", "validation"]:
            mean = dataset["train"].dataset.mean
            std = dataset["train"].dataset.std

        dataset[split] = get_loader(
            split_path,
            batch_size,
            device,
            mean,
            std,
            shuffle,
        )
    return dataset, mean, std


def prepare_model(input_dim, hidden_dim, device, num_layers=1, architecture="seq2seq"):
    if architecture == "seq2seq":
        enc = encoders.LSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim).to(
            device
        )
        dec = decoders.LSTMDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            device=device,
        ).to(device)
        model = seq2seq.Seq2Seq(enc, dec)

    elif architecture == "transformer_encoder":
        model = transformer.TransformerLSTMModel(
            input_dim,
            hidden_dim,
            4,
            hidden_dim,
            num_layers,
        )

    elif architecture == "transformer":
        model = transformer.TransformerModel(
            input_dim,
            hidden_dim,
            4,
            hidden_dim,
            num_layers,
        )

    elif architecture == "gnn":
        enc = gnn.GraphEncoder(
            input_dim=input_dim // gnn.SMPL_NR_JOINTS,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device,
        ).to(device)
        dec = gnn.GraphDecoder(
            hidden_dim=hidden_dim,
            output_dim=input_dim // gnn.SMPL_NR_JOINTS,
            num_layers=num_layers,
            device=device,
            lstm=enc.lstm
        ).to(device)

        model = gnn.GraphModel(enc, dec)

    elif architecture == "grnn":
        model = grnn.Model(
            input_dim=input_dim // gnn.SMPL_NR_JOINTS,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device
        )

    # Send model to proper device.
    model = model.to(device)

    # Zero gradients and initialize.
    model.zero_grad()
    model.init_weights()

    return model


def log_config(path, args):
    with open(os.path.join(path, "config.txt"), "w") as f:
        for key, value in args._get_kwargs():
            f.write(f"{key}:{value}\n")


def prepare_optimizer(model, opt: str, lr=None):
    kwargs = {}
    if lr is not None:
        kwargs["lr"] = lr

    if opt == "sgd":
        return optimizer.SGDOpt(model, **kwargs)
    elif opt == "adam":
        return optimizer.AdamOpt(model, **kwargs)
    elif opt == "noamopt":
        return optimizer.NoamOpt(model)


def prepare_tgt_seqs(architecture, src_seqs, tgt_seqs):
    if architecture == "st_transformer" or architecture == "rnn":
        return torch.cat((src_seqs[:, 1:], tgt_seqs), axis=1)
    elif architecture == "gat":
        return torch.cat((src_seqs[:, 2 ** 4 + 1:], tgt_seqs), axis=1).detach()
    else:
        return tgt_seqs
