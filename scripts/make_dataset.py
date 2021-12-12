import numpy as np
import argparse
import logging
import pickle
import torch
import os

from fairmotion.data import amass_dip
from fairmotion.ops import motion as motion_ops
from fairmotion.tasks.motion_prediction import utils
from fairmotion.utils import utils as fairmotion_utils


logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def split_into_windows(motion, window_size, stride):
    """
    Split motion object into list of motions with length window_size with
    the given stride.
    """
    n_windows = (motion.num_frames() - window_size) // stride + 1
    motion_ws = [
        motion_ops.cut(motion, start, start + window_size)
        for start in stride * np.arange(n_windows)
    ]
    return motion_ws


def process_file(ftuple, create_windows, convert_fn, lengths):
    src_len, tgt_len = lengths
    filepath, file_id = ftuple
    motion = amass_dip.load(filepath)
    motion.name = file_id

    if create_windows is not None:
        window_size, window_stride = create_windows
        if motion.num_frames() < window_size:
            return [], []
        matrices = [
            convert_fn(motion.rotations())
            for motion in split_into_windows(
                motion, window_size, window_stride
            )
        ]
    else:
        matrices = [convert_fn(motion.rotations())]

    return (
        [matrix[:src_len, ...].reshape((src_len, -1)) for matrix in matrices],
        [
            matrix[src_len : src_len + tgt_len, ...].reshape((tgt_len, -1))
            for matrix in matrices
        ],
    )


def process_split(
    all_fnames, output_path, rep, src_len, tgt_len, create_windows=None,
):
    """
    Process data into numpy arrays.

    Args:
        all_fnames: List of filenames that should be processed.
        output_path: Where to store numpy files.
        rep: If the output data should be rotation matrices, quaternions or
            axis angle.
        create_windows: Tuple (size, stride) of windows that should be
            extracted from each sequence or None otherwise.

    Returns:
        Some meta statistics (how many sequences processed etc.).
    """
    assert rep in ["aa", "rotmat", "quat"]
    convert_fn = utils.convert_fn_from_R(rep)

    data = fairmotion_utils.run_parallel(
        process_file,
        all_fnames,
        num_cpus=40,
        create_windows=create_windows,
        convert_fn=convert_fn,
        lengths=(src_len, tgt_len),
    )
    src_seqs, tgt_seqs = [], []
    for worker_data in data:
        s, t = worker_data
        src_seqs.extend(s)
        tgt_seqs.extend(t)
    logging.info(f"Processed {len(src_seqs)} sequences")
    pickle.dump((src_seqs, tgt_seqs), open(output_path, "wb"))


def read_content(filepath):
    content = []
    with open(filepath) as f:
        for line in f:
            content.append(line.strip())
    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rep",
        type=str,
        help="Angle representation to convert data to",
        choices=["aa", "quat", "rotmat"],
        default="aa",
    )
    parser.add_argument(
        "--src-len",
        type=int,
        default=120,
        help="Number of frames fed as input motion to the model",
    )
    parser.add_argument(
        "--tgt-len",
        type=int,
        default=24,
        help="Number of frames to be predicted by the model",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=180,
        help="Window size for test and validation, in frames.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=120,
        help="Window stride for test and validation, in frames. This is also"
        " used as training window size",
    )

    args = parser.parse_args()

    train_files = read_content("../motion_prediction/data/training_fnames.txt")
    validation_files = read_content("../motion_prediction/data/validation_fnames.txt")
    test_files = read_content("../motion_prediction/data/test_fnames.txt")

    train_ftuples = []
    test_ftuples = []
    validation_ftuples = []
    for filepath in fairmotion_utils.files_in_dir("../data/raw", ext="pkl"):
        db_name = os.path.split(os.path.dirname(filepath))[1]
        db_name = (
            "_".join(db_name.split("_")[1:])
            if "AMASS" in db_name
            else db_name.split("_")[0]
        )
        f = os.path.basename(filepath)
        file_id = "{}/{}".format(db_name, f)

        if file_id in train_files:
            train_ftuples.append((filepath, file_id))
        elif file_id in validation_files:
            validation_ftuples.append((filepath, file_id))
        elif file_id in test_files:
            test_ftuples.append((filepath, file_id))
        else:
            pass

    logging.info("Processing training data...")
    train_dataset = process_split(
        train_ftuples,
        "../data/interim/train.pkl",
        rep=args.rep,
        src_len=args.src_len,
        tgt_len=args.tgt_len,
        create_windows=(args.window_size, args.window_stride),
    )
    with open("../data/interim/train.pkl", "rb") as f:
        src, tgt = pickle.load(f)

        torch.save((torch.Tensor(np.array(src)), torch.Tensor(np.array(tgt))), "../data/proc/train.pt")

    logging.info("Processing validation data...")
    process_split(
        validation_ftuples,
        "../data/interim/val.pkl",
        rep=args.rep,
        src_len=args.src_len,
        tgt_len=args.tgt_len,
        create_windows=(args.window_size, args.window_stride),
    )
    with open("../data/interim/val.pkl", "rb") as f:
        src, tgt = pickle.load(f)

        torch.save((torch.Tensor(np.array(src)), torch.Tensor(np.array(tgt))), "../data/proc/val.pt")

    logging.info("Processing test data...")
    process_split(
        test_ftuples,
        "../data/interim/test.pkl",
        rep=args.rep,
        src_len=args.src_len,
        tgt_len=args.tgt_len,
        create_windows=(args.window_size, args.window_stride),
    )
    with open("../data/interim/test.pkl", "rb") as f:
        src, tgt = pickle.load(f)

        torch.save((torch.Tensor(np.array(src)), torch.Tensor(np.array(tgt))), "../data/proc/test.pt")
