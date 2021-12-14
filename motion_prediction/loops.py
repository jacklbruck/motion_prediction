import torch

from .utils import prepare_tgt_seqs
from tqdm import tqdm


def init(model, criterion, dataset, batch_size):
    model.train()

    with torch.no_grad():
        loop_loss = 0

        iterator = tqdm(dataset["train"])
        for src, tgt in iterator:
            # Pass data through model and calculate loss.
            out = model(src, tgt, teacher_forcing_ratio=1.0)
            loss = criterion(out, tgt)

            loop_loss += loss.item()

    return loop_loss / (len(dataset["train"]) * batch_size)


def train(
    model, criterion, opt, dataset, batch_size, teacher_forcing_ratio, architecture, iterator=None
):
    model.train()
    loop_loss = 0
    n = len(dataset["train"])

    for i, (src, tgt) in enumerate(dataset["train"]):
        # Zero gradients.
        opt.optimizer.zero_grad()

        # Pass data through model and calculate loss.
        out = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = criterion(out, prepare_tgt_seqs(architecture, src, tgt))

        loop_loss += loss.item()

        # Backpropogate and update epoch loss.
        loss.backward()
        opt.step()

        # Update iterator.
        if iterator is not None:
            cur_postfix = dict([tuple(s.split("=")) for s in iterator.postfix.split(", ")])
            cur_postfix.update({"Epoch Loss": loop_loss / ((i + 1) * batch_size), "Epoch Progress": f"{i + 1}/{n}"})

            iterator.set_postfix(cur_postfix)

    return model, opt, loop_loss / (n * batch_size)


def eval(model, criterion, dataset, batch_size, iterator=None):
    model.eval()

    with torch.no_grad():
        loop_loss = 0
        n = len(dataset["validation"])

        for i, (src, tgt) in enumerate(dataset["validation"]):
            # Get seeds and generative parameters.
            seed_tgt = src[:, -1].unsqueeze(1)
            max_len = tgt.size(1)

            # Pass data through model and calculate loss.
            out = model(src, seed_tgt, max_len=max_len, teacher_forcing_ratio=0)
            loss = criterion(out, tgt)

            loop_loss += loss.item()

            # Update iterator.
            if iterator is not None:
                cur_postfix = dict([tuple(s.split("=")) for s in iterator.postfix.split(", ")])
                cur_postfix.update({"Epoch Loss": loop_loss / ((i + 1) * batch_size), "Epoch Progress": f"{i + 1}/{n}"})

                iterator.set_postfix(cur_postfix)

    return loop_loss / (n * batch_size)


def generate(model, src_seqs, max_len, device):
    """
    Generates output sequences for given input sequences by running forward
    pass through the given model
    """
    model.eval()
    with torch.no_grad():
        tgt_seqs = src_seqs[:, -1].unsqueeze(1)
        src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
        outputs = model(src_seqs, tgt_seqs, max_len=max_len, teacher_forcing_ratio=0)
        return outputs.double()
