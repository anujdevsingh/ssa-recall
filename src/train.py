"""Train one (mixer, num_kv_pairs) config and return recall accuracy on held-out data."""
import math
import torch
import torch.nn.functional as F

from .mqar import generate_mqar
from .models import SeqModel


def _batches(X, Y, bs, generator):
    idx = torch.randperm(X.shape[0], generator=generator)
    for i in range(0, X.shape[0], bs):
        j = idx[i:i + bs]
        yield X[j], Y[j]


@torch.no_grad()
def recall_accuracy(model, X, Y, device):
    model.eval()
    correct = total = 0
    for i in range(0, X.shape[0], 256):
        xb, yb = X[i:i + 256].to(device), Y[i:i + 256].to(device)
        pred = model(xb).argmax(-1)
        m = yb != -100
        correct += (pred[m] == yb[m]).sum().item()
        total += m.sum().item()
    return correct / max(total, 1)


def train_one(mixer, num_kv_pairs, seq_len=256, steps=4000, bs=64,
              n_train=16000, n_test=1000, d=256, layers=2, heads=8,
              lr=1e-3, seed=0, device=None, log_every=0, target_acc=0.999):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Ytr, vocab = generate_mqar(n_train, seq_len, num_kv_pairs, seed=seed)
    Xte, Yte, _ = generate_mqar(n_test, seq_len, num_kv_pairs, seed=seed + 999)

    model = SeqModel(vocab, mixer=mixer, d=d, heads=heads, layers=layers,
                     max_len=seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # warmup + cosine decay: transformers need warmup to form induction heads
    warmup = max(1, int(0.1 * steps))

    def lr_scale(s):
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    gen = torch.Generator().manual_seed(seed)

    step = 0
    model.train()
    while step < steps:
        for xb, yb in _batches(Xtr, Ytr, bs, gen):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   yb.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if log_every and step % log_every == 0:
                acc = recall_accuracy(model, Xte, Yte, device)
                model.train()
                print(f"    [{mixer} N={num_kv_pairs}] step {step:5d} "
                      f"loss {loss.item():.3f} acc {acc:.3f}")
                if acc >= target_acc:  # solved — stop early, save the compute budget
                    return acc
            if step >= steps:
                break

    return recall_accuracy(model, Xte, Yte, device)
