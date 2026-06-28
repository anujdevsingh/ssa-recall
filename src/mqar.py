"""MQAR — Multi-Query Associative Recall (toy synthetic).

A sequence has a context of (key, value) pairs, then the keys reappear as queries.
The model must output, at each query position, the value that was bound to that key
in the context. Recall-weak models (fixed state) fail as the number of pairs grows.

Token layout:
  0                      : NOISE / filler
  1 .. num_keys          : key tokens
  num_keys+1 .. +num_values : value tokens
Labels are -100 (ignored) everywhere except query positions, where the label is the value.
"""
import torch

NOISE = 0


def generate_mqar(num_examples, seq_len, num_kv_pairs,
                  num_keys=64, num_values=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    # need 2N for the context pairs + N positions for the queries
    assert seq_len >= 3 * num_kv_pairs, "seq_len too short for this many kv pairs"
    assert num_kv_pairs <= num_keys, "not enough distinct keys"
    vocab_size = num_keys + num_values + 1

    inputs = torch.full((num_examples, seq_len), NOISE, dtype=torch.long)
    labels = torch.full((num_examples, seq_len), -100, dtype=torch.long)

    for b in range(num_examples):
        keys = torch.randperm(num_keys, generator=g)[:num_kv_pairs] + 1
        values = torch.randint(0, num_values, (num_kv_pairs,), generator=g) + num_keys + 1

        # context: k0 v0 k1 v1 ...
        ctx = torch.empty(2 * num_kv_pairs, dtype=torch.long)
        ctx[0::2] = keys
        ctx[1::2] = values
        inputs[b, :2 * num_kv_pairs] = ctx

        # queries: drop each key (with its value as label) at random later positions
        qstart = 2 * num_kv_pairs
        pos = torch.randperm(seq_len - qstart, generator=g)[:num_kv_pairs] + qstart
        inputs[b, pos] = keys
        labels[b, pos] = values

    return inputs, labels, vocab_size


def _self_check():
    """Each query position's label must equal the value bound to that key in context."""
    X, Y, V = generate_mqar(num_examples=200, seq_len=128, num_kv_pairs=16, seed=1)
    for b in range(X.shape[0]):
        # rebuild the context key->value map
        ctx_keys = X[b, 0:32:2]
        ctx_vals = X[b, 1:32:2]
        kv = {int(k): int(v) for k, v in zip(ctx_keys, ctx_vals)}
        qpos = (Y[b] != -100).nonzero().flatten()
        assert len(qpos) == 16, "wrong number of query positions"
        for p in qpos:
            key_at_p = int(X[b, p])
            assert kv[key_at_p] == int(Y[b, p]), "label does not match bound value"
    print("mqar self-check OK:", X.shape, "vocab", V)


if __name__ == "__main__":
    _self_check()
