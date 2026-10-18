"""The nano battery on the CPU: the slider reading (the training centres at +1 / -1, a training phrase read left out, a planted mood
axis found and a noise space not), the uniform comparator's push (one side at a time) and its evaluated sets, the stacks and the
marks on synthetic gauge tables. The extractor's checks live in test_nano_extract.py (the nano trainer's environment). Seconds, no download."""
import pytest
import torch

from alephllm_diffusion.nano import comparator as NC
from alephllm_diffusion.nano import compare as CM
from alephllm_diffusion.nano import mood as NM

F64 = torch.float64


def _phrases(n_up=30, n_down=30, n_neu=4, h=4, hn=2):
    cls = ["up"] * n_up + ["down"] * n_down + ["neutral"] * n_neu + ["up"] * h + ["down"] * h + ["neutral"] * hn
    split = ["train"] * (n_up + n_down + n_neu) + ["heldout"] * (2 * h + hn)
    return cls, split, [f"p{i}" for i in range(len(cls))]


def _planted(d=64, sep=3.0, seed=0, k=16):
    """cheerful rows at +sep/2 on k features, gloomy at -sep/2, neutral at 0, unit noise on every feature (the reading standardizes
    per feature, so a planted axis spreads over several features, as a mood does in a model's state)."""
    g = torch.Generator().manual_seed(seed)
    cls, split, phr = _phrases()
    X = torch.randn(len(cls), d, generator=g, dtype=F64)
    for i, c in enumerate(cls):
        X[i, :k] += {"up": sep / 2, "down": -sep / 2, "neutral": 0.0}[c]
    return X, cls, split, phr


def test_the_training_centres_sit_at_plus_and_minus_one():
    X, cls, split, _ = _planted()
    a = NM.in_sample(X, cls, split)
    up = [i for i in range(len(cls)) if cls[i] == "up" and split[i] == "train"]
    dn = [i for i in range(len(cls)) if cls[i] == "down" and split[i] == "train"]
    assert abs(float(a[up].mean()) - 1) < 1e-9 and abs(float(a[dn].mean()) + 1) < 1e-9


def test_a_training_phrase_is_read_left_out_and_the_rest_on_the_full_axis():
    X, cls, split, _ = _planted()
    a, a_in = NM.readings(X, cls, split), NM.in_sample(X, cls, split)
    held = [i for i in range(len(cls)) if split[i] != "train"]
    assert torch.allclose(a[held], a_in[held], atol=1e-12)
    i = 0
    rows = [r for r in range(len(cls)) if split[r] == "train" and r != i]
    assert abs(float(a[i]) - float(NM._a(X[i], NM._fit(X, cls, rows)))) < 1e-12
    assert float(a[i]) != pytest.approx(float(a_in[i]))


def test_a_planted_axis_is_found_and_noise_is_not():
    X, cls, split, phr = _planted(sep=3.0)
    s = NM.summary(NM.readings(X, cls, split), NM.in_sample(X, cls, split), cls, split, phr)
    assert s["loo_train"] >= 55 and s["heldout_sides"] == 8 and s["heldout_sep"] > 1.0
    Xn, *_ = _planted(sep=0.0, seed=1)
    sn = NM.summary(NM.readings(Xn, cls, split), NM.in_sample(Xn, cls, split), cls, split, phr)
    assert sn["loo_train"] <= 40           # left out, a noise axis reads at or below chance (it leans away from the row left out)


def test_the_push_takes_one_side_at_a_time():
    D = {"up": torch.tensor([1.0, 0.0]), "down": torch.tensor([0.0, 1.0])}
    assert torch.equal(NC.anima_push(0.5, 2.0, D), torch.tensor([1.0, 0.0]))
    assert torch.equal(NC.anima_push(-1.5, 2.0, D), torch.tensor([0.0, 3.0]))
    assert torch.equal(NC.anima_push(0.0, 2.0, D), torch.zeros(2))


def test_the_evaluated_sets_are_the_records_fourteen():
    cls, split, phr = _phrases()
    for c, ts in NC.EVAL_TRAIN.items():                          # place the record's evaluated training phrases
        for k, t in enumerate(ts):
            phr[[i for i in range(len(cls)) if cls[i] == c and split[i] == "train"][k]] = t
    s = NC.eval_sets(cls, split, phr)
    assert len(s) == 14 and list(s)[:2] == ["up/train/cheerful and upbeat", "up/train/joyful and uplifting"]
    assert len(NC.eval_sets(cls, split, phr, train=False)) == 10
    assert sum(v[0] == "neutral" for v in s.values()) == 2


def test_the_second_form_takes_the_neutral_push_off_both_sides():
    D = {"up": torch.tensor([3.0, 1.0]), "down": torch.tensor([1.0, 3.0]), "neutral": torch.tensor([1.0, 1.0])}
    out = NC.neutral_relative(D)
    assert set(out) == {"up", "down"} and torch.equal(out["up"], torch.tensor([2.0, 0.0]))
    assert torch.equal(out["down"], torch.tensor([0.0, 2.0]))


def test_the_slot_mean_ignores_padding():
    ctx = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    assert torch.equal(NC.slot_mean(ctx, torch.tensor([[1, 1, 0]])), torch.tensor([[2.0, 2.0]]))


def test_a_stack_concatenates_its_members_at_shared_taps():
    m1 = {"stream": {0: torch.ones(5, 2), 1: torch.ones(5, 2)}, "hub": {}, "size": torch.zeros(5)}
    m2 = {"stream": {1: torch.zeros(5, 3)}, "hub": {}, "size": torch.zeros(5)}
    s = CM.stacked([m1, m2])
    assert list(s["stream"]) == [1] and s["stream"][1].shape == (5, 5)


def _c1(vals):
    return {k: {"Q": {"alignment": v, "context_alignment": v / 2}} for k, v in vals.items()}


def test_the_stack_mark_needs_more_than_the_bar():
    C1 = _c1({"s|A-trunk-3": 0.45, "a|A-trunk-3": 0.40, "b|A-trunk-3": 0.39})
    C1b = _c1({"s|A-trunk-3": 0.452, "a|A-trunk-3": 0.405, "b|A-trunk-3": 0.39})
    out = CM.stack_marks({"C1": C1}, {"C1": C1b}, "s", ["a", "b"])
    assert out["A-trunk-3"]["U3"] == "THE STACK ADDS" and out["A-trunk-3"]["best_member"][0] == "a|A-trunk-3"
    C1["s|A-trunk-3"]["Q"]["alignment"] = 0.405
    assert CM.stack_marks({"C1": C1}, {"C1": C1b}, "s", ["a", "b"])["A-trunk-3"]["U3"] == "NOTHING ADDED"


def test_the_relay_closes_every_token_but_the_first_on_the_appended_space():
    from alephllm_diffusion.nano import relay as NR
    raw, a_end = b"A red bus.", [1, 5, 9, 10]                 # 'A', ' red', ' bus', '.'
    assert NR.closing_positions(a_end) == [6, 10, 11]           # in [DOC] + caption + ' '
    assert NR.prefixes(raw, a_end) == [b"A red ", b"A red bus.", b"A red bus. "]
    s = NR.states(torch.ones(4), torch.zeros(3, 4))
    assert s.shape == (4, 4) and bool((s[0] == 1).all()) and bool((s[1:] == 0).all())


def test_the_relay_map_recovers_a_planted_linear_map():
    from alephllm_diffusion.nano import relay as NR
    g = torch.Generator().manual_seed(0)
    X = torch.randn(400, 8, generator=g, dtype=F64)
    W = torch.randn(8, 5, generator=g, dtype=F64)
    Y = X @ W + 3.0 + 0.01 * torch.randn(400, 5, generator=g, dtype=F64)
    m = NR.fit(X, Y, torch.arange(400) // 4)
    assert m["cv_r2"] > 0.99 and float((NR.apply(m, X[:10]).double() - Y[:10]).abs().max()) < 0.1


class _ToyAdapter(torch.nn.Module):
    """the adapter's call on a small scale: T5 ids as queries, one head of attention over the masked source states."""

    def __init__(self, d=8, vocab=20, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Parameter(torch.randn(vocab, d, generator=g))
        self.wk = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)
        self.wv = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        q, h = self.emb[target_input_ids], source_hidden_states
        s = q @ (h @ self.wk).transpose(1, 2) / q.shape[-1] ** 0.5
        s = s.masked_fill(~source_attention_mask.bool()[:, None, :], float("-inf"))
        return torch.softmax(s, -1) @ (h @ self.wv)


def _toy_relay(n_cap=240, p=6, d=8, seed=0):
    """captions of 4-8 reader rows; Qwen3's states a saturating function of the reader's (what a ridge cannot fit); targets =
    the toy adapter's output from Qwen3's own states, the first token a fixed state of its own."""
    from alephllm_diffusion.nano import relay as NR
    from alephllm_diffusion.nano import relay_train as RT
    g = torch.Generator().manual_seed(seed)
    ns = torch.randint(4, 9, (n_cap,), generator=g).tolist()
    offs = RT.offsets(ns)
    X = torch.randn(offs[-1], p, generator=g)
    A = torch.randn(p, d, generator=g)
    Q = 2.0 * torch.tanh(X @ A) + 0.5
    first = torch.randn(n_cap, d, generator=g)
    t5 = [torch.randint(0, 20, (int(k),), generator=g) for k in torch.randint(3, 7, (n_cap,), generator=g)]
    ad = _ToyAdapter(d).requires_grad_(False)
    tgt = []
    for i in range(n_cap):
        src = torch.cat([first[i][None], Q[offs[i]:offs[i + 1]]])[None]
        tgt.append(ad(src, t5[i][None], None, torch.ones(1, len(src[0]), dtype=torch.long))[0])
    tr = list(range(n_cap * 3 // 4))
    rows = torch.cat([torch.arange(offs[i], offs[i + 1]) for i in tr])
    groups = torch.cat([torch.full((ns[i],), i) for i in tr])
    ridge = NR.fit(X[rows].double(), Q[rows].double(), groups)
    data = {"X": X, "offs": offs, "first": first, "t5": t5, "tgt": tgt, "Q": Q, "Q_mean": Q[rows].mean(0),
            "m": torch.cat([tgt[i] for i in tr]).mean(0)}
    return data, ridge, ad, tr, list(range(len(tr), n_cap))


def test_the_trained_relay_is_born_at_the_ridge():
    from alephllm_diffusion.nano import relay as NR
    from alephllm_diffusion.nano import relay_train as RT
    data, ridge, *_ = _toy_relay()
    conn = RT.Connector(ridge, hidden=16, seed=3)
    x = data["X"][:50]
    with torch.no_grad():
        assert float((conn(x).double() - NR.apply(ridge, x).double()).abs().max()) < 1e-5


def test_the_trained_relay_scores_only_real_slots_and_keeps_the_first_token():
    from alephllm_diffusion.nano import relay_train as RT
    data, ridge, ad, *_ = _toy_relay()
    conn = RT.Connector(ridge, hidden=16)
    src, mask = RT.source(conn, data, [0, 1])
    n0 = data["offs"][1] - data["offs"][0]
    assert torch.equal(src[0, 0], data["first"][0]) and int(mask[0].sum()) == n0 + 1
    with torch.no_grad():
        both = RT.batch_sums(conn, ad, data, [0, 1])
        apart = [RT.batch_sums(conn, ad, data, [i]) for i in (0, 1)]
    for k in (0, 1):                                  # padded source rows and padded T5 slots add nothing
        assert abs(float(both[k]) - float(apart[0][k] + apart[1][k])) < 1e-4 * float(both[k])


def test_the_trained_relay_learns_where_the_ridge_stops():
    from alephllm_diffusion.nano import relay_train as RT
    data, ridge, ad, tr, va = _toy_relay()
    conn = RT.Connector(ridge, hidden=32, seed=0)
    rec = RT.train(conn, ad, data, tr, va, steps=300, lr=3e-3, batch=32, seed=0, eval_every=100, log=lambda *_: None)
    assert rec["best_val_r2"] > rec["born_val_r2"] + 0.02, rec
    assert abs(RT.score(conn, ad, data, va) - rec["best_val_r2"]) < 1e-9
    assert RT.state_r2(conn, data, va) > 0.0
