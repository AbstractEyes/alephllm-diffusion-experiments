"""The caption reads of alephllm_diffusion.mount.refit on a stand-in (the arm mount itself lives in alephllm since 0.10.6 and is
tested there: geolip.alephllm.tests.test_arm_mount). The stand-in trunk and group come from that test (a 3-block random trunk,
three members written the training route's way). CPU only, seconds, no download.
  T5 caption_read_repro on a tiny pack (manifest + one shard per source, held rows, a stub frame): the stored read's fields, finite,
     the identities between them (alone_minus_off = alone - off), and compare_to_stored flags only a planted .05 difference.
  T6 refit_reference_logits with the stand-in trunk and files equals the library mount on a second copy of the trunk, 0.0, and with
     two members off equals the third mounted alone.
Runs under pytest, or alone (exit 1 on any failure).
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

from alephllm_diffusion.mount import refit as RM  # noqa: E402
from geolip.alephllm.tests.test_arm_mount import TINY, write_group  # noqa: E402

torch.manual_seed(0)
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)
    if not ok:
        FAILS.append(name)


def tiny_pack(out_dir, n_rows=24):
    """A pack with every source of the mix, one shard each, half the rows held."""
    man = {"sources": {}}
    for k in RM.PACK_MIX:
        os.makedirs(os.path.join(out_dir, k), exist_ok=True)
        f = f"{k}/{k}-00000.jsonl"
        with open(os.path.join(out_dir, f), "w", encoding="utf-8") as fh:
            for i in range(n_rows):
                row = {"id": f"{k}{i}", "split": "held" if i % 2 else "train", "tags": f"tag{i}, {k}", "caption": f"a {k} picture number {i}",
                       "short": f"short {k} {i}", "long": f"long {k} caption {i} " * 3, "rating": "general"}
                fh.write(json.dumps(row) + "\n")
        man["sources"][k] = {"shards": [{"file": f}]}
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8"))
    return out_dir


def stub_render(row):
    return "\n".join(f"{k}: {row[k]}" for k in ("tags", "caption", "short", "long", "rating") if row.get(k))


def main():
    members = ["s1_perspective", "s2_concept", "s9_caption"]
    tmp = tempfile.mkdtemp(prefix="refit_mount_test_")
    base = RM.load_trunk(step=None, cfg=TINY)
    state = {k: v.clone() for k, v in base.state_dict().items()}
    x = torch.randint(0, 256, (2, 40), generator=torch.Generator().manual_seed(1))
    bare = RM.logits(base, x)
    _, files = write_group(base, members, tmp)
    with torch.no_grad():
        ref = RM.logits(base, x)
    check("the stand-in arms write", float((ref - bare).abs().max()) > 1e-4)

    # T5 the caption read on a tiny pack
    m5 = RM.load_trunk(step=None, cfg=TINY, state=state)
    prog5 = RM.mount_refit_group(m5, "gTA", close=3, members=members, files=files, check_base=False)
    pack = tiny_pack(os.path.join(tmp, "pack"))
    bs = RM.caption_batches(pack, int(TINY.context), render_row=stub_render, nb=3, rows=2)
    check("T5 batches shaped (nb, rows, ctx + 1)", tuple(bs.shape) == (3, 2, int(TINY.context) + 1), str(tuple(bs.shape)))
    rec = RM.caption_read_repro(m5, prog5, bs, precision="fp32")
    r = rec["rows"]["s9_caption"]
    check("T5 the stored read's fields, finite", bool(np.isfinite([r["off"], r["all_on"]]).all() and set(rec["member_alone"]) == set(members)),
          f"off {r['off']:.3f} all_on {r['all_on']:.3f}")
    ident = all(abs(rec["member_alone"][m]["s9_caption"]["alone_minus_off"]["mean"] - (rec["member_alone"][m]["s9_caption"]["alone"] - r["off"])) < 1e-9
                for m in members)
    check("T5 identities between the fields", ident)
    stored = {"rows": {"s9_caption": {"off": r["off"], "all_on": r["all_on"] + 0.05}}, "member_alone": {}, "member_in_group": {}}
    for m in members:
        stored["member_alone"][m] = {"s9_caption": {"alone": rec["member_alone"][m]["s9_caption"]["alone"]}}
        stored["member_in_group"][m] = {"s9_caption": {"all_on_minus_masked": {"mean": rec["member_in_group"][m]["s9_caption"]["all_on_minus_masked"]["mean"]}}}
    table = RM.compare_to_stored(rec, stored)
    RM.print_comparison(table)
    check("T5 compare flags only the planted .05 row", [t["reading"] for t in table if t["flagged"]] == ["caption rows, all on"])
    with torch.no_grad():
        got = RM.logits(m5, x)
    check("T5 the read leaves every member on", torch.equal(got, ref))

    # T6 the full route against a second mount
    out6, _ = RM.refit_reference_logits(x, "gTA", trunk=RM.load_trunk(step=None, cfg=TINY, state=state), files=files, check_base=False,
                                        close=3, members=members)
    check("T6 refit_reference_logits == the mount on a second trunk (0.0)", torch.equal(out6, ref), f"max abs {float((out6 - ref).abs().max())}")
    m2 = RM.load_trunk(step=None, cfg=TINY, state=state)
    RM.mount_refit_group(m2, "gTA", close=3, members=["s9_caption"], files={"s9_caption": files["s9_caption"]}, check_base=False)
    with torch.no_grad():
        third_alone = RM.logits(m2, x)
    out6c, _ = RM.refit_reference_logits(x, "gTA", trunk=RM.load_trunk(step=None, cfg=TINY, state=state), files=files, check_base=False,
                                         close=3, members=members, off=["s1_perspective", "s2_concept"])
    check("T6 with two members off == the third alone", torch.equal(out6c, third_alone))

    print(f"\n{'ALL PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)} ({tmp})", flush=True)
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()


def test_caption_reads_and_reference_logits():
    """pytest entry: the checks above (exit 0 = all passed)."""
    try:
        main()
    except SystemExit as e:
        assert e.code == 0, FAILS
