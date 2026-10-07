"""alephllm_diffusion.stitch.pick - THE CHOSEN CELL, the pick of record (registered before any grid number): among her
(Beatrix's) record-fit cells of a full grid (her trunk x block x closing-byte convention x depth; the record fit = the ridge fitted
on COCO captions with the experiment's training captions), on the read of record (the 16 held-out phrases that T5 cuts into
fragments), the cell with the largest per-caption MOOD SIZE (stage 2's mood_size: the relay's signed projection on the ceiling's
leave-one-out mood axis over the ceiling's mean signed projection, the on-axis fraction delivered; a signed projection already
holds the cosine, so no product); TIES = the cells whose mood size lies inside the top cell's phrase-bootstrap interval
(mood_size_ci, as stored), broken by the higher centred primary, then by the shallower depth. Printed beside the pick, never
moving it: the coco-fit value (the ridge fitted on COCO captions alone) at the chosen block, convention and depth (the transfer
check, with its interval) and the per-scene companion (the axis-against-axis cosine and size, and their product). On the
rehearsal the pick is a PREVIEW, labelled; the pick of record is made on the final checkpoint's numbers.
THE COMPANION PICK (registered 2026-10-05, AFTER the 230,415 partial preview and BEFORE any 245,674 number, and labelled so
wherever it is printed): the same three keys with the tie band PAIRED: A = the cell with the largest mood size (as above); a cell
B ties A when the cluster-bootstrap interval of the per-caption difference size_i(A) - size_i(B) over the same captions of the
read of record (phrases as clusters, 2,000 draws, seed 0: stage 2's own cluster_ci) includes zero; among A and its ties the
higher centred primary, then the shallower depth. The pick of record stays the rule above; e029 (the picture test) runs the relay
at both cells when they differ (the untrained control at the pick of record only). The companion needs stage 2's per-caption
file with the mood rows, and torch; without them it is reported as not computable.
THE PICK PER MOUNT (registered 2026-10-06, before any read): a run holding several of her trunks (the 'mounts' config: her trunk
with a refit arm group mounted, one tag per mount) gets the same rule once per trunk, over that trunk's cells only:
`python -m alephllm_diffusion.stitch.pick mounts step245674_gCA` -> pick_mounts_step245674_gCA.json.
Usage: python -m alephllm_diffusion.stitch.pick <name> [trunk]
Reads stitch_<name>_full.json (a smoke run's stitch_<name>.json) and stitch_<name>_percaption.pt from the output folder
(settings.OUT_DIR); writes pick_<name>.json (pick_<name>_<trunk>.json with a trunk) to the report folder (settings.REPORT_DIR; a
smoke run's pick to the output folder)."""
import json
import os
import sys

from .. import settings

OUT_DIR = settings.OUT_DIR
REPORT_DIR = settings.REPORT_DIR
DEPTHS = ["0", "4", "8", "12", "16", "20", "24", "28pre", "28post"]
COMPANION_LABEL = ("THE COMPANION PICK (the paired tie band; registered 2026-10-05 03:0x after the 230,415 partial preview and before "
                   "any 245,674 number)")


def parse(key):
    """'record|step230415|b8|close|k12' -> (fit, trunk, block, convention, depth)."""
    fit, trunk, blk, conv, dep = key.split("|")
    return fit, trunk, int(blk[1:]), conv, dep[1:]


def companion(pc_path, rows, top):
    """The companion pick's paired tie band. Returns (the pick, its ties with their paired intervals) or (None, reason)."""
    if not os.path.exists(pc_path):
        return None, f"no per-caption file ({pc_path})"
    try:
        import torch
    except ImportError:
        return None, "torch is not importable here (install the package's requirements)"
    from .s2 import cluster_ci                      # the same function as every stored interval
    pc = torch.load(pc_path, weights_only=False)
    if "mood_proj_span" not in pc:
        return None, "the per-caption file predates the per-caption mood rows (a stage-2 file from before 2026-10-05)"
    rec, phr, ceil = pc["span_sets"]["record"], pc["phrase"], pc["ceiling_mood_proj_span"]

    def size(key):                                             # reads_span's arithmetic: the projection over the ceiling's mean
        pr = pc["mood_proj_span"][key]
        hx = rec & ~torch.isnan(pr)
        return pr / float(ceil[hx].mean()), hx

    s_a, h_a = size(top["key"])
    ties = []
    for x in rows:
        s_b, h_b = size(x["key"])
        assert torch.equal(h_a, h_b), f"{x['key']}: the captions with a mood differ from the top cell's"
        assert abs(float(s_b[h_b].mean()) - x["mood_size"]) < 1e-6, f"{x['key']}: the rows do not reproduce the stored mood size"
        lo, hi = cluster_ci((s_a - s_b)[h_a], phr[h_a])
        if x["key"] == top["key"] or (lo is not None and lo <= 0.0 <= hi):
            ties.append(dict(x, paired_ci=[lo, hi]))
    ties.sort(key=lambda x: (-x["primary_centred"], DEPTHS.index(x["depth"])))
    return ties[0], ties


def main(name, trunk=None):
    path = os.path.join(OUT_DIR, f"stitch_{name}_full.json")
    if not os.path.exists(path):
        path = os.path.join(OUT_DIR, f"stitch_{name}.json")
    result = json.load(open(path, encoding="utf-8"))
    if result.get("partial"):
        print(f"[pick] A PARTIAL GRID ({result['partial']['cells_done']} of {result['partial']['of']} cells; stopped: "
              f"{result['partial']['reason']}): the picks below are over the cells reached only")
    cells = result["span"]["cells"]
    her = [k for k in cells if k.startswith("record|step") and (trunk is None or parse(k)[1] == trunk)]
    trunks = sorted({parse(k)[1] for k in her})
    assert len(trunks) == 1, f"one trunk per pick: this grid holds {trunks}; name one (python -m alephllm_diffusion.stitch.pick {name} <trunk>)"
    rows = []
    for k in her:
        fit, tk, blk, conv, dep = parse(k)                    # tk, not trunk: the argument names the output file
        r = cells[k]["record"]
        if r.get("mood_size") is None:
            continue
        rows.append({"key": k, "trunk": tk, "block": blk, "conv": conv, "depth": dep, "mood_size": r["mood_size"],
                     "mood_size_ci": r["mood_size_ci"], "primary_centred": r["primary_centred"], "mood_cos": r["mood_cos"],
                     "transfer_axis_cos": r.get("transfer_axis_cos"), "transfer_axis_size": r.get("transfer_axis_size")})
    assert rows, "no record-fit cells of her trunk with a mood size"
    top = max(rows, key=lambda x: x["mood_size"])
    lo = top["mood_size_ci"][0]
    ties = [x for x in rows if lo is not None and x["mood_size"] >= lo]
    ties.sort(key=lambda x: (-x["primary_centred"], DEPTHS.index(x["depth"])))
    pick = ties[0] if ties else top
    ck = pick["key"].replace("record|", "coco|", 1)
    coco = cells.get(ck, {}).get("record")
    preview = name not in ("record", "mounts")
    label = ("PREVIEW (not the record's pick)" if preview else
             f"THE CHOSEN CELL OF THE MOUNT {trunks[0]}" if name == "mounts" else "THE CHOSEN CELL")
    print(f"[pick] {name}: {len(rows)} record-fit cells of her trunk; the largest mood size {top['key']} {top['mood_size']:.3f} "
          f"[{top['mood_size_ci'][0]:.3f}, {top['mood_size_ci'][1]:.3f}]; {len(ties)} cell(s) inside its interval")
    for x in ties[:12]:
        print(f"[pick]   tie: {x['key']} mood size {x['mood_size']:.3f}, centred primary {x['primary_centred']:+.3f}")
    ax_c, ax_s = pick["transfer_axis_cos"], pick["transfer_axis_size"]
    print(f"[pick] {label}: {pick['key']} (block {pick['block']}, {pick['conv']}, depth {pick['depth']}): mood size "
          f"{pick['mood_size']:.3f} [{pick['mood_size_ci'][0]:.3f}, {pick['mood_size_ci'][1]:.3f}], mood cosine {pick['mood_cos']:+.3f}, "
          f"centred primary {pick['primary_centred']:+.3f}; per-scene companion: axis cosine "
          f"{'n/a' if ax_c is None else format(ax_c, '+.2f')}, size {'n/a' if ax_s is None else format(ax_s, '.2f')}, product "
          f"{'n/a' if ax_c is None or ax_s is None else format(ax_c * ax_s, '.2f')}")
    if coco is not None:
        print(f"[pick] the coco fit at the same block, convention and depth (the transfer check, never moves the pick): mood size "
              f"{coco['mood_size']:.3f} [{coco['mood_size_ci'][0]:.3f}, {coco['mood_size_ci'][1]:.3f}], centred primary "
              f"{coco['primary_centred']:+.3f}")
    comp, comp_ties = companion(os.path.join(OUT_DIR, f"stitch_{name}_percaption.pt"), rows, top)
    if comp is None:
        print(f"[pick] {COMPANION_LABEL}: not computable: {comp_ties}")
    else:
        print(f"[pick] {COMPANION_LABEL}{': a PREVIEW' if preview else ''}: {comp['key']} (block {comp['block']}, {comp['conv']}, "
              f"depth {comp['depth']}): mood size {comp['mood_size']:.3f}, centred primary {comp['primary_centred']:+.3f}; "
              f"{len(comp_ties)} cell(s) tied with the top cell by the paired band")
        for x in comp_ties[:12]:
            print(f"[pick]   paired tie: {x['key']} mood size {x['mood_size']:.3f} (top minus it [{x['paired_ci'][0]:+.3f}, "
                  f"{x['paired_ci'][1]:+.3f}]), centred primary {x['primary_centred']:+.3f}")
        print("[pick] e029: " + ("the two picks are the same cell: nothing changes" if comp["key"] == pick["key"] else
                                 f"the picks differ: the relay arm runs at both cells (+128 images); the untrained control stays at "
                                 f"the pick of record {pick['key']}; the registered prediction is scored there"))
    ranked = sorted(rows, key=lambda x: -x["mood_size"])
    out = {"name": name, "label": label, "rule": __doc__.split("Usage")[0].strip(), "pick": pick, "top": top,
           "ties": ties, "coco_at_pick": coco, "ranked": ranked, "partial": result.get("partial"),
           "companion": {"label": COMPANION_LABEL, "pick": comp, "ties": comp_ties} if comp is not None else
           {"label": COMPANION_LABEL, "not_computable": comp_ties}}
    out["trunk"] = trunks[0]
    dest = OUT_DIR if name.startswith("smoke") else REPORT_DIR
    os.makedirs(dest, exist_ok=True)
    p = os.path.join(dest, f"pick_{name}{'_' + trunk if trunk else ''}.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"[pick] wrote {p}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
