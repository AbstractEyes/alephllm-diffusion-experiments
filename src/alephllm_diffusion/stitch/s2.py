"""Stage 2 of the stitch experiment: Qwen3's residuals, the closed-form ridges, the stitches and the reads.

Anima reads a caption through Qwen3 0.6B's last hidden state and its LLM adapter, a six-block transformer over the
caption's T5 token ids that cross-attends to Qwen3's states. This stage fits closed-form ridges from Beatrix's states ('her'
below: the byte-level AlephLLM model whose states stage 1 extracts), and from controls and floors, to Qwen3's residual at
several depths; writes a ridge's output into Qwen3 in place of its own residual (a stitch); runs the remaining layers and the
adapter; and compares the adapter's response with its response to Qwen3's own states. Only Qwen3 and Anima's LLM adapter run
(the diffusion-pipe fork's own adapter class and Anima's weights), in fp32 (a registered precision gate; a bf16 copy only
measures what half precision does); no image model. The definitions were registered on 2026-10-04, then amended and re-pointed
to the matched span, before any real run. 'Of record' marks the registered primary choice: the form of record, the read of
record, the grid of record.

Depth k = the residual entering Qwen3's layer k (k = 0: the input embedding); '28pre' = the residual after the last layer,
before the final norm (the norm then runs); '28post' = last_hidden_state (what Anima reads).
A RELAY maps a feature row to Qwen3's residual at depth k: her state (a stage-1 file, one block, the closing-byte convention =
her state at the byte after the token, or the last-byte convention = at the token's last byte), a control's state (the
untrained trunk: her architecture at random init, closing byte only), the SPELLING floor (hashed byte 1-3-gram counts of the
span, 4,096 dims), or 'Qwen static' (Qwen3's input embedding of the token). Each relay x depth is a closed-form ridge on
centred rows, its penalty picked by 5-fold cross-validation by caption over FIT_ALPHAS x trace/dim (fp64 solves).
Two fits: 'record' on every fit row, 'coco' on the COCO captions' rows only.
A cell = one fit x relay x depth (key 'fit|relay|k<depth>').
Each phrase has a mood: up (cheerful words), down (gloomy words) or neutral. The adapter's queries at the phrase's own T5
pieces are the 'questions' and the phrase's Qwen3 tokens the 'answer'. A(t, q) = the adapter's output for T5 ids t and Qwen3
states q at the question pieces, pooled (the mean over the pieces).

THE MATCHED SPAN (the form of record): the phrase stays in the caption and its own T5 pieces are the questions; the relay's
output replaces Qwen3's residual at depth k at EVERY Qwen3 token of the phrase (' and' included), each from that token's own
bytes, and the remaining layers run. CEILING = the unpatched caption. BASELINE (the answer absent) = Qwen3's states of the
caption with the phrase replaced by 'the filler', the neutral phrase of equal Qwen3 token count (stage 0's bank), under the
phrase caption's T5 ids. Effects at the question pieces: e_x = A(T5_phrase, Q_x) - A(T5_phrase, Q_filler). CENTRING by class:
each arm's effect minus its own leave-one-out mean over the phrases of the SAME T5 CLASS on the same scene, the up half and the
down half weighted equally, neutral phrases out of the reference.
The T5 classes, fixed a priori in stage 0 from T5's tokenization and reported separately: FRAGMENT-SPLIT = every content
word splits into pieces that carry no meaning of their own (its 16 held-out phrases are the read of record); MORPHEME-SPLIT =
split, and some piece is a real word belonging to the mood meaning; WHOLE = T5 keeps every content word whole (the
low-answer-share class; the smoke run measured its answer share at .30, not near zero, at the adapter's output); MIXED = the
rest (only in the 'all phrases' lines).
THE ANSWER SHARE per phrase on the stock model, |A(T5_phrase, Q_phrase) - A(T5_phrase, Q_filler)| / |A(T5_phrase, Q_phrase) -
A(T5_filler, Q_filler)|, each side pooled over its own pieces: printed, never used to move a phrase; one sensitivity line
regroups by it (>= .5).
TWO GUARDS (read from the same runs): the OFF-DIAGONAL column (the relay's centred effect for a phrase against the ceiling's
for a DIFFERENT phrase of the same class, filler and scene; the GAP = diagonal minus off-diagonal is the number quoted
when one arm stands alone, since the shared filler output lifts every arm's level alike; split by the partner's mood:
diagonal minus off-same = THIS WORD beyond its mood, off-same minus off-opposite = the mood alone); and the EIGHT STRONG
READING HEADS (hit and leak on them beside all heads; the share of answer tokens whose reading-piece logit stays inside the
real-word band of the logit read, beside the filler's share). Hit = the attention mass from the questions onto the answer;
leak = the same mass from the caption's other T5 positions. The heads and their bands come from two earlier analyses, the
interchange read and the logit read (logit_scale.json).
THE MOOD AXIS (beside the primary read): per caption, the up mean minus the down mean of the ceiling's centred effects
over the caption's class on its scene, the phrase itself left out. The relay's centred effect on that axis, signed by the
phrase's mood, gives the mood cosine and the mood size (its projection over the ceiling's mean projection).

THE ONE-SLOT FORM (the secondary column): the scene's caption "an illustration of {scene}." + one appended position (the
space token, unpatched = the EMPTY slot); the relay's output at the slot at depth k; the adapter reads it through T5's bare
space piece appended at the aligned position; CEILING = Qwen3's own final state at the phrase's last token, transplanted at
28post; SELF-STITCH(k) = Qwen3's own depth-k residual there; e_x = A(x) - A(empty) at the slot question, centred by the
leave-one-out mean over all phrases on the scene.
PRIMARY READ (both forms) = cos(centred e_relay, centred e_ceiling), the mean over the read captions (a held-out phrase, or a
seen phrase on a held-out scene), with its 95% cluster-bootstrap interval over phrases; beside it, the raw cosine and the
pooled companion. A written-out copy of the adapter gives the attention maps (hit, leak) and the answer's keys and values only.
THE MARGIN RULE (margin_block) decides which differences count. P1-P3 are the three registered predictions, printed beside
its results (registered_predictions): P1, her held-out margin over the untrained trunk and the spelling floor is smallest at
k = 0 and grows with depth; P2, the downstream penalty is lowest at an interior depth; P3, at that depth the primary read
sits inside the context spread for seen-type phrases and degrades on unseen words in step with the ridge's held-out residual.

The configs (CONFIGS): 'record' = the grid of record, on step 245,674; 'rehearsal' = the same grid on step 230,415;
'shakedown' = the one-slot plumbing run; 'smoke*' = smoke tests (their numbers mean nothing); 'mounts' = her cells with an arm
group mounted; 'frames' = her cells, bare and mounted, with every caption framed by the caption arm's label. Later programs
read the grid: alephllm_diffusion.stitch.pick chooses 'the pick', the cell with the largest mood size among her record-fit
cells on the read of record (and 'its companion', the same rule with a paired tie band); alephllm_diffusion.stitch.mounts
reads PM3(a), the registered prediction that mounting the arms raises the mood size at the pick (the mount contrast);
alephllm_diffusion.stitch.export writes the final Qwen3 states the picture test reads.

Reads: <OUT_DIR>/s0.pt (stage 0: the captions, the phrases and the token rows); <OUT_DIR>/s1_<tag>.pt for each of the
config's tags (stage 1); Qwen3's weights, Anima's DiT file (the adapter's weights) and the fork (the adapter class and Qwen3's
config), all located by alephllm_diffusion.settings; the reference files <REPORT_DIR>/logit_scale.json and, for the prior
printed beside P1-P3, <REPORT_DIR>/stitch_shakedown.json (alephllm_diffusion.storage.fetch_reference puts them in place).
Writes: one file per cell as it finishes, in <OUT_DIR>/stitch_<name>_cells_<start time>/ (or STITCH_CELLS_DIR); for a full
grid (the matched span, not a smoke test) the full JSON <OUT_DIR>/stitch_<name>_full.json and its summary
<REPORT_DIR>/stitch_<name>.json, otherwise <REPORT_DIR>/stitch_<name>.json (a smoke test's in OUT_DIR); the per-caption
reads <OUT_DIR>/stitch_<name>_percaption.pt; then, unless the run is a selection, the margin rule's results into those JSON
files. With STITCH_RUN_TAG set, the tag follows <name>.
Run controls (environment variables, defaults in brackets): the budgets STITCH_RAM_GB [16], STITCH_WALL_H [8],
STITCH_SOAK_CELLS [20], STITCH_SOAK_GB_PER_CELL [0.01], STITCH_MAX_CELLS [none], STITCH_RAM_HARD_GB [the budget + 16],
STITCH_MIDCELL_GB [3] and the card's per-process memory fraction STITCH_MEM_FRACTION [0.57]; the stop file <OUT_DIR>/STOP;
the selections and restarts STITCH_ONLY_CELLS, STITCH_ONLY_FIT, STITCH_EXTRA_RELAYS and STITCH_RESUME; STITCH_MAPS_ROWWISE=1
runs the frozen per-row maps; CUDA_VISIBLE_DEVICES=-1 runs on the CPU.
Usage: python -m alephllm_diffusion.stitch.s2 <config name>            (a name from CONFIGS)
       python -m alephllm_diffusion.stitch.s2 margins <config name>    (the margin rule again on a saved run, into its JSON)
"""
import copy
import json
import math
import os
import shutil
import sys
import threading
import time
import zlib

import torch
import torch.nn.functional as F

# the roots, from alephllm_diffusion.settings: each has a default (under the workspace folder; the fork: the checkout's submodule)
# and an environment variable that overrides it (STITCH_DP, STITCH_LLM_PATH, STITCH_DIT_PATH, STITCH_OUT_DIR, STITCH_REPORT_DIR)
from .. import settings  # noqa: E402

DP = settings.DP
LLM_PATH = settings.LLM_PATH
DIT_PATH = settings.DIT_PATH
OUT_DIR = settings.OUT_DIR
REPORT_DIR = settings.REPORT_DIR
LOGIT_JSON = os.path.join(REPORT_DIR, "logit_scale.json")    # the strong reading heads and their real-word logit bands
SUMMARY_SETS = ("record", "carried_held", "all_held", "single_held", "single_seen")
DEPTHS =[0, 4, 8, 12, 16, 20, 24, "28pre", "28post"]
FIT_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4]
HASH_DIM = 4096
CLASSES = ("FRAGMENT", "MORPHEME", "WHOLE", "MIXED")
CONFIGS = {
    # the plumbing run on step 230,415 (the one-slot form on the earlier 74-phrase stage 0, archived separately)
    "shakedown": {"her": ["step230415"], "controls": ["random0"], "blocks": [16, 24], "convs": ["close"],
                  "depths": [0, 12, "28pre", "28post"], "fits": ["record"], "floors": True, "saturation": False},
    # the grid of record on step 245,674: the matched span, the one-slot form as its secondary column (primary read, no maps)
    "record": {"her": ["step245674"], "controls": ["random0", "random1"], "blocks": [8, 12, 16, 18, 20, 22, 24, 28],
               "convs": ["close", "last"], "depths": DEPTHS, "fits": ["record", "coco"], "floors": True, "saturation": True,
               "span": True, "slot_full": False},
    # THE DRESS REHEARSAL: the record grid exactly, on step 230,415 (the start of the mixed anneal): full-scale plumbing (memory,
    # time per cell, every read on 32 scenes) and a pre-anneal comparison point; not the record, which is read on step 245,674
    "rehearsal": {"her": ["step230415"], "controls": ["random0", "random1"], "blocks": [8, 12, 16, 18, 20, 22, 24, 28],
                  "convs": ["close", "last"], "depths": DEPTHS, "fits": ["record", "coco"], "floors": True, "saturation": False,
                  "span": True, "slot_full": False},
    # smoke tests on the stage-1 limit files (their numbers mean nothing; kept out of the record): every code path once on the card,
    # and the matched span's paths on the CPU
    "smoke": {"her": ["step212000_limit80"], "controls": ["random0_limit80", "random1_limit80"], "blocks": [16],
              "convs": ["close", "last"], "depths": [0, "28pre", "28post"], "fits": ["record", "coco"], "floors": True,
              "saturation": False, "limit": 80, "span": True, "slot_full": True},
    "smoke_span": {"her": ["step212000_limit80"], "controls": ["random0_limit80", "random1_limit80"], "blocks": [16],
                   "convs": ["close"], "depths": [0, "28post"], "fits": ["record"], "floors": True, "saturation": False,
                   "limit": 80, "span": True, "slot_full": False},
    # the smoke chain of the mount configs below (numbers mean nothing, kept out of the record): the bare trunk at 245,674 with
    # the controls and floors, then the mounts and the frame column on the same few cells, for the pick per mount, the mount
    # contrast and the export with the mount arms (stage 1 at limit 80: step245674[_gCA|_gCB][_fcap]_limit80)
    "smoke_m0": {"her": ["step245674_limit80"], "controls": ["random0_limit80", "random1_limit80"], "blocks": [16],
                 "convs": ["close"], "depths": [0, "28post"], "fits": ["record"], "floors": True, "saturation": False,
                 "limit": 80, "span": True, "slot_full": False},
    "smoke_mounts": {"her": ["step245674_gCA_limit80", "step245674_gCB_limit80"], "controls": [], "blocks": [16],
                     "convs": ["close"], "depths": [0, "28post"], "fits": ["record"], "floors": False, "saturation": False,
                     "limit": 80, "span": True, "slot_full": False},
    "smoke_frames": {"her": ["step245674_fcap_limit80", "step245674_gCA_fcap_limit80", "step245674_gCB_fcap_limit80"],
                     "controls": [], "blocks": [16], "convs": ["close"], "depths": [0, "28post"], "fits": ["record"],
                     "floors": False, "saturation": False, "limit": 80, "span": True, "slot_full": False},
    # THE MOUNT FACTOR (registered 2026-10-06, before any read): her record-fit cells on the nine-arm mounts (the eight
    # stage arms and the caption arm; stage 1 --mount: step245674_gCA / _gCB, one per arm seed; STITCH_MOUNT_TAGS overrides
    # the list, e.g. adds step245674_gXA, the eight alone, when the mount read (alephllm_diffusion.mount.read) shows the eight
    # move caption states), both conventions x 8 blocks x 9 depths = 144 cells per mount. No controls and no floors: they do
    # not depend on her mount, so the record run's own are the shared ones (alephllm_diffusion.stitch.mounts reads both runs)
    "mounts": {"her": [t for t in os.environ.get("STITCH_MOUNT_TAGS", "step245674_gCA,step245674_gCB").split(",") if t],
               "controls": [], "blocks": [8, 12, 16, 18, 20, 22, 24, 28], "convs": ["close", "last"], "depths": DEPTHS,
               "fits": ["record"], "floors": False, "saturation": False, "span": True, "slot_full": False},
    # THE FRAME COLUMN of PM3(a) (registered with the mount factor): her trunk bare and mounted, every caption framed with
    # the caption arm's own label (stage 1 --frame=cap); run as a selection (STITCH_ONLY_CELLS) at the cells PM3(a) reads:
    # the record's pick and its companion, in each framed tag
    "frames": {"her": [t for t in os.environ.get("STITCH_FRAME_TAGS",
                                                 "step245674_fcap,step245674_gCA_fcap,step245674_gCB_fcap").split(",") if t],
               "controls": [], "blocks": [8, 12, 16, 18, 20, 22, 24, 28], "convs": ["close", "last"], "depths": DEPTHS,
               "fits": ["record"], "floors": False, "saturation": False, "span": True, "slot_full": False},
}
torch.backends.cuda.matmul.allow_tf32 = False             # TF32 off for every measurement (a standing rule)
torch.backends.cudnn.allow_tf32 = False
# EXECUTION CHANGES since the frozen registration (the protocol: definitions untouched, each named in the header, each gated
# against the frozen code before use and checked on one scene in every run; the stopped rehearsal's 7 cells are the reference)
EXECUTION_CHANGES = ["2026-10-05: adapter_maps_span's per-row copy-out batched (integer indices built once per batch from the "
                     "same masks; one copy to the CPU per block, not per row); the frozen function is kept as "
                     "adapter_maps_span_rowwise (STITCH_MAPS_ROWWISE=1 runs it)",
                     "2026-10-05: THE RESOURCE GATE (after the rehearsal grew to 36 GB of host memory): "
                     "the budgets declared in the header (host RAM, wall time, a planned cell limit for a soak run; the card's "
                     "per-process fraction as before); HOST = the larger of the resident set and committed bytes minus the card's "
                     "reserved memory (Windows charges the card's allocations to committed bytes: the stopped run's 36 GB committed "
                     "were 16 GB resident under a 19 GB card cap); a hard ceiling a few GB above the host budget plus the card's cap "
                     "(a Windows job object on committed bytes: an allocation past it fails inside Python and the run exits by itself, "
                     "no kill); host, resident, committed, the card's reserved memory and its peak printed after every cell; a "
                     "live-tensor census at cells 1, 5, 10, 15, 20, every 100th and any stop (CPU tensors reachable from Python, by "
                     "trailing shape: flat under growing host memory = the allocator's, growing = a kept reference, named); the soak "
                     "(host growth per cell over cells 5-20 against an allowance); a stop file; any of them stops the grid cleanly at "
                     "a cell boundary and saves the cells done as a PARTIAL grid; between cells the garbage collector and the C "
                     "runtime's heap release run (no effect on any number)",
                     "2026-10-05: THE PER-CELL FILES (a storage change): each cell's per-caption reads go "
                     "to disk as the cell finishes and leave memory: the centred primary of both forms (as before) and the span "
                     "form's per-caption mood projection and mood cosine (the rows behind mood_size and mood_cos, defined as before); "
                     "the per-caption file is assembled from them at the end or at a stop, so the margin rule and the paired tie band "
                     "run on any cells reached (a stopped grid labelled PARTIAL)",
                     "2026-10-05: THE SOAK fitted as a least-squares slope over this process's cells 5-20 (1-4 "
                     "named as warm-up) on both host measures (resident; committed minus the card's reserved); THE RESTART THAT "
                     "SKIPS FINISHED CELLS: each cell file also carries the cell's reads, its ridge R2 and its printed line; "
                     "STITCH_RESUME = an earlier run's cell folder reads its finished cells back (printed as first printed) instead "
                     "of recomputing them, a relay whose every depth is read back skips its ridge; the time-left estimate counts "
                     "only the cells this process computed",
                     "2026-10-05: every root overridable by an environment variable (STITCH_DP, STITCH_LLM_PATH, STITCH_DIT_PATH, "
                     "STITCH_OUT_DIR, STITCH_REPORT_DIR); since 2026-10-06 the roots come from alephllm_diffusion.settings (one "
                     "workspace folder, each location overridable by its variable) and this registration record from the "
                     "package's data/registration.json; no number moves",
                     "2026-10-05: THE SELECTIONS: STITCH_ONLY_CELLS (the port check: a few cells recomputed "
                     "on another card and compared with this card's printed lines) and STITCH_ONLY_FIT (one card per ridge fit); "
                     "every cell keeps its number in the full grid; a selection run is PARTIAL, writes under STITCH_RUN_TAG, "
                     "skips the margin rule; STITCH_RESUME takes several folders (the two halves; a merge run computes nothing); "
                     "STITCH_CELLS_DIR names the cell folder, which must start empty; no number moves",
                     "2026-10-05: THE HARD CEILING'S FLOOR: the model load peaks at 13.5 GB committed on the CPU (9.4 GB of it "
                     "the DiT file's copy-on-write mapping during its open) and a ceiling under that peak ended a test process with "
                     "an access violation in the mapped read, not a MemoryError; the ceiling is now the larger of the host ceiling "
                     "and 16 GB, plus the card's cap (the defaults were already above it); no number moves",
                     "2026-10-05 (the hazard: an abrupt end mid-card-work is the class of event that once reset the display "
                     "driver): THE STOPS IN THREE LAYERS. (a) THE HARD CEILING'S WIDE GAP: by default the budget + 16 GB + "
                     "the card's cap (was + 4), so only a runaway reaches it; (b) THE MID-CELL STOP: a watcher thread reads host "
                     "memory every half second and past the budget + 3 GB raises a flag that qwen_run, the adapter calls and the "
                     "ridge's folds read at their batch boundaries; the run leaves through a normal exception, the cell in hand "
                     "leaves no trace and the finished cells are saved as a PARTIAL grid (STITCH_TEST_MIDCELL_AT=n raises the flag "
                     "as cell n starts, for a test); (c) the header names each layer and says which one is abrupt; no number "
                     "moves",
                     "2026-10-05: STITCH_EXTRA_RELAYS: relays outside the registered grid (e.g. "
                     "the untrained trunk in the last-byte convention), appended after every registered relay and allowed only in a "
                     "selection of their own cells (checked before any load; the cells are numbered fit by fit, so beside a "
                     "registered cell of a second fit an outside relay would shift that cell's number), each such cell's line "
                     "labelled OUTSIDE THE REGISTERED GRID and the JSON listing them; no registered cell is computed in such a run; "
                     "no number moves",
                     "2026-10-06 (the nine-arm testing plan, registered before any read): two "
                     "configs of her cells only, MOUNTS (her trunk with a refit arm group mounted, stage 1 --mount) and FRAMES (the "
                     "caption arm's own label in front of every caption, stage 1 --frame=cap; read as a selection at the record's "
                     "picks); no controls and no floors in either (they do not depend on the mount or are read at the picks by "
                     "alephllm_diffusion.stitch.mounts from the record run); the record config, every definition and every "
                     "registered read untouched; no number of the record moves",
                     "2026-10-06: the stage programs moved into the alephllm_diffusion package with imports and paths only; the "
                     "smoke chain rerun through the package on the same inputs (stage 1 on every trunk, the grids, the picks, the "
                     "mount contrast, the exports) equals the earlier outputs tensor for tensor and number for number; no number "
                     "moves",
                     "2026-10-07: THE SCENE BATCHES: STITCH_SCENES_PER_PASS whole scenes share one pass of the one-slot and "
                     "matched-span runs (right-padded rows, each row's own slot position and masks; 1 = one scene per pass, the "
                     "earlier form); the copies to the CPU are made once per pass and the rows sliced there (the same values); "
                     "checked in every run that uses it: the empty slot and the filler span run in both forms, tier 1 bit-identical "
                     "or tier 2 under ten times the batch-shape floor, else the run stops"]
MAPS_ROWWISE = os.environ.get("STITCH_MAPS_ROWWISE") == "1"
SCENES_PER_PASS = int(os.environ.get("STITCH_SCENES_PER_PASS", "1"))     # whole scenes per pass (1 = one scene, the earlier form)
STOP_FILE = os.path.join(OUT_DIR, "STOP")    # touch it: the grid stops at the next cell boundary and saves what exists (a file older
#                                              than the job's start is ignored, so no stale file has to be deleted)
BUDGET = {"ram_gb": float(os.environ.get("STITCH_RAM_GB", "16")), "wall_h": float(os.environ.get("STITCH_WALL_H", "8")),
          "soak_cells": int(os.environ.get("STITCH_SOAK_CELLS", "20")),
          "soak_gb_per_cell": float(os.environ.get("STITCH_SOAK_GB_PER_CELL", "0.01")),
          "max_cells": int(os.environ.get("STITCH_MAX_CELLS", "0"))}       # > 0: a planned stop after that many cells (a soak run)
# THE HARD CEILING'S WIDE GAP: the budget + 16 GB (+ the card's cap, added where it is set), so slow growth meets the soft budget
# and the mid-cell stop long before it and only a true runaway reaches the abrupt last resort
BUDGET["ram_hard_gb"] = float(os.environ.get("STITCH_RAM_HARD_GB", str(BUDGET["ram_gb"] + 16)))
# the hard ceiling's floor (measured on the CPU): the model load peaks at 13.5 GB committed (4.0 GB resident),
# 9.4 GB of it the DiT file's copy-on-write mapping during its open; a ceiling under that peak ends the process with an access
# violation inside the mapped read (no MemoryError), so the ceiling never sits below this (2.5 GB of margin)
LOAD_COMMIT_GB = 16.0
MIDCELL_GB = float(os.environ.get("STITCH_MIDCELL_GB", "3"))    # THE MID-CELL STOP: host past the budget by this many GB
TEST_MIDCELL_AT = int(os.environ.get("STITCH_TEST_MIDCELL_AT", "0"))  # a test hook: the flag raised as this process's cell n starts
_MEM_STOP = threading.Event()
_MEM_STOP_AT: dict = {}


class HostMemoryStop(RuntimeError):
    """THE MID-CELL STOP: raised at a batch boundary once the host-memory watcher has seen host memory past the
    budget + MIDCELL_GB; the run leaves through a normal exception (no kill), the finished cells already on disk."""


def guard():
    """A batch boundary: leave through HostMemoryStop when the host-memory watcher has raised its flag."""
    if _MEM_STOP.is_set():
        raise HostMemoryStop(f"host memory {_MEM_STOP_AT.get('host', float('nan')):.1f} GB at {_MEM_STOP_AT.get('t', '?')}, "
                             f"past the budget {BUDGET['ram_gb']:g} + {MIDCELL_GB:g} GB")
HEAPMIN = os.environ.get("STITCH_HEAPMIN", "1") == "1"
CENSUS_CELLS, CENSUS_EVERY = (1, 5, 10, 15, 20), 100
RUN_TAG = os.environ.get("STITCH_RUN_TAG", "")                 # appended to the output names (a selection's files apart)


def host_mem_gb():
    """(host, resident, committed, the card's reserved) in GB. Windows' display driver model charges the card's allocations to the
    process's committed bytes (the stopped rehearsal: 36 GB committed, 16 GB resident, its card cap 19 GB), so committed bytes alone
    overstate the host's share by the card's: HOST, the figure the budget and the soak read, is the larger of the resident set and
    committed minus the card's reserved memory (psutil; 'private' = committed bytes on Windows). Elsewhere (Linux) there is no
    committed-bytes figure (the virtual size counts the card's whole address space), so committed reads as the resident set."""
    import psutil
    m = psutil.Process().memory_info()
    rss = m.rss / 2 ** 30
    priv = m.private / 2 ** 30 if hasattr(m, "private") else rss
    dev = torch.cuda.memory_reserved() / 2 ** 30 if torch.cuda.is_available() and torch.cuda.is_initialized() else 0.0
    return max(rss, priv - dev), rss, priv, dev


def mem_note():
    """'; host memory X GB (resident, committed, the card's reserved)' for the setup milestones."""
    host, rss, priv, dev = host_mem_gb()
    return f"; host memory {host:.2f} GB (resident {rss:.2f}, committed {priv:.2f}, the card's reserved {dev:.2f})"


def start_memory_watch(limit_gb, every=0.5):
    """The watcher behind the mid-cell stop: a daemon thread reads host memory (host_mem_gb) every `every` seconds and, past
    limit_gb, raises the flag that guard() reads at the next batch boundary (qwen_run, the adapter calls, the ridge's folds)."""
    def watch():
        while not _MEM_STOP.is_set():
            try:
                host = host_mem_gb()[0]
            except Exception:  # noqa: BLE001 - a failed reading never stops the run
                host = 0.0
            if host > limit_gb:
                _MEM_STOP_AT.update(host=host, t=time.strftime("%H:%M:%S"))
                _MEM_STOP.set()
                return
            time.sleep(every)
    th = threading.Thread(target=watch, name="stitch-host-memory-watch", daemon=True)
    th.start()
    return th


def tensor_census(top=5):
    """THE LIVE-TENSOR CENSUS: every CPU tensor the garbage collector can reach, each storage counted once. Returns (tensors, GB, the
    largest groups by trailing shape and dtype). Flat while committed memory grows = the growth is the allocator's (freed memory not
    returned to the system); growing = a reference kept across cells, named by its group."""
    import gc
    seen, groups, total = set(), {}, 0
    for o in gc.get_objects():
        try:
            if not torch.is_tensor(o) or o.device.type != "cpu":
                continue
            st = o.untyped_storage()
            ptr, nb = st.data_ptr(), st.nbytes()
        except Exception:  # noqa: BLE001 - a census never raises mid-run
            continue
        if ptr == 0 or ptr in seen:
            continue
        seen.add(ptr)
        total += nb
        g = groups.setdefault((tuple(o.shape[1:]), str(o.dtype).replace("torch.", "")), [0, 0])
        g[0] += 1
        g[1] += nb
    big = sorted(groups.items(), key=lambda kv: -kv[1][1])[:top]
    return len(seen), total / 2 ** 30, [{"trailing_shape": list(k[0]), "dtype": k[1], "n": v[0], "gb": v[1] / 2 ** 30}
                                        for k, v in big]


def hard_memory_limit(gb):
    """THE HARD CEILING: a Windows job object memory limit on this process's committed bytes, set above the budget (by
    default by 16 GB, plus the card's cap), so a runaway allocation fails inside Python (MemoryError, or torch's 'not enough
    memory') and the process exits by itself: no kill. A read through a file mapping past the limit is not an allocation that
    raises: the process ends with an access violation (measured), so the caller keeps the limit above the model load's
    committed peak (LOAD_COMMIT_GB). Returns the limit in GB, or None when it cannot be set (the soft budget still holds)."""
    if os.name != "nt" or gb <= 0:
        return None
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x100                                  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = int(gb * 2 ** 30)
    if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):    # JobObjectExtendedLimitInformation
        return None
    if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):            # the handle stays open for the process's life
        return None
    return gb


def release_heap():
    """Between cells: the garbage collector, and the C runtime heap asked to return its free memory to the system (Windows:
    _heapmin; Linux: glibc's malloc_trim); no effect on any number."""
    import ctypes
    import gc
    gc.collect()
    try:
        if os.name == "nt":
            ctypes.CDLL("ucrtbase")._heapmin()
        elif sys.platform.startswith("linux"):
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 - best effort
        pass


# ---- models ------------------------------------------------------------------------------------------------------------
def load_models(device, dtype=torch.float32):
    import transformers
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    sys.path.insert(0, DP)
    cwd = os.getcwd()
    os.chdir(DP)
    try:
        from models.llm_adapter import LLMAdapter
        cfg = transformers.Qwen3Config.from_pretrained("configs/qwen3_06b", local_files_only=True)
        cfg.use_cache = False
        with init_empty_weights():
            lm = transformers.Qwen3ForCausalLM(cfg)
        with safe_open(LLM_PATH, "pt") as f:
            for key in f.keys():
                set_module_tensor_to_device(lm, key, device="cpu", dtype=dtype, value=f.get_tensor(key))
        qwen = lm.model.to(device).eval().requires_grad_(False)
        ad = LLMAdapter(source_dim=1024, target_dim=1024, model_dim=1024, num_layers=6, self_attn=True)
        sd = {}
        with safe_open(DIT_PATH, "pt") as f:
            for key in f.keys():
                if key.startswith("net.llm_adapter."):
                    sd[key[len("net.llm_adapter."):]] = f.get_tensor(key)
        res = ad.load_state_dict(sd, strict=False)
        assert not res.unexpected_keys and all(k.endswith("inv_freq") for k in res.missing_keys), res
        ad = ad.to(device=device, dtype=dtype).eval().requires_grad_(False)
    finally:
        os.chdir(cwd)
    return qwen, ad


@torch.no_grad()
def qwen_run(qwen, ids_list, device, patch=None, collect=None, collect_all=None):
    """Right-padded batch through Qwen3 in its own dtype. patch = (depth, positions [B], vectors [B, D]) replaces one residual per row
    entering that depth, or (depth, rows [N], positions [N], vectors [N, D]) any set of them (the matched span); collect = {depth:
    positions [B]} returns those residuals [B, D] float32; collect_all = [depths] returns the whole residual [B, L, D] float32 (on the
    CPU). Returns (last_hidden_state [B, L, D] float32, {depth: tensor})."""
    guard()                                                    # a batch boundary (the mid-cell stop)
    L = max(len(x) for x in ids_list)
    ids = torch.zeros(len(ids_list), L, dtype=torch.long)
    am = torch.zeros(len(ids_list), L, dtype=torch.long)
    for b, x in enumerate(ids_list):
        ids[b, :len(x)], am[b, :len(x)] = torch.tensor(x), 1
    ids, am = ids.to(device), am.to(device)
    rows = torch.arange(len(ids_list), device=device)
    got, handles = {}, []
    if patch is not None:
        p_rows, p_pos, p_vec = ((rows, patch[1].to(device), patch[2]) if len(patch) == 3 else
                                (patch[1].to(device), patch[2].to(device), patch[3]))

    def module_at(d):
        return qwen.norm if d == "28pre" else qwen.layers[d]

    def pre_hook(d):
        def hook(module, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            if collect and d in collect:
                got[d] = hs[rows, collect[d].to(device)].float().clone()
            if collect_all and d in collect_all:
                got[d] = hs.float().cpu()
            if patch is not None and patch[0] == d:
                hs = hs.clone()
                hs[p_rows, p_pos] = p_vec.to(device, hs.dtype)
                if args:
                    return (hs, *args[1:]), kwargs
                kwargs["hidden_states"] = hs
                return args, kwargs
            return None
        return hook

    want = set(collect or {}) | set(collect_all or []) | ({patch[0]} if patch is not None else set())
    for d in want - {"28post"}:
        handles.append(module_at(d).register_forward_pre_hook(pre_hook(d), with_kwargs=True))
    try:
        out = qwen(input_ids=ids, attention_mask=am).last_hidden_state
    finally:
        for h in handles:
            h.remove()
    out = out.clone()
    if patch is not None and patch[0] == "28post":
        out[p_rows, p_pos] = p_vec.to(device, out.dtype)
    if collect and "28post" in collect:
        got["28post"] = out[rows, collect["28post"].to(device)].float().clone()
    if collect_all and "28post" in collect_all:
        got["28post"] = out.float().cpu()
    return out.float(), got


@torch.no_grad()
def adapter_module(ad, src, src_mask, t5_ids, t5_mask, device):
    """The adapter MODULE's output [B, T, D] float32 (the reads of record)."""
    guard()                                                    # a batch boundary (the mid-cell stop)
    p = next(ad.parameters())
    return ad(source_hidden_states=src.to(device, p.dtype), target_input_ids=t5_ids.to(device),
              target_attention_mask=t5_mask.to(device), source_attention_mask=src_mask.to(device)).float()


@torch.no_grad()
def adapter_maps(ad, src, src_mask, t5_ids, t5_mask, slot_pos, device):
    """The adapter's forward written out to keep its cross-attention maps: returns (output [B, T, D] float32, hit [B], leak [B], the
    slot's per-block per-head key (RMS-normed, before RoPE) [B, 6, H, hd] and value [B, 6, H, hd]). Used for the maps only."""
    guard()                                                    # a batch boundary (the mid-cell stop)
    from models.llm_adapter import apply_rotary_pos_emb
    p = next(ad.parameters())
    src, t5_ids = src.to(device, p.dtype), t5_ids.to(device)
    tm = t5_mask.to(device).bool()
    sm = src_mask.to(device).bool()
    rows = torch.arange(src.shape[0], device=device)
    x = ad.in_proj(ad.embed(t5_ids))
    pe_t = ad.rotary_emb(x, torch.arange(x.shape[1], device=device)[None])
    pe_s = ad.rotary_emb(x, torch.arange(src.shape[1], device=device)[None])
    hit, leak, keys, vals = [], [], [], []
    qpos = tm.sum(1) - 2                                                                     # the question: before T5's end token
    for blk in ad.blocks:
        normed = blk.norm_self_attn(x)
        x = x + blk.self_attn(normed, mask=tm[:, None, None, :], position_embeddings=pe_t, position_embeddings_context=pe_t)
        ca = blk.cross_attn
        h = blk.norm_cross_attn(x)
        B, T, _ = h.shape
        S = src.shape[1]
        q = ca.q_norm(ca.q_proj(h).view(B, T, ca.n_heads, ca.head_dim)).transpose(1, 2)
        k_raw = ca.k_norm(ca.k_proj(src).view(B, S, ca.n_heads, ca.head_dim))
        v_raw = ca.v_proj(src).view(B, S, ca.n_heads, ca.head_dim)
        keys.append(k_raw[rows, slot_pos.to(device)].float())
        vals.append(v_raw[rows, slot_pos.to(device)].float())
        q = apply_rotary_pos_emb(q, *pe_t)
        k = apply_rotary_pos_emb(k_raw.transpose(1, 2), *pe_s)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(ca.head_dim)
        att = att.masked_fill(~sm[:, None, None, :], float("-inf")).softmax(-1)              # [B, H, T, S]
        on_slot = att[rows, :, :, slot_pos.to(device)].float().mean(1)                       # [B, T]: the heads' mean mass on the slot
        hit.append(on_slot[rows, qpos])
        others = tm.clone()
        others[rows, qpos] = False
        leak.append((on_slot * others).sum(1) / others.sum(1))
        out = (att @ v_raw.transpose(1, 2)).transpose(1, 2).reshape(B, T, -1)
        x = x + ca.o_proj(out)
        x = x + blk.mlp(blk.norm_mlp(x))
    y = ad.norm(ad.out_proj(x)).float()
    return y, torch.stack(hit, 1).mean(1), torch.stack(leak, 1).mean(1), torch.stack(keys, 1), torch.stack(vals, 1)


@torch.no_grad()
def adapter_maps_span(ad, src, src_mask, t5_ids, t5_mask, qmask, amask, device, strong=()):
    """The adapter's forward written out for the matched span (maps only): per block and head, hit = the attention mass from the
    phrase's question pieces (qmask [B, T]) onto the phrase's answer tokens (amask [B, S]), the mean over the pieces; leak = the same
    mass from the caption's other T5 positions. Returns (output [B, T, D] float32, hit [B] and leak [B] as the means over blocks and
    heads, per row the answer tokens' per-block per-head keys (RMS-normed, before RoPE) and values [n_answer, 6, H, hd] on the CPU,
    hit and leak per block and head [B, 6, H], and per row the pre-softmax logits of the question pieces on each answer token, the
    mean over the pieces, on the `strong` (block, head) pairs [n_answer, len(strong)])."""
    guard()                                                    # a batch boundary (the mid-cell stop)
    from models.llm_adapter import apply_rotary_pos_emb
    p = next(ad.parameters())
    src, t5_ids = src.to(device, p.dtype), t5_ids.to(device)
    tm, sm = t5_mask.to(device).bool(), src_mask.to(device).bool()
    qm, am = qmask.to(device).bool(), amask.to(device).bool()
    B, S = src.shape[0], src.shape[1]
    others = tm & ~qm
    # the rows' answer and question positions as integer indices, built once from the masks on the CPU (a boolean index selects in
    # ascending order, and so do these), so the per-row selections need no device sync; the selected values are copied to the CPU
    # once per block, not once per row (the batched form of adapter_maps_span_rowwise; the maps check compares them in every run)
    a_idx = [amask[b].bool().nonzero().flatten() for b in range(B)]
    q_idx = [qmask[b].bool().nonzero().flatten() for b in range(B)]
    n_ans = [len(a) for a in a_idx]
    a_rows = torch.cat([torch.full((n,), b, dtype=torch.long) for b, n in enumerate(n_ans)]).to(device)
    a_cols = torch.cat(a_idx).to(device)
    a_idx_d, q_idx_d = [a.to(device) for a in a_idx], [q.to(device) for q in q_idx]
    x = ad.in_proj(ad.embed(t5_ids))
    T = x.shape[1]
    pe_t = ad.rotary_emb(x, torch.arange(T, device=device)[None])
    pe_s = ad.rotary_emb(x, torch.arange(S, device=device)[None])
    hit_h, leak_h, keys, vals = [], [], [], []
    slog = [[None] * len(strong) for _ in range(B)]
    for blk_i, blk in enumerate(ad.blocks):
        x = x + blk.self_attn(blk.norm_self_attn(x), mask=tm[:, None, None, :], position_embeddings=pe_t,
                              position_embeddings_context=pe_t)
        ca = blk.cross_attn
        h = blk.norm_cross_attn(x)
        q = ca.q_norm(ca.q_proj(h).view(B, T, ca.n_heads, ca.head_dim)).transpose(1, 2)
        k_raw = ca.k_norm(ca.k_proj(src).view(B, S, ca.n_heads, ca.head_dim))
        v_raw = ca.v_proj(src).view(B, S, ca.n_heads, ca.head_dim)
        keys.append(list(k_raw[a_rows, a_cols].float().cpu().split(n_ans)))
        vals.append(list(v_raw[a_rows, a_cols].float().cpu().split(n_ans)))
        q = apply_rotary_pos_emb(q, *pe_t)
        k = apply_rotary_pos_emb(k_raw.transpose(1, 2), *pe_s)
        lg = (q @ k.transpose(-1, -2)) / math.sqrt(ca.head_dim)                              # [B, H, T, S] before the softmax
        att = lg.masked_fill(~sm[:, None, None, :], float("-inf")).softmax(-1)
        on_h = (att * am[:, None, None, :]).sum(-1).float()                                   # [B, H, T]: each head's mass on the answer
        hit_h.append((on_h * qm[:, None, :]).sum(-1) / qm.sum(-1, keepdim=True))
        leak_h.append((on_h * others[:, None, :]).sum(-1) / others.sum(-1, keepdim=True).clamp(min=1))
        here = [(si, sh) for si, (sb, sh) in enumerate(strong) if sb == blk_i]
        if here:
            parts = [lg[b, sh][q_idx_d[b]][:, a_idx_d[b]].float().mean(0) for si, sh in here for b in range(B)]    # [n_answer] each
            got = torch.cat(parts).cpu().split([n for _ in here for n in n_ans])
            for j, (si, _sh) in enumerate(here):
                for b in range(B):
                    slog[b][si] = got[j * B + b]
        x = x + ca.o_proj((att @ v_raw.transpose(1, 2)).transpose(1, 2).reshape(B, T, -1))
        x = x + blk.mlp(blk.norm_mlp(x))
    y = ad.norm(ad.out_proj(x)).float()
    kk = [torch.stack([kb[b] for kb in keys], 1) for b in range(B)]
    vv = [torch.stack([vb[b] for vb in vals], 1) for b in range(B)]
    hh, lh = torch.stack(hit_h, 1).cpu(), torch.stack(leak_h, 1).cpu()                        # [B, 6, H]
    sl = [torch.stack(row, -1) if strong else None for row in slog]
    return y, hh.mean(-1).mean(-1), lh.mean(-1).mean(-1), kk, vv, hh, lh, sl


@torch.no_grad()
def adapter_maps_span_rowwise(ad, src, src_mask, t5_ids, t5_mask, qmask, amask, device, strong=()):
    """adapter_maps_span as frozen at the registration (the per-row copy-out, verbatim): kept for the equivalence check and for
    STITCH_MAPS_ROWWISE=1."""
    from models.llm_adapter import apply_rotary_pos_emb
    p = next(ad.parameters())
    src, t5_ids = src.to(device, p.dtype), t5_ids.to(device)
    tm, sm = t5_mask.to(device).bool(), src_mask.to(device).bool()
    qm, am = qmask.to(device).bool(), amask.to(device).bool()
    B, S = src.shape[0], src.shape[1]
    others = tm & ~qm
    x = ad.in_proj(ad.embed(t5_ids))
    T = x.shape[1]
    pe_t = ad.rotary_emb(x, torch.arange(T, device=device)[None])
    pe_s = ad.rotary_emb(x, torch.arange(S, device=device)[None])
    hit_h, leak_h, keys, vals = [], [], [], []
    slog = [[None] * len(strong) for _ in range(B)]
    for blk_i, blk in enumerate(ad.blocks):
        x = x + blk.self_attn(blk.norm_self_attn(x), mask=tm[:, None, None, :], position_embeddings=pe_t,
                              position_embeddings_context=pe_t)
        ca = blk.cross_attn
        h = blk.norm_cross_attn(x)
        q = ca.q_norm(ca.q_proj(h).view(B, T, ca.n_heads, ca.head_dim)).transpose(1, 2)
        k_raw = ca.k_norm(ca.k_proj(src).view(B, S, ca.n_heads, ca.head_dim))
        v_raw = ca.v_proj(src).view(B, S, ca.n_heads, ca.head_dim)
        keys.append([k_raw[b, am[b]].float().cpu() for b in range(B)])
        vals.append([v_raw[b, am[b]].float().cpu() for b in range(B)])
        q = apply_rotary_pos_emb(q, *pe_t)
        k = apply_rotary_pos_emb(k_raw.transpose(1, 2), *pe_s)
        lg = (q @ k.transpose(-1, -2)) / math.sqrt(ca.head_dim)                              # [B, H, T, S] before the softmax
        att = lg.masked_fill(~sm[:, None, None, :], float("-inf")).softmax(-1)
        on_h = (att * am[:, None, None, :]).sum(-1).float()                                   # [B, H, T]: each head's mass on the answer
        hit_h.append((on_h * qm[:, None, :]).sum(-1) / qm.sum(-1, keepdim=True))
        leak_h.append((on_h * others[:, None, :]).sum(-1) / others.sum(-1, keepdim=True).clamp(min=1))
        for si, (sb, sh) in enumerate(strong):
            if sb == blk_i:
                for b in range(B):
                    slog[b][si] = lg[b, sh][qm[b]][:, am[b]].float().mean(0).cpu()            # [n_answer]
        x = x + ca.o_proj((att @ v_raw.transpose(1, 2)).transpose(1, 2).reshape(B, T, -1))
        x = x + blk.mlp(blk.norm_mlp(x))
    y = ad.norm(ad.out_proj(x)).float()
    kk = [torch.stack([kb[b] for kb in keys], 1) for b in range(B)]
    vv = [torch.stack([vb[b] for vb in vals], 1) for b in range(B)]
    hh, lh = torch.stack(hit_h, 1).cpu(), torch.stack(leak_h, 1).cpu()                        # [B, 6, H]
    sl = [torch.stack(row, -1) if strong else None for row in slog]
    return y, hh.mean(-1).mean(-1), lh.mean(-1).mean(-1), kk, vv, hh, lh, sl


def maps_equivalence(a, b, bar):
    """The two-tier gate on two outputs of the maps functions: tier 1 = every tensor torch.equal; tier 2 = the largest
    relative difference (|a - b| max over |b| max, per tensor) under `bar` (ten times the fp32 batch-shape floor). Returns
    (tier 1 or 2 or None, the largest relative difference)."""
    flat_a, flat_b = [], []
    for x, y in zip(a, b):
        xs, ys = (list(x), list(y)) if isinstance(x, list) else ([x], [y])
        for u, v in zip(xs, ys):
            if u is None and v is None:
                continue
            flat_a.append(u)
            flat_b.append(v)
    if all(u.shape == v.shape and torch.equal(u.cpu(), v.cpu()) for u, v in zip(flat_a, flat_b)):
        return 1, 0.0
    if any(u.shape != v.shape for u, v in zip(flat_a, flat_b)):
        return None, float("inf")
    mx = max(float((u.cpu() - v.cpu()).abs().max() / v.cpu().abs().max().clamp(min=1e-30)) for u, v in zip(flat_a, flat_b))
    return (2 if mx < bar else None), mx


# ---- features ----------------------------------------------------------------------------------------------------------
def spelling(texts):
    """Hashed byte 1-3-gram counts [N, HASH_DIM] float32."""
    out = torch.zeros(len(texts), HASH_DIM)
    for i, t in enumerate(texts):
        b = t.encode("utf-8")
        for n in (1, 2, 3):
            for j in range(len(b) - n + 1):
                out[i, zlib.crc32(bytes([n]) + b[j:j + n]) % HASH_DIM] += 1.0
    return out


def eval_view(s0, limit):
    """The eval captions and the matched-span token rows a run uses (stage 1's eval_view: limit = the eval captions of the first two
    scenes, the token rows re-indexed to them)."""
    evals, tok = s0["evals"], s0["eval_tok"].clone()
    if not limit:
        return evals, tok
    keep_sc = sorted({e["scene"] for e in evals})[:2]
    keep = [i for i, e in enumerate(evals) if e["scene"] in keep_sc]
    remap = {old: new for new, old in enumerate(keep)}
    m = torch.tensor([int(x) in remap for x in tok[:, 0].tolist()], dtype=torch.bool)
    tok = tok[m]
    tok[:, 0] = torch.tensor([remap[int(x)] for x in tok[:, 0].tolist()], dtype=torch.long)
    return [evals[i] for i in keep], tok


def pad_batch(seqs):
    L = max(len(x) for x in seqs)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    m = torch.zeros(len(seqs), L, dtype=torch.long)
    for b, x in enumerate(seqs):
        ids[b, :len(x)], m[b, :len(x)] = torch.tensor(x), 1
    return ids, m


# ---- ridge -------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def ridge_cv(X, Ys, folds, device):
    """Closed-form ridges X -> each Y in Ys (a list), the penalty per target picked by cross-validation over the folds (alpha x
    trace/dim). Returns per target (W [p, D], mu_x [p], mu_y [D], alpha, cv R2). Fits and solves in fp64."""
    X = X.to(device, torch.float64)
    folds = folds.to(device)
    out = []
    p = X.shape[1]
    nf = int(folds.max()) + 1
    fold_stats = []
    for f in range(nf):                                 # per fold: the training Gram's eigenbasis and the held-out rows in it
        guard()                                         # a batch boundary (the mid-cell stop)
        tr, te = folds != f, folds == f
        if not bool(te.any()):
            continue
        mx = X[tr].mean(0)
        Xt = X[tr] - mx
        if Xt.shape[0] <= p:                            # fewer rows than features: the thin SVD (exact: W lives in the row space)
            _, sv, Vh = torch.linalg.svd(Xt, full_matrices=False)
            s, V, trace = sv ** 2, Vh.T, float((sv ** 2).sum())
        else:
            G = Xt.T @ Xt
            s, V = torch.linalg.eigh(G)
            s, trace = s.clamp(min=0), float(G.trace())
        del Xt
        fold_stats.append((tr, te, mx, (X[te] - mx) @ V, V, s, trace / p))
    for Y in Ys:
        guard()                                         # a batch boundary (the mid-cell stop)
        Y = Y.to(device, torch.float64)
        err = torch.zeros(len(FIT_ALPHAS), device=device, dtype=torch.float64)
        tot = 0.0
        for tr, te, mx, A, V, s, tr_scale in fold_stats:
            my = Y[tr].mean(0)
            Bm = V.T @ ((X[tr] - mx).T @ (Y[tr] - my))
            Yh = Y[te] - my
            tot += float((Yh ** 2).sum())
            for ai, a in enumerate(FIT_ALPHAS):
                pred = A @ (Bm / (s + a * tr_scale)[:, None])
                err[ai] += ((Yh - pred) ** 2).sum()
        best = int(err.argmin())
        mx, my = X.mean(0), Y.mean(0)
        Xc = X - mx
        G = Xc.T @ Xc
        lam = FIT_ALPHAS[best] * float(G.trace()) / p
        W = torch.linalg.solve(G + lam * torch.eye(p, device=device, dtype=torch.float64), Xc.T @ (Y - my))
        out.append((W, mx, my, FIT_ALPHAS[best], 1.0 - float(err[best]) / tot))
    return out


# ---- the run -----------------------------------------------------------------------------------------------------------
def cosine(a, b):
    return F.cosine_similarity(a, b, dim=-1)


def cluster_ci(vals, clusters, n_boot=2000, seed=0):
    """95% interval of the mean of vals by a cluster bootstrap: the clusters (phrases) resampled with replacement, each drawn
    cluster bringing all its captions (the margin rule). (None, None) under two clusters."""
    uniq = sorted(set(clusters.tolist()))
    if len(uniq) < 2:
        return None, None
    sums = torch.tensor([float(vals[clusters == c].double().sum()) for c in uniq], dtype=torch.float64)
    cnts = torch.tensor([float((clusters == c).sum()) for c in uniq], dtype=torch.float64)
    draws = torch.randint(len(uniq), (n_boot, len(uniq)), generator=torch.Generator().manual_seed(seed))
    means = sums[draws].sum(1) / cnts[draws].sum(1)
    q = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(q[0]), float(q[1])


def pooled_cos(a, b):
    """The pooled companion: the summed inner products over captions over the root of the product of the summed squared
    norms, so captions weigh by how much effect they carry."""
    return float((a * b).sum() / ((a ** 2).sum() * (b ** 2).sum()).sqrt().clamp(min=1e-12))


def strong_band():
    """The eight strong reading heads of the interchange read and their REAL-WORD logit bands from the logit read (two earlier
    analyses, stored in LOGIT_JSON): per head, [the lower of the opposite-pair and same-pair 10th percentiles, the higher of their
    90th] of the reading pieces' logit change when another equal-count word answers (the pooled percentiles were not stored).
    Returns [(block, head)], lo [8], hi [8]."""
    j = json.load(open(LOGIT_JSON, encoding="utf-8"))
    pairs, lo, hi = [], [], []
    for n in j["strong_heads"]:
        b, h = (int(x[1:]) for x in n.split())
        r = j["heads"][n]["real"]
        pairs.append((b, h))
        lo.append(min(r["opposite"]["delta_p10"], r["same"]["delta_p10"]))
        hi.append(max(r["opposite"]["delta_p90"], r["same"]["delta_p90"]))
    return pairs, torch.tensor(lo), torch.tensor(hi)


def registration():
    """The registration record (the package's data/registration.json: when the grid's definitions were frozen and what
    changed in the stage programs since) with this run's execution changes."""
    from .. import __version__
    reg = dict(settings.registration())
    reg.update({"read_at": f"alephllm-diffusion-experiments {__version__}", "execution_changes": EXECUTION_CHANGES,
                "maps_path": "rowwise (frozen)" if MAPS_ROWWISE else "batched"})
    return reg


def summary(result, full_path):
    """The compact report of a full grid: the run's header, the span form's ceiling and cells on SUMMARY_SETS, the one-slot cells'
    held-out read, the margins; the full JSON stays in OUT_DIR (and ships to the data repo)."""
    s = {k: result[k] for k in ("config", "depths", "n_rows", "n_eval", "precision", "exactness", "registration", "empty",
                                "context_spread_centred_cos", "saturation", "partial") if k in result}
    if "resources" in result:                                  # the budgets, the soak and the stop; the per-cell log stays in the full JSON
        s["resources"] = {k: v for k, v in result["resources"].items() if k != "per_cell"}
    s["full_json"] = full_path
    if "span" in result:
        sp = result["span"]
        s["span"] = {k: sp[k] for k in ("precision", "exactness", "n_record_phrases", "classes", "per_phrase",
                                        "context_spread_centred_cos")}
        s["span"]["ceiling"] = {lab: sp["ceiling"][lab] for lab in SUMMARY_SETS}
        s["span"]["cells"] = {key: {"alpha": c["alpha"], "cv_r2": c["cv_r2"], "shrink": c.get("shrink"),
                                    "record_rank_mean": c["record_rank_mean"], "record_rank_chance": c["record_rank_chance"],
                                    **{lab: c[lab] for lab in SUMMARY_SETS}} for key, c in sp["cells"].items()}
    s["slot_self"] = {d: v["held_all"] for d, v in result["self"].items()}
    s["slot_cells"] = {key: {"cv_r2": c["cv_r2"], "held_all": c["held_all"]} for key, c in result["cells"].items()}
    if "margins" in result:
        s["margins"] = result["margins"]
    return s


def band_shares(arm, ceil, lo, hi):
    """arm, ceil: the reading pieces' mean logit on each answer token per strong head [n_answer, 8]. Returns the share of (token,
    head) pairs whose change against the ceiling stays inside the real-word band, the share of tokens inside on all eight heads, and
    the share of heads inside for the change pooled over the phrase's tokens (the logit read's own unit)."""
    d = arm - ceil
    inside = (d >= lo) & (d <= hi)
    dw = d.mean(0)
    return (float(inside.float().mean()), float(inside.all(-1).float().mean()), float(((dw >= lo) & (dw <= hi)).float().mean()))


def outside_relays():
    """STITCH_EXTRA_RELAYS: "trunk|bB|conv,..." = relays outside the registered grid. Allowed only in a
    selection of their own cells (STITCH_ONLY_CELLS naming those cells alone), so a run with them computes no registered cell: the
    cells are numbered fit by fit, and in a two-fit config an outside relay's cells would sit inside the first fit's numbers."""
    extra = [x for x in os.environ.get("STITCH_EXTRA_RELAYS", "").split(",") if x]
    only = {k for k in os.environ.get("STITCH_ONLY_CELLS", "").split(",") if k}
    assert not extra or (only and all("|".join(k.split("|")[1:4]) in extra for k in only)), (
        "relays outside the registered grid run only in a selection of their own cells (STITCH_ONLY_CELLS = those cells alone): "
        f"{extra} with {sorted(only)}")
    return extra


def main(name: str, device: str = "cuda"):
    cfg = CONFIGS[name]
    span_on = cfg.get("span", False)
    slot_full = cfg.get("slot_full", True)
    extra = outside_relays()                                   # checked before any load: a wrong selection fails in a second
    t0 = time.time()
    # THE HARD CEILING first, so it covers every load: it limits committed bytes, which on Windows include the card's allocations,
    # so it is the host ceiling plus the card's cap (a limit on the host alone would refuse the card's own allocations)
    card_cap = (float(os.environ.get("STITCH_MEM_FRACTION", "0.57")) * torch.cuda.get_device_properties(0).total_memory / 2 ** 30
                if device == "cuda" else 0.0)
    hard = hard_memory_limit(max(BUDGET["ram_hard_gb"], LOAD_COMMIT_GB) + card_cap)
    s0 = torch.load(os.path.join(OUT_DIR, "s0.pt"), weights_only=False)
    fit, rows, bases = s0["fit"], s0["rows"], s0["bases"]
    lim = cfg.get("limit", 0)
    if span_on:
        evals, eval_tok = eval_view(s0, lim)
    else:
        evals = s0["evals"] if not lim else [e for e in s0["evals"] if e["scene"] in sorted({x["scene"] for x in s0["evals"]})[:2]]
        eval_tok = None
    if lim:
        fit = fit[:lim]
        rows = rows[rows[:, 0] < lim]
    depths = cfg["depths"]
    if device == "cuda":                                # the card's memory cap for this job
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("STITCH_MEM_FRACTION", "0.57")))
    qwen, ad = load_models(device)
    E = len(evals)
    print(f"[stitch s2] {name}: models loaded in fp32 ({time.time() - t0:.0f} s); {len(rows)} fit rows, {E} eval captions "
          f"({sum(e['read'] for e in evals)} read), depths {depths}; forms: "
          f"{'the matched span (of record) + the one-slot (secondary)' if span_on else 'the one-slot'}", flush=True)
    reg = registration()
    print(f"[stitch s2] THE REGISTRATION FROZEN AT {reg['frozen']}; the stage tools since: "
          f"{'unchanged' if not reg['tools_changed_since'] else 'CHANGED (rehearsal-informed changes, flagged): ' + reg['tools_changed_since']}"
          f"{'; git: ' + reg['git_error'] if reg.get('git_error') else ''}"
          f"{' -- THE DRESS REHEARSAL on 230,415, NOT the read of record' if name == 'rehearsal' else ''}", flush=True)
    print("[stitch s2] EXECUTION CHANGES SINCE THE FREEZE (definitions untouched; each gated against the frozen code): "
          + " | ".join(EXECUTION_CHANGES) + f"; the maps path in this run: {reg['maps_path']}", flush=True)
    start_memory_watch(BUDGET["ram_gb"] + MIDCELL_GB)          # after the load: its transient file-mapping charge is known
    print(f"[stitch s2] BUDGETS: host RAM {BUDGET['ram_gb']:g} GB (host = the larger of the resident set and committed bytes "
          f"minus the card's reserved memory); wall {BUDGET['wall_h']:g} h; the card's per-process fraction "
          f"{os.environ.get('STITCH_MEM_FRACTION', '0.57')} ({card_cap:.1f} GB); THE SOAK: a least-squares slope over this "
          f"process's cells 5-{BUDGET['soak_cells']} (cells 1-4 are warm-up: first-use allocations) on both host measures "
          f"(resident; committed minus the card's reserved), each at most {BUDGET['soak_gb_per_cell']:g} GB per cell"
          + (f"; a planned stop after {BUDGET['max_cells']} cells (a soak run)" if BUDGET["max_cells"] else "")
          + f"; the heap release between cells {'on' if HEAPMIN else 'off'}. THE STOPS, gentlest first: (1) THE CELL-BOUNDARY "
          f"STOP: past the host budget, the wall budget, the soak's allowance or on request (touch {STOP_FILE}), the grid stops "
          f"after the cell in hand and saves every cell done as a PARTIAL grid; (2) THE MID-CELL STOP: a watcher thread reads "
          f"host memory every half second and, past the budget + {MIDCELL_GB:g} GB "
          f"({BUDGET['ram_gb'] + MIDCELL_GB:g} GB), the run leaves at the next batch boundary inside the cell through a "
          f"normal exception (no kill): the finished cells are on disk and saved as a PARTIAL grid, and only the cell in hand "
          f"is lost (a restart recomputes it); (3) THE HARD CEILING, ABRUPT, the last resort: "
          + (f"a Windows job object at {hard:.1f} GB committed = the host ceiling "
             f"{max(BUDGET['ram_hard_gb'], LOAD_COMMIT_GB):g} GB (by default the budget + 16; never under the model load's "
             f"committed peak) + the card's cap, set where only a runaway reaches it; an allocation past it fails inside Python, "
             f"but a read through a file mapping past it ends the process at once"
             if hard else "not set on this system (the two stops above hold)") + mem_note(), flush=True)

    # 1. EXACTNESS CHECKS (registered: they gate every read)
    b0 = bases[evals[0]["scene"]]
    ids = [b0["qwen_ids"]] * 2
    slot = torch.tensor([b0["slot_n"]] * 2)
    base_out, got = qwen_run(qwen, ids, device, collect={d: slot for d in depths})
    for d in depths:
        o2, _ = qwen_run(qwen, ids, device, patch=(d, slot, got[d]))
        assert torch.equal(o2, base_out), f"the self-patch at depth {d} is not bit-exact"
    ev0 = evals[0]
    _, gotf = qwen_run(qwen, [ev0["qwen_ids"]], device, collect={"28post": torch.tensor([ev0["t_ceil"]])})
    base1, _ = qwen_run(qwen, [b0["qwen_ids"]], device)          # the same batch shape as the hooked run (fp32 kernels vary by shape)
    batch_floor = float((base1[0] - base_out[0]).abs().max() / base_out[0].abs().max())
    o_hook, _ = qwen_run(qwen, [b0["qwen_ids"]], device, patch=("28post", slot[:1], gotf["28post"]))
    direct = base1.clone()
    direct[0, b0["slot_n"]] = gotf["28post"][0].to(direct.device)
    sm1 = torch.ones(1, len(b0["qwen_ids"]))
    t51 = torch.tensor([b0["t5_ids"]])
    tm1 = torch.ones_like(t51)
    ya = adapter_module(ad, o_hook, sm1, t51, tm1, device)
    yb = adapter_module(ad, direct, sm1, t51, tm1, device)
    assert torch.equal(ya, yb), "the 28post transplant does not reproduce the ceiling's adapter output"
    yc = adapter_maps(ad, o_hook, sm1, t51, tm1, slot[:1], device)[0]
    gap = float((ya - yc).abs().max() / ya.abs().max())
    assert gap < 1e-4, f"the written-out adapter differs from the module in fp32 ({gap:.2e} relative; the gate is 1e-4)"
    print(f"[stitch s2] exactness checks passed (the one-slot form): the self-patch is bit-exact at every depth; the 28post "
          f"transplant reproduces the ceiling's adapter output; the written-out adapter matches the module in fp32 ({gap:.1e} "
          f"relative); the batch-shape floor (one caption alone vs in a batch of two, Qwen3's output) {batch_floor:.1e} relative",
          flush=True)

    # 2. Qwen3's residuals at the fit rows (targets) and at the eval captions' phrase-final tokens (the slot's self-stitch, ceiling)
    by_cap: dict = {}
    for r, (ci, t, *_rest) in enumerate(rows.tolist()):
        by_cap.setdefault(ci, []).append((r, t))
    Y = {d: torch.zeros(len(rows), 1024) for d in depths}
    caps = sorted(by_cap)
    for c0 in range(0, len(caps), 64):
        grp = caps[c0:c0 + 64]
        _, g = qwen_run(qwen, [fit[c]["qwen_ids"] for c in grp], device, collect_all=depths)
        gi = torch.tensor([gi for gi, c in enumerate(grp) for _ in by_cap[c]])
        tt = torch.tensor([t for c in grp for _, t in by_cap[c]])
        rr = torch.tensor([r for c in grp for r, _ in by_cap[c]])
        for d in depths:
            Y[d][rr] = g[d][gi, tt]
        if (c0 // 64) % 10 == 0:
            el = time.time() - t0
            print(f"[stitch s2] targets: {min(c0 + 64, len(caps))}/{len(caps)} captions, {el:.0f} s spent", flush=True)
    selfv = {d: torch.zeros(E, 1024) for d in depths}
    for e0 in range(0, E, 64):
        grp = evals[e0:e0 + 64]
        _, g = qwen_run(qwen, [e["qwen_ids"] for e in grp], device, collect={d: torch.tensor([e["t_ceil"] for e in grp])
                                                                             for d in depths})
        for d in depths:
            selfv[d][e0:e0 + len(grp)] = g[d].cpu()
    print(f"[stitch s2] residuals collected ({time.time() - t0:.0f} s){mem_note()}", flush=True)

    # 3. features (her states and the controls kept in fp16 until a ridge needs them)
    folds = torch.tensor([fit[c]["fold"] for c in rows[:, 0].tolist()])
    coco_rows = torch.tensor([fit[c]["source"] == "coco" for c in rows[:, 0].tolist()])
    span_txt = [fit[c]["text"][a:e] for c, a, e in zip(rows[:, 0].tolist(), rows[:, 5].tolist(), rows[:, 6].tolist())]
    phrase_txt = [s0["phrases"][e["phrase"]]["text"] for e in evals]
    emb = qwen.embed_tokens.weight.float().cpu()
    relays = {}                                         # name -> (fit-row features, slot features [E], span token features [NT])
    forms = {}                                          # stage 1's batch form per file (files before 0.3.0: the equal form)
    for tag in cfg["her"] + cfg["controls"]:
        s1 = torch.load(os.path.join(OUT_DIR, f"s1_{tag}.pt"), weights_only=False)
        forms[tag] = (s1.get("batching") or {"form": "equal"})["form"]
        assert s1["fit_close"].shape[0] == len(rows) and s1["eval_close"].shape[0] == E, (tag, s1["fit_close"].shape,
                                                                                         s1["eval_close"].shape, len(rows), E)
        assert not span_on or s1["evaltok_close"].shape[0] == len(eval_tok), (tag, "the stage-1 file predates the matched span")
        for b in cfg["blocks"]:
            bi = s1["blocks"].index(b)
            for conv in (cfg["convs"] if tag in cfg["her"] else ["close"]):
                relays[f"{tag}|b{b}|{conv}"] = (s1[f"fit_{conv}"][:, bi].clone(), s1[f"eval_{conv}"][:, bi].clone(),
                                                s1[f"evaltok_{conv}"][:, bi].clone() if span_on else None)
        del s1
    assert len(set(forms.values())) == 1, (f"stage-1 files of different batch forms {forms}: under bf16 they round differently; "
                                           "remake them in one form (ALEPHLLM_DIFFUSION_S1_BATCHING)")
    if cfg["floors"]:
        tok_txt = ([evals[ei]["text"][a:e] for ei, a, e in zip(eval_tok[:, 0].tolist(), eval_tok[:, 5].tolist(),
                                                                eval_tok[:, 6].tolist())] if span_on else None)
        relays["spelling"] = (spelling(span_txt), spelling(phrase_txt), spelling(tok_txt) if span_on else None)
        relays["qwen_static"] = (emb[rows[:, 4]], torch.stack([emb[e["phrase_tok_ids"]].mean(0) for e in evals]),
                                 emb[eval_tok[:, 4]] if span_on else None)
    # RELAYS OUTSIDE THE REGISTERED GRID (outside_relays): STITCH_EXTRA_RELAYS adds those relays
    # after every registered one, for a selection of their own cells that reads, with the same functions, a cell the grid does not
    # hold (the untrained control in the last-byte convention when the pick is last-byte); no registered cell is computed beside it
    for x in extra:
        tag, blk, conv = x.split("|")
        assert x not in relays and tag in cfg["her"] + cfg["controls"] and int(blk[1:]) in cfg["blocks"], x
        s1 = torch.load(os.path.join(OUT_DIR, f"s1_{tag}.pt"), weights_only=False)
        assert (s1.get("batching") or {"form": "equal"})["form"] in set(forms.values()), (x, "a stage-1 file of another batch form")
        bi = s1["blocks"].index(int(blk[1:]))
        relays[x] = (s1[f"fit_{conv}"][:, bi].clone(), s1[f"eval_{conv}"][:, bi].clone(),
                     s1[f"evaltok_{conv}"][:, bi].clone() if span_on else None)
        del s1
    print(f"[stitch s2] {len(relays)} relays x {len(depths)} depths x fits {cfg['fits']}"
          + (f"; RELAYS OUTSIDE THE REGISTERED GRID (read with the same functions, labelled so): {extra}" if extra else "")
          + mem_note(), flush=True)

    # 4. THE ONE-SLOT FORM (the secondary column when the matched span runs)
    scene_of = [e["scene"] for e in evals]
    scenes = sorted(set(scene_of))
    by_scene = {sc: [i for i, s in enumerate(scene_of) if s == sc] for sc in scenes}
    offset = torch.tensor([bases[s]["q_m"] - bases[s]["slot_n"] for s in scene_of])

    def scene_groups(split=1, per_pass=None):
        """The passes: `per_pass` whole scenes each (SCENES_PER_PASS by default; scene order kept), each pass cut into `split`
        interleaved parts (a second batch shape, for the fp32 repeat floor). per_pass 1 = the earlier form, one scene per pass."""
        k = per_pass or SCENES_PER_PASS
        sc_lists = list(by_scene.values())
        packs = [sum(sc_lists[a:a + k], []) for a in range(0, len(sc_lists), k)]
        return [p[h::split] for p in packs for h in range(split)]

    def run_slot(vecs_at, qm=None, adm=None, maps=True, split=1, per_pass=None):
        """vecs_at: None (the empty slot) or (depth, [E, D] vectors). Returns the slot's final states, the module's output at the
        slot question and at the other T5 positions, and (maps) hit, leak, the slot's keys and values. split = the batches per
        pass (2: a second batch shape, for the fp32 repeat floor); per_pass = whole scenes per pass (scene_groups)."""
        qm = qwen if qm is None else qm
        adm = ad if adm is None else adm
        res = {"maps": maps, "final": torch.zeros(E, 1024), "aq": torch.zeros(E, 1024), "other": [None] * E,
               "hit": torch.zeros(E), "leak": torch.zeros(E), "keys": [None] * E, "vals": [None] * E}
        for idx in scene_groups(split, per_pass):
            bss = [bases[scene_of[i]] for i in idx]
            pos = torch.tensor([b["slot_n"] for b in bss])
            patch = None if vecs_at is None else (vecs_at[0], pos, vecs_at[1][idx])
            if len({scene_of[i] for i in idx}) == 1:           # one scene: every row the same prompt, no padding (as before)
                bs = bss[0]
                out, _ = qwen_run(qm, [bs["qwen_ids"]] * len(idx), device, patch=patch)
                smb = torch.ones(len(idx), len(bs["qwen_ids"]))
                t5b = torch.tensor([bs["t5_ids"]] * len(idx))
                t5m = torch.ones_like(t5b)
            else:                                              # several scenes: right-padded rows, each its own masks
                q_list = [b["qwen_ids"] for b in bss]
                out, _ = qwen_run(qm, q_list, device, patch=patch)
                smb = pad_batch(q_list)[1]
                t5b, t5m = pad_batch([b["t5_ids"] for b in bss])
            y = adapter_module(adm, out, smb, t5b, t5m, device)
            rows_d = torch.arange(len(idx), device=out.device)
            res["final"][idx] = out[rows_d, pos.to(out.device)].cpu()
            res["aq"][idx] = y[rows_d, torch.tensor([b["q_m"] for b in bss], device=y.device)].cpu()
            yc = y.cpu()                                       # one copy per pass; the rows are sliced on the CPU
            for j, (i, b) in enumerate(zip(idx, bss)):
                res["other"][i] = torch.cat([yc[j, :b["q_m"]], yc[j, b["q_m"] + 1:len(b["t5_ids"])]])
            if maps:
                _, h, lk, k, v = adapter_maps(adm, out, smb, t5b, t5m, pos, device)
                res["hit"][idx], res["leak"][idx] = h.cpu(), lk.cpu()
                kc, vc = k.cpu(), v.cpu()
                for j, i in enumerate(idx):
                    res["keys"][i], res["vals"][i] = kc[j], vc[j]
        if maps:
            res["keys"], res["vals"] = torch.stack(res["keys"]), torch.stack(res["vals"])
        return res

    def centred(e):
        """Each effect minus its own arm's leave-one-out mean over the phrases on the same scene (the phrase itself excluded)."""
        out = torch.zeros_like(e)
        for sc, idx in by_scene.items():
            s = e[idx].sum(0)
            out[idx] = e[idx] - (s - e[idx]) / (len(idx) - 1)
        return out

    empty = run_slot(None)
    scene_check = {"scenes_per_pass": SCENES_PER_PASS}

    def scene_batch_check(form, new, ref, keys):
        """THE SCENE-BATCH CHECK: a run's outputs with SCENES_PER_PASS scenes a pass against one scene a pass; tier 1 bit-identical,
        tier 2 under ten times the batch-shape floor, else the run stops."""
        tier, dmax = maps_equivalence([new[k] for k in keys], [ref[k] for k in keys], 10.0 * batch_floor)
        scene_check[form] = {"tier": tier, "max_relative_difference": dmax}
        print(f"[stitch s2] THE SCENE-BATCH CHECK ({form}, {SCENES_PER_PASS} scenes a pass against one): "
              + ("bit-identical (tier 1)" if tier == 1 else f"equal within the registered repeat floor (tier 2; max relative "
                 f"difference {dmax:.1e} under {10.0 * batch_floor:.1e})" if tier == 2 else
                 f"DIFFERENT (max relative difference {dmax:.1e}): the run stops"), flush=True)
        assert tier is not None, f"the scene batches differ from one scene a pass ({form}: {dmax:.2e} relative)"
    if SCENES_PER_PASS > 1:
        scene_batch_check("the empty slot", empty, run_slot(None, per_pass=1),
                          ("final", "aq", "other", "hit", "leak", "keys", "vals"))
    ceil_run = run_slot(("28post", selfv["28post"])) if "28post" in depths else None
    e_ceil = ceil_run["aq"] - empty["aq"]
    e_ceil_c = centred(e_ceil)
    held = torch.tensor([e["held_phrase"] for e in evals])
    readm = torch.tensor([e["read"] for e in evals])
    single = torch.tensor([e["n_tok"] == 1 for e in evals])
    phr = torch.tensor([e["phrase"] for e in evals])
    percap = {"ceiling": torch.ones(E)}                 # every arm's per-caption centred primary (the margin rule's pairs)

    # 5. THE MATCHED SPAN (the form of record)
    if span_on:
        NT = len(eval_tok)
        tok_of: dict = {}                               # eval caption -> [(token row, Qwen3 position)]
        for r, (ei, k, *_rest) in enumerate(eval_tok.tolist()):
            tok_of.setdefault(ei, []).append((r, k))
        assert all([k for _, k in tok_of[i]] == evals[i]["phrase_toks"] for i in range(E)), "the span rows do not match stage 0"
        pol = [s0["phrases"][e["phrase"]]["class"] for e in evals]
        cls = [e["cls"] for e in evals]
        by_sc_cls: dict = {}
        for i, e in enumerate(evals):
            by_sc_cls.setdefault((e["scene"], e["cls"]), []).append(i)
        self_phrase = torch.tensor([e["filler_is_phrase"] for e in evals])
        strong, band_lo, band_hi = strong_band()            # the reading heads' columns and the real-word logit band
        n_blk, n_head = len(ad.blocks), ad.blocks[0].cross_attn.n_heads
        sb_idx, sh_idx = torch.tensor([b for b, _ in strong]), torch.tensor([h for _, h in strong])
        og: dict = {}                                       # the off-diagonal partners (same scene, class and filler)
        for i, e in enumerate(evals):
            if not e["filler_is_phrase"]:
                og.setdefault((e["scene"], e["cls"], e["filler"]), []).append(i)
        off_groups = [g for g in og.values() if len(g) >= 2]
        # THE MAPS CHECK: the batched copy-out against the frozen per-row function on one scene's ceiling batch, in every run;
        # tier 1 = bit-identical, tier 2 = the largest relative difference under ten times the batch-shape floor; else stop
        idx0 = by_scene[sorted(by_scene)[0]]
        q0 = [evals[i]["qwen_ids"] for i in idx0]
        sm0 = pad_batch(q0)[1]
        t0_ids, tm0 = pad_batch([evals[i]["t5_ids"] for i in idx0])
        out0, _ = qwen_run(qwen, q0, device)
        qm0 = torch.zeros(t0_ids.shape, dtype=torch.bool)
        am0 = torch.zeros(sm0.shape, dtype=torch.bool)
        for j, i in enumerate(idx0):
            qm0[j, evals[i]["t5_span"]] = True
            am0[j, evals[i]["phrase_toks"]] = True
        maps_tier, maps_dmax = maps_equivalence(adapter_maps_span(ad, out0, sm0, t0_ids, tm0, qm0, am0, device, strong),
                                                adapter_maps_span_rowwise(ad, out0, sm0, t0_ids, tm0, qm0, am0, device, strong),
                                                10.0 * batch_floor)
        print(f"[stitch s2] THE MAPS CHECK (one scene, {len(idx0)} captions, the batched copy-out against the frozen per-row one): "
              + ("bit-identical (tier 1)" if maps_tier == 1 else f"equal within the registered repeat floor (tier 2; max relative "
                 f"difference {maps_dmax:.1e} under {10.0 * batch_floor:.1e})" if maps_tier == 2 else
                 f"DIFFERENT (max relative difference {maps_dmax:.1e}): the run stops"), flush=True)
        assert maps_tier is not None, f"the batched maps differ from the frozen ones ({maps_dmax:.2e} relative)"

        def class_centred(e):
            """Each effect minus its own arm's leave-one-out mean over the phrases of the SAME CLASS on the same scene, the up half and
            the down half weighted equally, neutral phrases out of the reference."""
            out = torch.zeros_like(e)
            for (_sc, _c), idx in by_sc_cls.items():
                up = [i for i in idx if pol[i] == "up"]
                dn = [i for i in idx if pol[i] == "down"]
                su, sd = e[up].sum(0), e[dn].sum(0)
                for i in idx:
                    mu_up = (su - e[i]) / (len(up) - 1) if pol[i] == "up" else su / len(up)
                    mu_dn = (sd - e[i]) / (len(dn) - 1) if pol[i] == "down" else sd / len(dn)
                    out[i] = e[i] - 0.5 * (mu_up + mu_dn)
            return out

        def run_span(vecs_at=None, q_ids="qwen_ids", t_ids="t5_ids", t_span="t5_span", qm=None, adm=None, maps=False, split=1,
                     collect_depths=None, base=None, keep_other=False, check_equal=None, per_pass=None):
            """The matched-span runs, whole scenes per pass (scene_groups; split: batches per pass; per_pass: scenes per pass).
            vecs_at = None (unpatched) or (depth, [NT, D] vectors, one per phrase-token row). q_ids / t_ids / t_span name stage 0's fields (the phrase caption's or the filler
            caption's). Returns the module's output pooled over the question pieces [E, D], the final states at the phrase tokens,
            the leak into the other T5 positions against `base` (another run of this function, kept with keep_other), and (maps)
            hit, leak, the answer tokens' keys and values; collect_depths -> Qwen3's residuals at the phrase tokens [NT, D] per
            depth; check_equal = depth -> asserts the patched Qwen3 output equals the unpatched one bit for bit."""
            qm = qwen if qm is None else qm
            adm = ad if adm is None else adm
            res = {"maps": maps, "aq": torch.zeros(E, 1024), "final": [None] * E, "other": [None] * E,
                   "leak_task": torch.zeros(E), "hit": torch.zeros(E), "leak": torch.zeros(E), "keys": [None] * E,
                   "vals": [None] * E, "collected": {d: torch.zeros(NT, 1024) for d in (collect_depths or [])}, "copy_gap": 0.0,
                   "hit_h": torch.zeros(E, n_blk, n_head), "leak_h": torch.zeros(E, n_blk, n_head), "slog": [None] * E}
            for idx in scene_groups(split, per_pass):
                q_list = [evals[i][q_ids] for i in idx]
                sm = pad_batch(q_list)[1]
                tids, tm = pad_batch([evals[i][t_ids] for i in idx])
                prow = torch.tensor([j for j, i in enumerate(idx) for _ in tok_of[i]])
                ppos = torch.tensor([k for i in idx for _, k in tok_of[i]])
                prr = torch.tensor([r for i in idx for r, _ in tok_of[i]])
                patch = None if vecs_at is None else (vecs_at[0], prow, ppos, vecs_at[1][prr])
                out, got = qwen_run(qm, q_list, device, patch=patch, collect_all=collect_depths)
                if check_equal is not None:
                    o_ref, _ = qwen_run(qm, q_list, device)
                    assert torch.equal(out, o_ref), f"the span self-patch at depth {check_equal} is not the ceiling exactly"
                for d in (collect_depths or []):
                    res["collected"][d][prr] = got[d][prow, ppos]
                y = adapter_module(adm, out, sm, tids, tm, device)
                # the same per-row means on the card, then one copy per pass; the gathers are made on the CPU (the same values)
                aq = torch.stack([y[j, evals[i][t_span]].mean(0) for j, i in enumerate(idx)]).cpu()
                oc, yc = out.cpu(), y.cpu()
                for j, i in enumerate(idx):
                    sp = evals[i][t_span]
                    res["aq"][i] = aq[j]
                    res["final"][i] = oc[j, evals[i]["phrase_toks"]]
                    oth = [t for t in range(len(evals[i][t_ids])) if t not in sp]
                    o = yc[j, oth]
                    if keep_other:
                        res["other"][i] = o
                    if base is not None:
                        res["leak_task"][i] = (o - base["other"][i]).norm(dim=-1).mean()
                if maps:
                    qmask = torch.zeros(tids.shape, dtype=torch.bool)
                    amask = torch.zeros(sm.shape, dtype=torch.bool)
                    for j, i in enumerate(idx):
                        qmask[j, evals[i][t_span]] = True
                        amask[j, evals[i]["phrase_toks"]] = True
                    maps_fn = adapter_maps_span_rowwise if MAPS_ROWWISE else adapter_maps_span
                    y2, h, lk, kk, vv, hh, lh, sl = maps_fn(adm, out, sm, tids, tm, qmask, amask, device, strong)
                    valid = tm.bool().to(y.device)
                    res["copy_gap"] = max(res["copy_gap"], float((y - y2)[valid].abs().max() / y[valid].abs().max()))
                    res["hit"][idx], res["leak"][idx] = h, lk
                    res["hit_h"][idx], res["leak_h"][idx] = hh, lh
                    for j, i in enumerate(idx):
                        res["keys"][i], res["vals"][i], res["slog"][i] = kk[j], vv[j], sl[j]
            return res

        base_sp = run_span(q_ids="filler_qwen_ids", keep_other=True, maps=True)
        if SCENES_PER_PASS > 1:
            scene_batch_check("the filler span", base_sp, run_span(q_ids="filler_qwen_ids", keep_other=True, maps=True, per_pass=1),
                              ("aq", "final", "other", "hit", "leak", "keys", "vals", "hit_h", "leak_h", "slog"))
        ceil_sp = run_span(maps=True, collect_depths=depths, base=base_sp)
        fband = torch.tensor([band_shares(a, c, band_lo, band_hi) for a, c in zip(base_sp["slog"], ceil_sp["slog"])])   # [E, 3]
        fband[self_phrase] = float("nan")                   # the filler's own band shares (the reference beside every arm)
        assert ceil_sp["copy_gap"] < 1e-4, f"the written-out span adapter differs from the module ({ceil_sp['copy_gap']:.2e})"
        for d in depths:                                # THE SPAN SELF-PATCH INVARIANCE, every phrase, every depth
            run_span((d, ceil_sp["collected"][d]), check_equal=d)
        ff_sp = run_span(q_ids="filler_qwen_ids", t_ids="filler_t5_ids", t_span="filler_t5_span")
        e_ceil_sp = ceil_sp["aq"] - base_sp["aq"]
        e_ceil_sp_c = class_centred(e_ceil_sp)
        # THE MOOD AXIS of the ceiling, per caption: the up mean minus the down mean of the ceiling's centred effects over the
        # caption's class on its scene, the phrase itself left out (unit vector; NaN for neutral phrases)
        sgn = torch.tensor([1.0 if p == "up" else -1.0 if p == "down" else float("nan") for p in pol])
        U = torch.full((E, 1024), float("nan"))
        for (_sc, _c), idx in by_sc_cls.items():
            up = [i for i in idx if pol[i] == "up"]
            dn = [i for i in idx if pol[i] == "down"]
            su, sd = e_ceil_sp_c[up].sum(0), e_ceil_sp_c[dn].sum(0)
            for i in up + dn:
                mu = (su - e_ceil_sp_c[i]) / (len(up) - 1) if pol[i] == "up" else su / len(up)
                md = (sd - e_ceil_sp_c[i]) / (len(dn) - 1) if pol[i] == "down" else sd / len(dn)
                U[i] = (mu - md) / (mu - md).norm().clamp(min=1e-12)
        pc_mood = sgn * (e_ceil_sp_c * U).sum(-1)           # the ceiling's own signed projection on its leave-one-out mood axis
        cc_mood = sgn * cosine(e_ceil_sp_c, U)
        whole = (ceil_sp["aq"] - ff_sp["aq"]).norm(dim=-1)
        ans_share = e_ceil_sp.norm(dim=-1) / whole.clamp(min=1e-12)
        ans_share[self_phrase] = float("nan")
        share_ph = {}
        for j in sorted(set(phr.tolist())):
            v = ans_share[(phr == j) & ~torch.isnan(ans_share)]
            share_ph[j] = float(v.mean()) if len(v) else float("nan")
        carried = torch.tensor([share_ph[e["phrase"]] >= 0.5 for e in evals])
        updown = torch.tensor([p in ("up", "down") for p in pol])
        cmask = {c: torch.tensor([x == c for x in cls]) for c in CLASSES}
        span_sets = {"record": cmask["FRAGMENT"] & held & readm,
                     "fragment_seen": cmask["FRAGMENT"] & ~held & readm,
                     "morpheme_held": cmask["MORPHEME"] & held & readm, "morpheme_seen": cmask["MORPHEME"] & ~held & readm,
                     "whole_held": cmask["WHOLE"] & held & readm, "whole_seen": cmask["WHOLE"] & ~held & readm,
                     "all_held": held & readm, "all_seen": ~held & readm,
                     "single_held": held & readm & single, "single_seen": ~held & readm & single,
                     "carried_held": held & readm & updown & carried}
        offset_sp = torch.tensor([e["t5_span"][0] - e["phrase_toks"][0] for e in evals])
        n_rec = len(set(phr[span_sets["record"]].tolist()))
        print(f"[stitch s2] THE MATCHED SPAN: the self-patch reproduces the ceiling bit for bit at every depth ({len(depths)} depths, "
              f"{E} captions); the written-out adapter matches the module on padded batches ({ceil_sp['copy_gap']:.1e} relative); "
              f"{NT} phrase tokens; the read of record = FRAGMENT-SPLIT held-out, {n_rec} phrases "
              f"({int(span_sets['record'].sum())} captions); effects pooled over each phrase's own T5 pieces before any cosine",
              flush=True)
        print("[stitch s2] EACH PHRASE ON THE STOCK MODEL (the mean over its scenes): |ceiling effect| raw and class-centred, and "
              "THE ANSWER SHARE (printed, never used to move a phrase):", flush=True)
        order = sorted(set(phr.tolist()), key=lambda j: (CLASSES.index(s0["phrases"][j]["cls"]), s0["phrases"][j]["split"] != "heldout",
                                                         s0["phrases"][j]["class"], j))
        per_phrase = {}
        for j in order:
            mk = phr == j
            p = s0["phrases"][j]
            per_phrase[p["text"]] = {"class": p["cls"], "split": p["split"], "polarity": p["class"],
                                     "ceiling_effect": float(e_ceil_sp[mk].norm(dim=-1).mean()),
                                     "ceiling_effect_centred": float(e_ceil_sp_c[mk].norm(dim=-1).mean()),
                                     "answer_share": share_ph[j]}
            r = per_phrase[p["text"]]
            print(f"[stitch s2]   {p['cls']:8s} {p['split']:7s} {p['class']:7s} {p['text']!r:32s} |e| {r['ceiling_effect']:.4f} "
                  f"centred {r['ceiling_effect_centred']:.4f}  answer share "
                  f"{'n/a (its filler is itself)' if math.isnan(r['answer_share']) else format(r['answer_share'], '.3f')}", flush=True)
        for c in CLASSES:
            mk = cmask[c]
            print(f"[stitch s2]   class {c}: mean |ceiling effect| {float(e_ceil_sp[mk].norm(dim=-1).mean()):.4f}, centred "
                  f"{float(e_ceil_sp_c[mk].norm(dim=-1).mean()):.4f}, answer share (phrase means) "
                  f"{sum(share_ph[j] for j in set(phr[mk].tolist()) if not math.isnan(share_ph[j])) / max(1, sum(not math.isnan(share_ph[j]) for j in set(phr[mk].tolist()))):.3f}",
                  flush=True)
        print(f"[stitch s2]   the sensitivity set (held-out up/down phrases with a measured share >= .5): "
              f"{len(set(phr[span_sets['carried_held']].tolist()))} phrases: "
              f"{sorted(s0['phrases'][j]['text'] for j in set(phr[span_sets['carried_held']].tolist()))}", flush=True)
        percap_sp = {"ceiling": torch.ones(E)}
        mood_sp = {}                                    # key -> (the per-caption mood projection, the mood cosine): to disk per cell

    # 6. PRECISION: what half precision does (the whole chain in bf16 against fp32, both centred), per form
    qwen_bf, ad_bf = copy.deepcopy(qwen).to(torch.bfloat16), copy.deepcopy(ad).to(torch.bfloat16)
    ceil_bf = torch.zeros(E, 1024)
    for e0 in range(0, E, 64):
        grp = evals[e0:e0 + 64]
        _, g = qwen_run(qwen_bf, [e["qwen_ids"] for e in grp], device, collect={"28post": torch.tensor([e["t_ceil"] for e in grp])})
        ceil_bf[e0:e0 + len(grp)] = g["28post"].cpu()
    e_bf = (run_slot(("28post", ceil_bf), qwen_bf, ad_bf, maps=False)["aq"]
            - run_slot(None, qwen_bf, ad_bf, maps=False)["aq"])
    if span_on:
        e_bf_sp = run_span(qm=qwen_bf, adm=ad_bf)["aq"] - run_span(q_ids="filler_qwen_ids", qm=qwen_bf, adm=ad_bf)["aq"]
    del qwen_bf, ad_bf
    prec_cos = cosine(centred(e_bf), e_ceil_c)
    rel_size = e_ceil_c.norm(dim=-1) / ceil_run["aq"].norm(dim=-1)
    med = float(rel_size.median())
    precision = {"bf16_vs_fp32_centred_cos_min": float(prec_cos.min()), "bf16_vs_fp32_centred_cos_mean": float(prec_cos.mean()),
                 "bf16_vs_fp32_centred_cos_min_read": float(prec_cos[readm].min()),
                 "centred_effect_over_output_mean": float(rel_size.mean()),
                 "centred_effect_over_output_min": float(rel_size.min()), "bf16_retired": bool(prec_cos.min() < 0.99),
                 # the per-caption distribution of |centred ceiling effect| / |adapter output|
                 "centred_effect_over_output_median": med, "centred_effect_over_output_p10": float(rel_size.quantile(0.10)),
                 "captions_under_a_tenth_of_the_median": int((rel_size < 0.1 * med).sum())}
    # the fp32 instrument's own repeat floor (the registered definition, as corrected): the same ceiling's centred effect at a
    # second batch shape (each scene in two batches), one minus the centred cosine; the bar is ten times its mean
    e_rep = run_slot(("28post", selfv["28post"]), maps=False, split=2)["aq"] - run_slot(None, maps=False, split=2)["aq"]
    rep = 1.0 - cosine(centred(e_rep), e_ceil_c)
    precision.update({"fp32_repeat_one_minus_centred_cos_mean": float(rep.mean()),
                      "fp32_repeat_one_minus_centred_cos_max": float(rep.max())})
    print(f"[stitch s2] WHAT HALF PRECISION DOES TO A SLOT'S PHRASE-SPECIFIC WRITE: bf16 vs fp32 centred ceiling cos mean "
          f"{precision['bf16_vs_fp32_centred_cos_mean']:.4f}, per-caption min {precision['bf16_vs_fp32_centred_cos_min']:.4f} (read "
          f"set {precision['bf16_vs_fp32_centred_cos_min_read']:.4f}){' -> bf16 RETIRED from the instrument' if precision['bf16_retired'] else ''}"
          f"; |centred effect| / |output| mean {precision['centred_effect_over_output_mean']:.4f}, median {med:.4f}, 10th "
          f"percentile {precision['centred_effect_over_output_p10']:.4f}, {precision['captions_under_a_tenth_of_the_median']} "
          f"captions under a tenth of the median; THE FP32 REPEAT FLOOR (two batch shapes): 1 - centred cos mean "
          f"{float(rep.mean()):.2e}, max {float(rep.max()):.2e} (the bar: ten times the mean)", flush=True)
    if span_on:
        prec_sp = cosine(class_centred(e_bf_sp), e_ceil_sp_c)
        rel_sp = e_ceil_sp_c.norm(dim=-1) / ceil_sp["aq"].norm(dim=-1)
        ok = ~self_phrase
        med_sp = float(rel_sp[ok].median())
        e_rep_sp = run_span(split=2)["aq"] - run_span(q_ids="filler_qwen_ids", split=2)["aq"]
        rep_sp = 1.0 - cosine(class_centred(e_rep_sp), e_ceil_sp_c)
        precision_sp = {"bf16_vs_fp32_centred_cos_mean": float(prec_sp[ok].mean()), "bf16_vs_fp32_centred_cos_min": float(prec_sp[ok].min()),
                        "bf16_vs_fp32_centred_cos_min_record": float(prec_sp[span_sets["record"]].min()),
                        "centred_effect_over_output_median": med_sp, "centred_effect_over_output_p10": float(rel_sp[ok].quantile(0.10)),
                        "captions_under_a_tenth_of_the_median": int((rel_sp[ok] < 0.1 * med_sp).sum()),
                        "fp32_repeat_one_minus_centred_cos_mean": float(rep_sp[ok].mean()),
                        "fp32_repeat_one_minus_centred_cos_max": float(rep_sp[ok].max())}
        print(f"[stitch s2] WHAT HALF PRECISION DOES TO A PHRASE'S ANSWER WRITE (the matched span): bf16 vs fp32 class-centred "
              f"ceiling cos mean {precision_sp['bf16_vs_fp32_centred_cos_mean']:.4f}, per-caption min "
              f"{precision_sp['bf16_vs_fp32_centred_cos_min']:.4f} (record {precision_sp['bf16_vs_fp32_centred_cos_min_record']:.4f}); "
              f"|centred effect| / |output| median {med_sp:.4f}, 10th percentile {precision_sp['centred_effect_over_output_p10']:.4f}, "
              f"{precision_sp['captions_under_a_tenth_of_the_median']} under a tenth of the median; THE FP32 REPEAT FLOOR: 1 - "
              f"centred cos mean {precision_sp['fp32_repeat_one_minus_centred_cos_mean']:.2e}, max "
              f"{precision_sp['fp32_repeat_one_minus_centred_cos_max']:.2e} (the 'neutral' captions, whose effect is zero by "
              f"construction, left out of these four numbers)", flush=True)

    def reads(run, slot_in=None, slot_true=None, key=None):
        e_r = run["aq"] - empty["aq"]
        e_rc = centred(e_r)
        prim_c = cosine(e_rc, e_ceil_c)
        if key is not None:
            percap[key] = prim_c.clone()
        prim = cosine(e_r, e_ceil)
        share = e_rc.norm(dim=-1) / e_ceil_c.norm(dim=-1).clamp(min=1e-9)
        fin_cos = cosine(run["final"], ceil_run["final"])
        fin_rel = (run["final"] - ceil_run["final"]).norm(dim=-1) / ceil_run["final"].norm(dim=-1)
        leak_task = torch.stack([(o - eo).norm(dim=-1).mean() for o, eo in zip(run["other"], empty["other"])])
        if run["maps"]:
            kcos = cosine(run["keys"], ceil_run["keys"]).mean(-1).mean(-1)                  # heads, then blocks
            vrat = (run["vals"].norm(dim=-1) / ceil_run["vals"].norm(dim=-1).clamp(min=1e-6)).mean(-1).mean(-1)
        rank = []
        for sc, idx in by_scene.items():                # among the held-out phrases on the scene: is THIS phrase recovered?
            hi = [i for i in idx if bool(held[i])]
            if len(hi) < 2:
                continue
            C = cosine(e_rc[hi][:, None], e_ceil_c[hi][None])
            rank += [float((C[j] > C[j, j]).sum()) + 1 for j in range(len(hi))]
        shared = []                                     # what centring removes: the slot's constant tint, per scene
        for sc, idx in by_scene.items():
            mr, mc = e_r[idx].mean(0), e_ceil[idx].mean(0)
            shared.append((float(cosine(mr, mc)), float(mr.norm() / mc.norm().clamp(min=1e-9))))

        def m(v, mask, needs_maps=False):
            if needs_maps and not run["maps"]:
                return None
            return float(v[mask].mean()) if bool(mask.any()) else None

        out = {}
        for split, mk in (("held", held & readm), ("seen", ~held & readm)):
            for sub, mk2 in (("all", mk), ("single", mk & single), ("multi", mk & ~single)):
                out[f"{split}_{sub}"] = {"primary_centred": m(prim_c, mk2),
                                         "primary_centred_ci": list(cluster_ci(prim_c[mk2], phr[mk2])) if bool(mk2.any())
                                         else [None, None], "n_phrases": len(set(phr[mk2].tolist())),
                                         # the pooled companion: captions weighed by how much effect they carry
                                         "primary_pooled": pooled_cos(e_rc[mk2], e_ceil_c[mk2]) if bool(mk2.any()) else None,
                                         "primary_raw": m(prim, mk2), "share": m(share, mk2),
                                         "final_cos": m(fin_cos, mk2), "final_rel": m(fin_rel, mk2),
                                         "leak_task": m(leak_task, mk2), "hit": m(run["hit"], mk2, True),
                                         "leak": m(run["leak"], mk2, True), "key_cos": m(kcos, mk2, True) if run["maps"] else None,
                                         "value_ratio": m(vrat, mk2, True) if run["maps"] else None, "n": int(mk2.sum())}
        out["held_by_offset"] = {str(o): {"primary_centred": m(prim_c, held & readm & (offset == o)),
                                          "hit": m(run["hit"], held & readm & (offset == o), True),
                                          "n": int((held & readm & (offset == o)).sum())}
                                 for o in sorted(set(offset.tolist()))}
        out["held_rank_mean"] = sum(rank) / len(rank) if rank else None
        out["held_rank_chance"] = (sum(len([i for i in idx if bool(held[i])]) for idx in by_scene.values()) / len(by_scene) + 1) / 2
        out["shared_shift_cos"] = sum(s[0] for s in shared) / len(shared)
        out["shared_shift_size_ratio"] = sum(s[1] for s in shared) / len(shared)
        out["shared_shift_per_scene"] = {str(sc): s for sc, s in zip(by_scene, shared)}
        if slot_in is not None and slot_true is not None:
            mu = slot_true.mean(0)
            out["shrink"] = float(((slot_in - mu).norm(dim=-1) / (slot_true - mu).norm(dim=-1).clamp(min=1e-6)).mean())
        return out

    def reads_span(run, key=None, tok_in=None, tok_true=None):
        """The reads on the matched span, per read set (the record first); the same columns as the one-slot form's."""
        e_r = run["aq"] - base_sp["aq"]
        e_rc = class_centred(e_r)
        prim_c = cosine(e_rc, e_ceil_sp_c)
        if key is not None:
            percap_sp[key] = prim_c.clone()
        prim = cosine(e_r, e_ceil_sp)
        share = e_rc.norm(dim=-1) / e_ceil_sp_c.norm(dim=-1).clamp(min=1e-9)
        fin_cos = torch.tensor([float(cosine(a, b).mean()) for a, b in zip(run["final"], ceil_sp["final"])])
        fin_rel = torch.tensor([float(((a - b).norm(dim=-1) / b.norm(dim=-1)).mean()) for a, b in zip(run["final"], ceil_sp["final"])])
        if run["maps"]:
            kcos = torch.tensor([float(cosine(a, b).mean()) for a, b in zip(run["keys"], ceil_sp["keys"])])   # tokens, blocks, heads
            vrat = torch.tensor([float((a.norm(dim=-1) / b.norm(dim=-1).clamp(min=1e-6)).mean()) for a, b in
                                 zip(run["vals"], ceil_sp["vals"])])
            hit_s = run["hit_h"][:, sb_idx, sh_idx].mean(-1)                                   # the eight strong heads
            leak_s = run["leak_h"][:, sb_idx, sh_idx].mean(-1)
            band = torch.tensor([band_shares(a, c, band_lo, band_hi) for a, c in zip(run["slog"], ceil_sp["slog"])])
        offd = torch.full((E,), float("nan"))           # against the ceiling of a DIFFERENT phrase (same class, filler, scene)
        offs, offo = torch.full((E,), float("nan")), torch.full((E,), float("nan"))     # split by the partner's mood
        for g in off_groups:
            C = cosine(e_rc[g][:, None], e_ceil_sp_c[g][None])
            offd[g] = (C.sum(1) - C.diag()) / (len(g) - 1)
            pg = [pol[i] for i in g]
            for a, i in enumerate(g):
                same = [b for b in range(len(g)) if b != a and pg[b] == pg[a]]
                opp = [b for b in range(len(g)) if {pg[a], pg[b]} == {"up", "down"}]
                if same:
                    offs[i] = C[a, same].mean()
                if opp:
                    offo[i] = C[a, opp].mean()
        rank = []                                       # among the record's phrases on the scene: is THIS phrase recovered?
        for sc, idx in by_scene.items():
            hi = [i for i in idx if bool(span_sets["record"][i])]
            if len(hi) < 2:
                continue
            C = cosine(e_rc[hi][:, None], e_ceil_sp_c[hi][None])
            rank += [float((C[j] > C[j, j]).sum()) + 1 for j in range(len(hi))]
        shared, shared_rec = [], []                     # what centring removes, per scene: over all phrases, over the record class
        for sc, idx in by_scene.items():
            for lst, sel in ((shared, idx), (shared_rec, [i for i in idx if cls[i] == "FRAGMENT"])):
                mr, mc = e_r[sel].mean(0), e_ceil_sp[sel].mean(0)
                lst.append((float(cosine(mr, mc)), float(mr.norm() / mc.norm().clamp(min=1e-9))))

        def m(v, mask, needs_maps=False):
            if needs_maps and not run["maps"]:
                return None
            return float(v[mask].mean()) if bool(mask.any()) else None

        out = {}
        word_sig, mood_sig = prim_c - offs, offs - offo
        pr_mood = sgn * (e_rc * U).sum(-1)                 # the relay's signed projection on the ceiling's mood axis
        cr_mood = sgn * cosine(e_rc, U)
        if key is not None:                                # the per-caption mood rows, to disk with the cell
            mood_sp[key] = (pr_mood.clone(), cr_mood.clone())
        for lab, mk in span_sets.items():
            hp = mk & ~torch.isnan(offd)
            gap = prim_c - offd
            nf = mk & ~self_phrase
            hw, hm = mk & ~torch.isnan(word_sig), mk & ~torch.isnan(mood_sig)
            hx = mk & ~torch.isnan(pr_mood)
            den = float(pc_mood[hx].mean()) if bool(hx.any()) else float("nan")
            size_i = pr_mood / den
            ta = []                                     # the interchange's form: the relay's up-minus-down axis against the ceiling's
            for _sc, idx in by_scene.items():
                u = [i for i in idx if bool(mk[i]) and pol[i] == "up"]
                d = [i for i in idx if bool(mk[i]) and pol[i] == "down"]
                if u and d:
                    ar, ac = e_rc[u].mean(0) - e_rc[d].mean(0), e_ceil_sp_c[u].mean(0) - e_ceil_sp_c[d].mean(0)
                    ta.append((float(cosine(ar, ac)), float(ar.norm() / ac.norm().clamp(min=1e-12))))
            out[lab] = {"primary_centred": m(prim_c, mk),
                        "primary_centred_ci": list(cluster_ci(prim_c[mk], phr[mk])) if bool(mk.any()) else [None, None],
                        "n_phrases": len(set(phr[mk].tolist())),
                        "primary_pooled": pooled_cos(e_rc[mk], e_ceil_sp_c[mk]) if bool(mk.any()) else None,
                        "primary_raw": m(prim, mk), "share": m(share, mk), "final_cos": m(fin_cos, mk),
                        "final_rel": m(fin_rel, mk), "leak_task": m(run["leak_task"], mk), "hit": m(run["hit"], mk, True),
                        "leak": m(run["leak"], mk, True), "key_cos": m(kcos, mk, True) if run["maps"] else None,
                        "value_ratio": m(vrat, mk, True) if run["maps"] else None, "n": int(mk.sum()),
                        # guard 1: quote the GAP, not the level, whenever one arm's number is stated alone
                        "offdiag": m(offd, hp), "gap": m(gap, hp),
                        "gap_ci": list(cluster_ci(gap[hp], phr[hp])) if bool(hp.any()) else [None, None],
                        "n_with_partner": int(hp.sum()),
                        # the partner's mood: diagonal minus off-same = THIS WORD beyond its mood; off-same minus
                        # off-opposite = the mood by itself (a relay that delivers the class and not the word: the second large,
                        # the first near zero; the ceiling's own first difference = the word-level detail there is to deliver)
                        "offdiag_same": m(offs, mk & ~torch.isnan(offs)), "offdiag_opposite": m(offo, mk & ~torch.isnan(offo)),
                        "word_signal": m(word_sig, hw), "word_signal_ci": list(cluster_ci(word_sig[hw], phr[hw])) if bool(hw.any())
                        else [None, None], "mood_signal": m(mood_sig, hm),
                        "mood_signal_ci": list(cluster_ci(mood_sig[hm], phr[hm])) if bool(hm.any()) else [None, None],
                        # THE MOOD-AXIS TRANSFER: the primary mostly scores word identity; the mood is what pictures show.
                        # Per caption, the relay's centred effect on the ceiling's leave-one-out mood axis, signed by the phrase's mood:
                        # its cosine (beside the ceiling's own) and its projection over the ceiling's mean projection (the size ratio);
                        # and per scene the interchange's form, the relay's up-minus-down axis against the ceiling's (cosine, size)
                        "mood_cos": m(cr_mood, hx), "mood_cos_ci": list(cluster_ci(cr_mood[hx], phr[hx])) if bool(hx.any())
                        else [None, None], "mood_cos_ceiling": m(cc_mood, hx), "mood_size": m(size_i, hx),
                        "mood_size_ci": list(cluster_ci(size_i[hx], phr[hx])) if bool(hx.any()) else [None, None],
                        "transfer_axis_cos": sum(t[0] for t in ta) / len(ta) if ta else None,
                        "transfer_axis_size": sum(t[1] for t in ta) / len(ta) if ta else None, "transfer_axis_scenes": len(ta),
                        # guard 2: the eight strong reading heads, and the real-word logit band
                        "hit_strong": m(hit_s, mk, True) if run["maps"] else None,
                        "leak_strong": m(leak_s, mk, True) if run["maps"] else None,
                        "band_share_token_head": m(band[:, 0], mk, True) if run["maps"] else None,
                        "band_share_token_all8": m(band[:, 1], mk, True) if run["maps"] else None,
                        "band_share_word": m(band[:, 2], mk, True) if run["maps"] else None,
                        "filler_band_share_token_head": m(fband[:, 0], nf), "filler_band_share_token_all8": m(fband[:, 1], nf),
                        "filler_band_share_word": m(fband[:, 2], nf)}
        rec = span_sets["record"]
        out["record_by_offset"] = {str(o): {"primary_centred": m(prim_c, rec & (offset_sp == o)),
                                            "hit": m(run["hit"], rec & (offset_sp == o), True), "n": int((rec & (offset_sp == o)).sum())}
                                   for o in sorted(set(offset_sp[rec].tolist()))}
        out["record_rank_mean"] = sum(rank) / len(rank) if rank else None
        out["record_rank_chance"] = (n_rec + 1) / 2
        out["shared_shift_cos"] = sum(s[0] for s in shared) / len(shared)
        out["shared_shift_size_ratio"] = sum(s[1] for s in shared) / len(shared)
        out["shared_shift_cos_record_class"] = sum(s[0] for s in shared_rec) / len(shared_rec)
        out["shared_shift_size_ratio_record_class"] = sum(s[1] for s in shared_rec) / len(shared_rec)
        if tok_in is not None and tok_true is not None:
            mu = tok_true.mean(0)
            out["shrink"] = float(((tok_in - mu).norm(dim=-1) / (tok_true - mu).norm(dim=-1).clamp(min=1e-6)).mean())
        return out

    result = {"config": name, "registration": reg, "depths": [str(d) for d in depths], "n_rows": len(rows), "n_eval": E,
              "outside_grid": extra,
              "precision": precision,
              "exactness": {"self_patch_bit_exact": True, "transplant_exact": True, "maps_copy_gap_rel": gap,
                            "batch_shape_floor_rel": batch_floor},
              "empty": {"hit": float(empty["hit"].mean()), "leak": float(empty["leak"].mean())},
              "ceiling": reads(ceil_run), "self": {}, "cells": {}}
    spread = {}                                         # the context yardstick: one phrase's centred ceiling on two scenes
    for label, mk in (("seen_on_held_scenes", ~held & readm), ("held_phrases", held & readm)):
        cs = []
        for j in {evals[i]["phrase"] for i in range(E) if bool(mk[i])}:
            idx = [i for i in range(E) if evals[i]["phrase"] == j and bool(mk[i])]
            if len(idx) > 1:
                C = cosine(e_ceil_c[idx][:, None], e_ceil_c[idx][None])
                cs.append(float((C.sum() - C.diag().sum()) / (len(idx) * (len(idx) - 1))))
        spread[label] = sum(cs) / len(cs) if cs else None
    result["context_spread_centred_cos"] = spread
    if span_on:
        spread_sp = {}
        for label in ("record", "fragment_seen", "all_seen", "all_held"):
            cs = []
            for j in set(phr[span_sets[label]].tolist()):
                idx = [i for i in range(E) if evals[i]["phrase"] == j and bool(span_sets[label][i])]
                if len(idx) > 1:
                    C = cosine(e_ceil_sp_c[idx][:, None], e_ceil_sp_c[idx][None])
                    cs.append(float((C.sum() - C.diag().sum()) / (len(idx) * (len(idx) - 1))))
            spread_sp[label] = sum(cs) / len(cs) if cs else None
        result["span"] = {"precision": precision_sp, "exactness": {"span_self_patch_bit_exact_every_depth": True,
                                                                     "maps_copy_gap_rel": ceil_sp["copy_gap"],
                                                                     "maps_batched_vs_frozen": {"tier": maps_tier,
                                                                                                "max_rel_diff": maps_dmax},
                                                                     "scene_batches": scene_check},
                          "n_record_phrases": n_rec, "classes": {c: int(cmask[c].sum()) for c in CLASSES},
                          "per_phrase": per_phrase, "ceiling": reads_span(ceil_sp), "context_spread_centred_cos": spread_sp,
                          "cells": {}}
    for d in depths:
        result["self"][str(d)] = reads(run_slot((d, selfv[d]), maps=slot_full), key=f"self|k{d}")
    print(f"[stitch s2] empty, ceiling and self-stitch runs done ({time.time() - t0:.0f} s); context spread (centred, one-slot) "
          f"{spread}" + (f"; (the matched span) {spread_sp}" if span_on else "") + mem_note(), flush=True)

    ncell = len(relays) * len(cfg["fits"]) * len(depths)
    done = 0
    ceil_l = result["ceiling"]["held_all"]["leak"]
    if span_on:
        cr = result["span"]["ceiling"]["record"]
        ceil_sp_l, ceil_sp_h, ceil_sp_hs = cr["leak"], cr["hit"], cr["hit_strong"]
        print(f"[stitch s2] THE GUARDS on the ceiling, record set: its own off-diagonal {cr['offdiag']:+.3f} (gap "
              f"{cr['gap']:+.3f}, {cr['n_with_partner']} captions with a partner; same-mood partners {cr['offdiag_same']:+.3f}, "
              f"opposite {cr['offdiag_opposite']:+.3f}: the word-level detail there is to deliver {cr['word_signal']:+.3f}, the mood "
              f"alone {cr['mood_signal']:+.3f}); hit all heads {cr['hit']:.4f}, the eight strong heads "
              f"{cr['hit_strong']:.4f}; leak {cr['leak']:.4f} / {cr['leak_strong']:.4f}; THE NEUTRAL FILLER inside the real-word "
              f"logit band: {cr['filler_band_share_token_head']:.2f} of (token, head) pairs, {cr['filler_band_share_token_all8']:.2f} "
              f"of tokens on all eight, {cr['filler_band_share_word']:.2f} of heads pooled per phrase; strong heads "
              f"{', '.join(f'b{b} h{h}' for b, h in strong)}; bands (lower of the opposite/same 10th percentiles, higher of their "
              f"90th) " + ", ".join(f"[{float(a):+.2f}, {float(b):+.2f}]" for a, b in zip(band_lo, band_hi)), flush=True)
    mem_log, stop_reason, fresh, filed = [], None, 0, 0         # fresh = cells computed by THIS process; filed = cell files written
    midcell = None                                             # (why, where) when the mid-cell stop fired

    def midcell_stop(why, where):
        """THE MID-CELL STOP's record, taken after the exception has unwound: the memory line and the census, printed and kept."""
        release_heap()
        host, rss, priv, dev = host_mem_gb()
        nt, gb, grp = tensor_census()
        mem_log.append({"cell": done + 1, "fresh": fresh, "midcell_stop": where, "host_gb": host, "resident_gb": rss,
                        "committed_gb": priv, "card_reserved_gb": dev, "s": time.time() - t0, "census_tensors": nt,
                        "census_gb": gb, "census_top": grp})
        print(f"[stitch s2] THE MID-CELL STOP in {where}: {why}; host after the unwind {host:.2f} GB (resident {rss:.2f}, "
              f"committed {priv:.2f}, the card's reserved {dev:.2f}); the live-tensor census: {nt} CPU tensors holding {gb:.2f} GB, "
              f"the largest groups: " + "; ".join(f"[*, {', '.join(map(str, g['trailing_shape']))}] {g['dtype']} x{g['n']} "
                                                  f"{g['gb']:.2f} GB" for g in grp), flush=True)
        return f"the mid-cell stop in {where} ({why}): the cell in hand is lost, the finished cells are saved"
    result["resources"] = {"budget": BUDGET, "hard_ceiling_gb": hard, "per_cell": mem_log, "soak": None, "stopped": None}
    cells_dir = os.environ.get("STITCH_CELLS_DIR") or os.path.join(
        OUT_DIR, f"stitch_{name}{RUN_TAG}_cells_{time.strftime('%Y%m%d_%H%M%S', time.localtime(t0))}")
    os.makedirs(cells_dir, exist_ok=True)                      # this run's own folder: only STITCH_RESUME reads an earlier one
    assert not os.listdir(cells_dir), f"the cell folder {cells_dir} is not empty: a run writes only into a folder of its own"
    result["cells_dir"] = cells_dir
    # THE RESTART THAT SKIPS FINISHED CELLS (every stop becomes a pause): STITCH_RESUME = an earlier run's cell
    # folder, or several separated by commas (the two halves of a two-card run; same config); their finished cells are read back
    # (their reads, per-caption rows and printed line) and copied into this run's folder, not recomputed; a relay whose every depth
    # is read back skips its ridge
    rdirs, resume = [d for d in os.environ.get("STITCH_RESUME", "").split(",") if d], {}
    for rdir in rdirs:
        for f in sorted(os.listdir(rdir)):
            row = torch.load(os.path.join(rdir, f), weights_only=False)
            if "slot_reads" in row:                            # cell files without their reads (an older format) are recomputed
                assert row["key"] not in resume, f"{row['key']} is in two resume folders"
                resume[row["key"]] = os.path.join(rdir, f)
            del row
    if rdirs:
        result["resumed_from"] = {"dirs": rdirs, "cells": len(resume)}
    # THE SELECTIONS: STITCH_ONLY_CELLS = the cell keys to compute (the port gate: a few cells checked against
    # another card's printed lines); STITCH_ONLY_FIT = one ridge fit (one card computes the record fit, the other the coco fit; a
    # RESUME run over both folders then computes nothing). Every cell keeps its number in the full grid; a selection run is PARTIAL.
    only_cells = {k for k in os.environ.get("STITCH_ONLY_CELLS", "").split(",") if k}
    only_fit = os.environ.get("STITCH_ONLY_FIT", "")

    def selected(k):
        return (not only_cells or k in only_cells) and (not only_fit or k.split("|")[0] == only_fit)
    selection = (f"only the cells {sorted(only_cells)}" if only_cells else "") + (f"only the {only_fit} fit" if only_fit else "")
    assert not extra or selection, "relays outside the registered grid run only in a selection (never inside a grid of record)"
    todo = sum(1 for fn in cfg["fits"] for rn in relays for d in depths
               if selected(f"{fn}|{rn}|k{d}") and f"{fn}|{rn}|k{d}" not in resume)     # the cells this process will compute
    print(f"[stitch s2] THE GRID: {ncell} cells, {todo} to compute here; each cell's per-caption reads go to {cells_dir} as it "
          f"finishes" + (f"; RESUMING from {', '.join(rdirs)}: {len(resume)} finished cells read back, not recomputed" if rdirs else "")
          + (f"; A SELECTION: {selection}" if selection else "") + mem_note(), flush=True)
    cv_r2 = {}
    t_grid = time.time()
    for fit_name in cfg["fits"]:
        mask = torch.ones(len(rows), dtype=torch.bool) if fit_name == "record" else coco_rows
        for rname, (Xf, Xe, Xt) in relays.items():
            keys_here = [f"{fit_name}|{rname}|k{d}" for d in depths]
            try:
                sol = (None if all(k in resume or not selected(k) for k in keys_here) else
                       ridge_cv(Xf[mask].float(), [Y[d][mask] for d in depths], folds[mask], device))
            except HostMemoryStop as e:                        # the mid-cell stop inside the ridge (recorded after the unwind)
                midcell, stop_reason = (str(e), f"the ridge of {fit_name}|{rname}"), "the mid-cell stop"
                break
            for i, d in enumerate(depths):
                key = keys_here[i]
                if key in resume:                              # read back, not recomputed
                    row = torch.load(resume[key], weights_only=False)
                    done += 1
                    filed += 1
                    assert row["key"] == key and row["n"] == done, f"the resumed cell {row['key']} #{row['n']} sits at {key} #{done}"
                    result["cells"][key] = row["slot_reads"]
                    if span_on:
                        result["span"]["cells"][key] = row["span_reads"]
                    cv_r2[key] = row["cv_r2"]
                    shutil.copyfile(resume[key], os.path.join(cells_dir, f"{done:04d}.pt"))
                    print(row["line"], flush=True)             # the cell's own printed line, as first printed
                    print(f"[stitch s2] (cell {done} read back from {os.path.dirname(resume[key])}, not recomputed)", flush=True)
                    del row
                    continue
                if not selected(key):                          # outside the selection: keeps its number, not computed
                    done += 1
                    continue
                if TEST_MIDCELL_AT and fresh + 1 == TEST_MIDCELL_AT:     # the test hook: the watcher's flag as this cell starts
                    _MEM_STOP_AT.update(host=float("nan"), t="the test hook")
                    _MEM_STOP.set()
                try:
                    W, mx, my, alpha, r2 = sol[i]
                    cv_r2[key] = r2
                    pred = ((Xe.float().to(device, torch.float64) - mx) @ W + my).float().cpu()
                    cell = {"alpha": alpha, "cv_r2": r2}
                    cell.update(reads(run_slot((d, pred), maps=slot_full), pred, selfv[d], key=key))
                    if slot_full:
                        tgt_norm = float(Y[d][mask].norm(dim=-1).mean())
                        pn = pred * (tgt_norm / pred.norm(dim=-1, keepdim=True).clamp(min=1e-6))
                        cell["norm_matched_held"] = reads(run_slot((d, pn), maps=False))["held_all"]
                    result["cells"][key] = cell
                    if span_on:
                        pt = ((Xt.float().to(device, torch.float64) - mx) @ W + my).float().cpu()
                        sc_ = reads_span(run_span((d, pt), maps=True, base=base_sp), key=key, tok_in=pt,
                                         tok_true=ceil_sp["collected"][d])
                        sc_.update({"alpha": alpha, "cv_r2": r2})
                        result["span"]["cells"][key] = sc_
                except HostMemoryStop as e:                    # the mid-cell stop: the cell in hand leaves no trace
                    for store in (result["cells"], result["span"]["cells"] if span_on else {}, percap,
                                  percap_sp if span_on else {}, mood_sp if span_on else {}, cv_r2):
                        store.pop(key, None)
                    midcell, stop_reason = (str(e), f"cell {done + 1} ({key})"), "the mid-cell stop"
                    break
                done += 1
                fresh += 1
                el = time.time() - t0
                h = cell["held_all"]
                line = f"[stitch s2] cell {done}/{ncell} {key}: " + ("(OUTSIDE THE REGISTERED GRID) " if rname in extra else "")
                if span_on:
                    rr = result["span"]["cells"][key]["record"]
                    line += (f"SPAN record {rr['primary_centred']:+.3f} {fmt_ci(rr['primary_centred_ci'])} over {rr['n_phrases']} "
                             f"phrases, GAP over the off-diagonal {rr['gap']:+.3f} {fmt_ci(rr['gap_ci'])} (off {rr['offdiag']:+.3f}: "
                             f"same mood {rr['offdiag_same']:+.3f}, opposite {rr['offdiag_opposite']:+.3f}; THIS WORD beyond its mood "
                             f"{rr['word_signal']:+.3f} {fmt_ci(rr['word_signal_ci'])}, the mood alone {rr['mood_signal']:+.3f} "
                             f"{fmt_ci(rr['mood_signal_ci'])}; pooled {rr['primary_pooled']:+.3f}, raw {rr['primary_raw']:+.3f}); "
                             f"THE MOOD AXIS cos {rr['mood_cos']:+.3f} {fmt_ci(rr['mood_cos_ci'])} (ceiling "
                             f"{rr['mood_cos_ceiling']:+.3f}), size {rr['mood_size']:.2f} {fmt_ci(rr['mood_size_ci'])}, axis transfer "
                             f"{rr['transfer_axis_cos']:+.2f} / {rr['transfer_axis_size']:.2f}; leak {rr['leak']:.4f} (ceiling "
                             f"{ceil_sp_l:.4f}), hit {rr['hit']:.4f} (ceiling {ceil_sp_h:.4f}), strong-head hit "
                             f"{rr['hit_strong']:.4f} (ceiling {ceil_sp_hs:.4f}), in the real-word logit band "
                             f"{rr['band_share_token_head']:.2f} (filler {rr['filler_band_share_token_head']:.2f}), rank "
                             f"{result['span']['cells'][key]['record_rank_mean']:.2f} of {result['span']['cells'][key]['record_rank_chance']:.1f}"
                             f" chance | slot (secondary) held {h['primary_centred']:+.3f} {fmt_ci(h['primary_centred_ci'])}")
                else:
                    s_ = result["self"][str(d)]["held_all"]
                    line += (f"held centred {h['primary_centred']:+.3f} {fmt_ci(h['primary_centred_ci'])} over {h['n_phrases']} "
                             f"phrases (pooled {h['primary_pooled']:+.3f}), raw {h['primary_raw']:+.3f} (self "
                             f"{s_['primary_centred']:+.3f}); the two failure reads: leak {h['leak']:.4f} (ceiling {ceil_l:.4f}), "
                             f"rank {cell['held_rank_mean']:.2f} of {cell['held_rank_chance']:.1f} chance; shared "
                             f"{cell['shared_shift_cos']:+.2f}")
                line += f"; cv R2 {r2:.3f}; {el:.0f} s spent, about {(time.time() - t_grid) / fresh * (todo - fresh):.0f} s left"
                # the cell to disk and out of memory: its per-caption rows, its reads, its printed line
                row = {"n": done, "key": key, "slot": percap.pop(key), "slot_reads": cell, "cv_r2": r2, "line": line}
                if span_on:
                    mp, mc = mood_sp.pop(key)
                    row.update({"span": percap_sp.pop(key), "mood_proj": mp, "mood_cos": mc,
                                "span_reads": result["span"]["cells"][key]})
                torch.save(row, os.path.join(cells_dir, f"{done:04d}.pt"))
                filed += 1
                del row
                print(line, flush=True)
                # THE RESOURCE GATE: memory after every cell, the census, the soak, the budgets, the stop file, the planned cell
                # limit; a stop leaves at this cell boundary
                if HEAPMIN:
                    release_heap()
                host, rss, priv, dev = host_mem_gb()
                card = torch.cuda.max_memory_reserved() / 2 ** 30 if device == "cuda" else 0.0
                mem_log.append({"cell": done, "fresh": fresh, "host_gb": host, "resident_gb": rss, "committed_gb": priv,
                                "card_reserved_gb": dev, "card_peak_gb": card, "s": el})
                print(f"[stitch s2] mem after cell {done}: host {host:.2f} GB (budget {BUDGET['ram_gb']:g}; resident {rss:.2f}, "
                      f"committed {priv:.2f}, the card's reserved {dev:.2f}), card peak {card:.2f} GB", flush=True)

                def census():
                    nt, gb, grp = tensor_census()
                    mem_log[-1].update({"census_tensors": nt, "census_gb": gb, "census_top": grp,
                                        "python_blocks": sys.getallocatedblocks()})
                    print(f"[stitch s2] THE LIVE-TENSOR CENSUS after cell {done}: {nt} CPU tensors holding {gb:.2f} GB (host "
                          f"{host:.2f}); Python's allocated blocks {sys.getallocatedblocks()}; the largest groups: "
                          + "; ".join(f"[*, {', '.join(map(str, g['trailing_shape']))}] {g['dtype']} x{g['n']} {g['gb']:.2f} GB"
                                      for g in grp), flush=True)
                if fresh in CENSUS_CELLS or done % CENSUS_EVERY == 0:
                    census()
                if fresh == BUDGET["soak_cells"] and fresh > 5:
                    # a least-squares slope over this process's cells 5..20 (1-4 = warm-up, first-use allocations), on BOTH
                    # host measures (Windows can trim leaked heap out of the resident set while it stays committed)
                    pts = [m for m in mem_log if 5 <= m["fresh"] <= fresh]

                    def slope(vals):
                        xs = [m["fresh"] for m in pts]
                        mx, my = sum(xs) / len(xs), sum(vals) / len(vals)
                        return sum((x - mx) * (y - my) for x, y in zip(xs, vals)) / sum((x - mx) ** 2 for x in xs)
                    s_res = slope([m["resident_gb"] for m in pts])
                    s_net = slope([m["committed_gb"] - m["card_reserved_gb"] for m in pts])
                    s_dev = slope([m["card_reserved_gb"] for m in pts])
                    passed = max(s_res, s_net) <= BUDGET["soak_gb_per_cell"]
                    result["resources"]["soak"] = {"cells_fitted": [5, fresh], "warm_up_cells": [1, 4],
                                                   "resident_gb_per_cell": s_res, "committed_minus_card_gb_per_cell": s_net,
                                                   "card_reserved_gb_per_cell": s_dev, "passed": passed}
                    print(f"[stitch s2] THE SOAK (a slope fitted over cells 5-{fresh}; cells 1-4 are warm-up): resident "
                          f"{s_res:+.4f} GB per cell, committed minus the card's reserved {s_net:+.4f} GB per cell (allowance "
                          f"{BUDGET['soak_gb_per_cell']:g} on both): {'PASSED' if passed else 'FAILED'}; the card's reserved memory "
                          f"{s_dev:+.4f} GB per cell (bounded by its cap, printed only)", flush=True)
                    if not passed:
                        stop_reason = (f"the soak: host memory grew {max(s_res, s_net):.4f} GB per cell over cells 5-{fresh} "
                                       f"(resident {s_res:+.4f}, committed minus the card's reserved {s_net:+.4f})")
                if stop_reason is None:
                    if host > BUDGET["ram_gb"]:
                        stop_reason = f"host RAM: {host:.1f} GB, over the {BUDGET['ram_gb']:g} GB budget"
                    elif el / 3600 > BUDGET["wall_h"]:
                        stop_reason = f"wall time: {el / 3600:.1f} h, over the {BUDGET['wall_h']:g} h budget"
                    elif os.path.exists(STOP_FILE) and os.path.getmtime(STOP_FILE) > t0:
                        stop_reason = "the stop file"
                    elif BUDGET["max_cells"] and fresh >= BUDGET["max_cells"]:
                        stop_reason = f"the planned cell limit ({BUDGET['max_cells']} cells computed: a soak run)"
                if stop_reason:
                    if "census_gb" not in mem_log[-1]:
                        census()
                    break
            if stop_reason:
                break
        if stop_reason:
            break
    if midcell:                                                # recorded here, after the exception has unwound
        stop_reason = midcell_stop(*midcell)
    if stop_reason:
        result["resources"]["stopped"] = stop_reason
        result["partial"] = {"cells_done": filed, "position": done, "of": ncell, "reason": stop_reason}
        print(f"[stitch s2] STOPPED CLEANLY at cell {done} of {ncell} ({filed} cells on file): {stop_reason}; the cells done are "
              "saved as a PARTIAL grid", flush=True)
    elif selection:
        result["partial"] = {"cells_done": filed, "position": done, "of": ncell, "reason": f"a selection: {selection}"}
    if cfg.get("saturation"):
        result["saturation"] = "owed: run at the chosen cell after the grid (1,000 / 2,000 / 4,090 COCO captions)"
    big = span_on and not name.startswith("smoke")             # a full grid: the full JSON in OUT_DIR, a summary in REPORT_DIR
    out_dir = OUT_DIR if name.startswith("smoke") else REPORT_DIR    # a smoke test's numbers mean nothing: kept out of the record
    os.makedirs(out_dir, exist_ok=True)
    stem = f"stitch_{name}{RUN_TAG}"                           # STITCH_RUN_TAG keeps a selection's files apart from the grid's
    path = os.path.join(OUT_DIR, f"{stem}_full.json") if big else os.path.join(out_dir, f"{stem}.json")
    sum_path = os.path.join(REPORT_DIR, f"{stem}.json")

    def save():
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        if big:
            with open(sum_path, "w", encoding="utf-8") as fh:
                json.dump(summary(result, path), fh, indent=1)
    save()                                                     # saved BEFORE the margin analysis: a fault there loses no reads
    files = sorted(os.listdir(cells_dir))                      # the per-caption reads back from the cell files
    assert len(files) == filed, f"{len(files)} cell files for {filed} cells filed in {cells_dir}"
    for f in files:
        row = torch.load(os.path.join(cells_dir, f), weights_only=False)
        percap[row["key"]] = row["slot"]
        if span_on:
            percap_sp[row["key"]] = row["span"]
            mood_sp[row["key"]] = (row["mood_proj"], row["mood_cos"])
    pc_path = os.path.join(OUT_DIR, f"{stem}_percaption.pt")
    pc = {"percap": percap, "phrase": phr, "scene": torch.tensor(scene_of), "held_phrase": held, "read": readm,
          "single": single, "offset": offset, "depths": depths, "relays": list(relays), "cfg": cfg, "cv_r2": cv_r2,
          "context_spread": spread,
          "floor_parts": {"bf16_vs_fp32": 1.0 - precision["bf16_vs_fp32_centred_cos_mean"], "batch_shape": batch_floor,
                          "fp32_repeat_centred": precision["fp32_repeat_one_minus_centred_cos_mean"]},
          "partial": result.get("partial")}                    # no local path here: the file ships as written (the JSON is scrubbed)
    if span_on:
        pc.update({"percap_span": percap_sp, "span_sets": span_sets, "context_spread_span": spread_sp,
                   "floor_span": precision_sp["fp32_repeat_one_minus_centred_cos_mean"],
                   # the rows behind mood_size and mood_cos: a set's per-caption mood size = mood_proj over the mean of the
                   # ceiling's own projection on that set's captions with a mood (as reads_span computes it)
                   "mood_proj_span": {k: v[0] for k, v in mood_sp.items()}, "mood_cos_span": {k: v[1] for k, v in mood_sp.items()},
                   "ceiling_mood_proj_span": pc_mood, "ceiling_mood_cos_span": cc_mood})
    torch.save(pc, pc_path)
    print(f"[stitch s2] wrote {path}{' + the summary ' + sum_path if big else ''} + the per-caption reads {pc_path} "
          f"({time.time() - t0:.0f} s)", flush=True)
    if selection:                                              # a selection is a check or a half: the merged grid gets the rule
        print(f"[stitch s2] a selection ({selection}): no margin rule here", flush=True)
        return result
    try:                                                       # diagnostics collect, never raise mid-run
        result["margins"] = margins(pc_path)                   # a PARTIAL grid: on the cells reached, labelled (margins())
        save()
        print(f"[stitch s2] the margin rule added to {path}{' and ' + sum_path if big else ''} ({time.time() - t0:.0f} s)", flush=True)
    except Exception:
        import traceback
        print("[stitch s2] THE MARGIN ANALYSIS FAILED (the reads are saved; rerun: python -m alephllm_diffusion.stitch.s2 margins "
              f"{name}):\n{traceback.format_exc()}", flush=True)
        if name.startswith("smoke"):                           # the smoke test exists to fail loudly
            raise
    return result


def fmt_ci(ci):
    return "[n/a]" if ci[0] is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def margin_block(percap, phr, sets, prefixes, p1_sets, cfg, relays, depths, floor_fp32, form):
    """THE MARGIN RULE on one form's per-caption reads: a difference counts only if its 95% cluster-bootstrap interval over
    phrases (2,000 draws, pairs matched by caption) excludes zero AND it exceeds ten times the fp32 instrument's own repeat
    floor in the primary's units (the literal floor was dropped once the card confirmed the repeat number)."""
    def paired(a, b, mk):
        dlt = percap[a] - percap[b]
        lo, hi = cluster_ci(dlt[mk], phr[mk])
        mean = float(dlt[mk].mean())
        excl = lo is not None and (lo > 0 or hi < 0)
        return {"diff": mean, "ci": [lo, hi], "excludes_zero": excl, "clears_fp32_floor": excl and abs(mean) > floor_fp32}

    def depth_curve(prefix, mk):
        keys = [f"{prefix}|k{d}" for d in depths]
        means = [float(percap[k][mk].mean()) for k in keys]
        b = max(range(len(depths)), key=lambda i: means[i])
        nb = {str(depths[j]): paired(keys[b], keys[j], mk) for j in (b - 1, b + 1) if 0 <= j < len(depths)}
        ok = bool(nb) and all(v["clears_fp32_floor"] and v["diff"] > 0 for v in nb.values())
        return {"n": int(mk.sum()), "n_phrases": len(set(phr[mk].tolist())), "means": dict(zip(map(str, depths), means)),
                "best": str(depths[b]), "neighbours": nb,
                "verdict_fp32_floor": f"best at {depths[b]}, clearing its neighbours" if ok else "flat within noise at ridge capacity"}

    def whole(prefix):                                         # a PARTIAL grid: only relays with every depth read take part
        return all(f"{prefix}|k{d}" in percap for d in depths)

    not_reached = [p for p in prefixes if not whole(p)]
    curves = {}
    for prefix in prefixes:
        if whole(prefix):
            curves[prefix] = {lab: depth_curve(prefix, mk) for lab, mk in sets.items() if bool(mk.any())}
    p1 = {}
    for lab, mk in p1_sets.items():
        if not bool(mk.any()):
            continue
        for fn in cfg["fits"]:
            for rn in relays:
                parts = rn.split("|")
                if len(parts) != 3 or parts[0] not in cfg["her"] or not whole(f"{fn}|{rn}"):
                    continue
                ctrl = [f"{c}|{parts[1]}|close" for c in cfg["controls"]] + [r for r in ("spelling", "qwen_static") if r in relays]
                ctrl = [c for c in ctrl if whole(f"{fn}|{c}")]
                p1.setdefault(lab, {})[f"{fn}|{rn}"] = {str(d): {c: paired(f"{fn}|{rn}|k{d}", f"{fn}|{c}|k{d}", mk) for c in ctrl}
                                                        for d in depths}
    seed_spread = {}
    rec = next(iter(p1_sets.values()))
    if len(cfg["controls"]) >= 2:
        c0, c1 = cfg["controls"][:2]
        for fn in cfg["fits"]:
            for b in cfg["blocks"]:
                if whole(f"{fn}|{c0}|b{b}|close") and whole(f"{fn}|{c1}|b{b}|close"):
                    seed_spread[f"{fn}|b{b}"] = {str(d): paired(f"{fn}|{c0}|b{b}|close|k{d}", f"{fn}|{c1}|b{b}|close|k{d}", rec)
                                                 for d in depths}
    if not_reached:
        print(f"[stitch s2] A PARTIAL GRID, {form}: {len(not_reached)} of {len(prefixes)} relays not read at every depth, left out "
              f"of the margin rule: {', '.join(not_reached)}", flush=True)
    print(f"[stitch s2] THE MARGIN RULE, {form}: the floor of record {floor_fp32:.1e} (ten times the fp32 repeat in the primary's "
          f"units):", flush=True)
    for prefix, by_set in curves.items():
        for lab, c in by_set.items():
            nbs = "; ".join(f"vs k{k} {v['diff']:+.3f} {fmt_ci(v['ci'])}{' clears' if v['clears_fp32_floor'] else ''}"
                            for k, v in c["neighbours"].items())
            print(f"[stitch s2]   {prefix} on {lab} ({c['n']} captions, {c['n_phrases']} phrases): {c['verdict_fp32_floor']}; "
                  f"{nbs}", flush=True)
    for lab, by_her in p1.items():
        for her, by_d in by_her.items():
            for d, by_c in by_d.items():
                print(f"[stitch s2]   P1 margins on {lab}, {her} k{d}: " + "; ".join(
                    f"minus {c} {v['diff']:+.3f} {fmt_ci(v['ci'])}{' clears' if v['clears_fp32_floor'] else ''}"
                    for c, v in by_c.items()), flush=True)
    for key, by_d in seed_spread.items():
        print(f"[stitch s2]   seed spread {key}: " + "; ".join(f"k{d} {v['diff']:+.3f} {fmt_ci(v['ci'])}" for d, v in by_d.items()),
              flush=True)
    out = {"floor_fp32": floor_fp32, "depth_curves": curves, "p1_her_minus_controls": p1, "seed_spread": seed_spread}
    if not_reached:
        out["not_reached"] = not_reached
    return out


def margins(pc_path):
    """The margin rule on a saved per-caption file: the matched span (the form of record) when present, then the one-slot form;
    then P1-P3 printed as registered, with the 230,415 plumbing read's values beside them as the prior."""
    pc = torch.load(pc_path, weights_only=False)
    percap, phr, held, readm, single = pc["percap"], pc["phrase"], pc["held_phrase"], pc["read"], pc["single"]
    depths, relays, cfg, fp = pc["depths"], pc["relays"], pc["cfg"], pc["floor_parts"]
    cells = [f"{fn}|{rn}" for fn in cfg["fits"] for rn in relays]
    out = {}
    if pc.get("partial"):
        out["partial"] = pc["partial"]
        print(f"[stitch s2] THE MARGIN RULE ON A PARTIAL GRID ({pc['partial']['cells_done']} of {pc['partial']['of']} cells; "
              f"stopped: {pc['partial']['reason']}): the relays read at every depth only; NOT the full grid", flush=True)
    if "percap_span" in pc:
        ss = pc["span_sets"]
        sets = {"the read of record: FRAGMENT-SPLIT held-out (P1, P2, P3)": ss["record"],
                "single-token, seen phrases on held-out scenes (P2 as registered)": ss["single_seen"],
                "single-token, held-out phrases (P2 as registered)": ss["single_held"],
                "answer-carried by measured share >= .5, held-out up/down (sensitivity)": ss["carried_held"],
                "all held-out phrases, every class (all phrases)": ss["all_held"]}
        out["span"] = margin_block(pc["percap_span"], phr, sets, cells,
                                   {"record": ss["record"], "carried_held (sensitivity)": ss["carried_held"]},
                                   cfg, relays, depths, 10.0 * pc["floor_span"], "THE MATCHED SPAN (the form of record)")
    sets = {"held-out phrases (P1, P3)": held & readm, "single-token, seen phrases on held-out scenes (P2)": ~held & readm & single,
            "single-token, held-out phrases (P2)": held & readm & single}
    floor_fp32 = 10.0 * fp.get("fp32_repeat_centred", fp["batch_shape"])
    out["slot"] = margin_block(percap, phr, sets, ["self"] + cells, {"held": held & readm}, cfg, relays, depths, floor_fp32,
                               "THE ONE-SLOT FORM (secondary)" if "percap_span" in pc else "THE ONE-SLOT FORM")
    if "percap_span" in pc:
        out["registered_predictions"] = registered_predictions(out, pc)
        return out
    out = out["slot"]                                          # the one-slot runs keep their earlier JSON shape
    return out


def registered_predictions(mg, pc):
    """P1-P3 printed exactly as registered (with the registered erratum; P2's self-stitch clause on the secondary only),
    with the 230,415 plumbing read (the one-slot form, 10 phrases) beside them as the prior. Prints and returns; decides nothing."""
    cfg, depths = pc["cfg"], pc["depths"]
    prior = {}
    sp = os.path.join(REPORT_DIR, "stitch_shakedown.json")
    if os.path.exists(sp):
        prior = json.load(open(sp, encoding="utf-8")).get("margins", {})
    print("[stitch s2] REGISTERED P1: her held-out margin over the untrained trunk and over the SPELLING floor (the whole phrase) is "
          "smallest at k = 0 and grows with depth. On the read of record:", flush=True)
    rows = {}
    for her, by_d in mg["span"]["p1_her_minus_controls"].get("record", {}).items():
        for c in next(iter(by_d.values())):
            seq = [(d, by_d[d][c]) for d in map(str, depths)]
            small = min(seq, key=lambda x: x[1]["diff"])[0]
            large = max(seq, key=lambda x: x[1]["diff"])[0]
            rows[f"{her} minus {c}"] = {"smallest_at": small, "largest_at": large,
                                        "clearing_depths": [d for d, v in seq if v["clears_fp32_floor"]]}
            print(f"[stitch s2]   {her} minus {c}: " + ", ".join(f"k{d} {v['diff']:+.3f}{'*' if v['clears_fp32_floor'] else ''}"
                                                              for d, v in seq)
                  + f"; smallest at k{small}, largest at k{large} (* = clears the margin rule)", flush=True)
    for her, by_d in prior.get("p1_her_minus_controls", {}).items():
        print(f"[stitch s2]   PRIOR (230,415, one-slot, 10 phrases) {her}: " + "; ".join(
            f"k{d} " + ", ".join(f"{c.split('|')[0]} {v['diff']:+.2f}{'*' if v.get('clears_fp32_floor') else ''}" for c, v in by_c.items())
            for d, by_c in by_d.items()), flush=True)
    print("[stitch s2] REGISTERED P2: the downstream penalty is lowest at an INTERIOR depth; an interior optimum counts as support "
          "only where the self-stitch curve is flat, or on single-token phrases (the self-stitch clause applies to the one-slot "
          "secondary only). Her depth curves:", flush=True)
    p2 = {}
    for prefix, by_set in mg["span"]["depth_curves"].items():
        if prefix.split("|")[1] not in cfg["her"]:
            continue
        p2[prefix] = {lab: (c["best"], c["verdict_fp32_floor"]) for lab, c in by_set.items()}
        for lab, c in by_set.items():
            print(f"[stitch s2]   {prefix} on {lab}: best at k{c['best']}: {c['verdict_fp32_floor']}", flush=True)
    for prefix, by_set in prior.get("depth_curves", {}).items():
        if "step230415" in prefix:
            print(f"[stitch s2]   PRIOR {prefix}: " + "; ".join(f"{lab}: k{c['best']} {c.get('verdict_fp32_floor')}"
                                                           for lab, c in by_set.items()), flush=True)
    print("[stitch s2] REGISTERED P3: at that depth the PRIMARY READ sits inside the context spread for seen-type phrases and "
          "degrades on unseen words in step with the ridge's held-out residual. At each of her relays' best depth on the record:",
          flush=True)
    p3 = {}
    spread = pc.get("context_spread_span", {})
    for prefix, by_set in mg["span"]["depth_curves"].items():
        if prefix.split("|")[1] not in cfg["her"]:
            continue
        rec = next(iter(by_set.values()))
        d = rec["best"]
        key = f"{prefix}|k{d}"
        seen = pc["percap_span"][key][pc["span_sets"]["fragment_seen"]]
        p3[prefix] = {"depth": d, "record": rec["means"][d], "fragment_seen": float(seen.mean()) if len(seen) else None,
                      "context_spread_fragment_seen": spread.get("fragment_seen"), "context_spread_record": spread.get("record"),
                      "cv_r2": pc["cv_r2"].get(key)}
        r = p3[prefix]
        print(f"[stitch s2]   {prefix} at k{d}: record {r['record']:+.3f}, seen fragment-split phrases "
              f"{r['fragment_seen'] if r['fragment_seen'] is None else format(r['fragment_seen'], '+.3f')} against their context "
              f"spread {r['context_spread_fragment_seen']}; the ridge's held-out R2 {r['cv_r2']:.3f}", flush=True)
    return {"P1": rows, "P2": p2, "P3": p3, "prior_source": sp if prior else None}


if __name__ == "__main__":
    if sys.argv[1] == "margins":                               # offline: the margin rule on a saved run, then into its JSON
        nm = sys.argv[2]
        mg = margins(os.path.join(OUT_DIR, f"stitch_{nm}_percaption.pt"))
        full = os.path.join(OUT_DIR, f"stitch_{nm}_full.json")
        jp = full if os.path.exists(full) else os.path.join(OUT_DIR if nm.startswith("smoke") else REPORT_DIR, f"stitch_{nm}.json")
        res = json.load(open(jp, encoding="utf-8"))
        res["margins"] = mg
        with open(jp, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        if jp == full:                                         # a full grid: refresh its summary in REPORT_DIR too
            with open(os.path.join(REPORT_DIR, f"stitch_{nm}.json"), "w", encoding="utf-8") as fh:
                json.dump(summary(res, full), fh, indent=1)
        print(f"[stitch s2] the margin rule added to {jp}")
    else:
        main(sys.argv[1], device="cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else "cuda")
