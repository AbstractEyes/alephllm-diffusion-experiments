"""The experiment's sessions on one card: one call runs a session's steps in order and skips the steps already done.

    from alephllm_diffusion import sessions
    sessions.run(1)      # the mount checks, the port check, stage 1 of the grid, the first 64 pictures   (about 2.5-3 h)
    sessions.run(2)      # the grids, the picks, the mount contrast, the export                         (about 4-7 h)
    sessions.run(3)      # the pictures of the relay arms                                               (about 1-1.5 h)
    sessions.status()    # every step: done, or not yet
    sessions.stop()      # a running grid stops at its next cell and keeps its cells; run the session again to resume

Every step runs as its own process, so the card is empty between steps. The output shows one line when a step starts, a
progress line every few minutes and one line when it ends; the step's full output goes to <home>/logs/<step>.log, and its last
lines are shown when it fails. A finished step leaves a marker in <home>/markers/.

With the data store on (the default), the session's files go to the private data repo at its end and every half hour during the
long steps, and a session started on a new machine first fetches the earlier sessions' files, so nothing done is computed twice.
The token comes from HF_TOKEN or Colab's Secrets panel; the compute steps run without it (every model they read is public), and
only the uploads and the picture runs, which publish their results, receive it.

smoke=True runs the same steps on 80 captions with the small grid configurations (their numbers mean nothing) and skips the port
check, the pictures and every upload: a rehearsal of the runner itself, for a scratch workspace (ALEPHLLM_DIFFUSION_HOME).
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from . import __version__, settings
from .paths import git_commit, repo_root

STEP, GATE_STEP = 245674, 230415      # the final checkpoint; the checkpoint of the reference run the port check repeats
GROUPS = ("gCA", "gCB")               # the nine-arm group (eight stage arms + the caption arm), two training seeds
FRAMES = ("cap",)                     # the frame column: every caption prefixed with the caption arm's own label
SMOKE_LIMIT = 80                      # a smoke run's stage 1: the first 80 fit captions
BEAT_S = 300                          # seconds between progress lines
EXPERIMENTS_REPO = "AbstractPhil/geolip-beatrix-anima"            # where the picture test publishes its results
E029_META = "experiments/e029_anima_relay_in_pictures/meta.json"


class StepFailed(RuntimeError):
    """A step ended with an error; the session stops there and the steps before it stay done."""


class Stopped(RuntimeError):
    """A grid stopped before its end (a stop request or its wall budget); running the session again resumes it."""


@dataclass(frozen=True)
class Step:
    name: str
    session: int
    about: str
    minutes: int
    smoke: bool = True                # False: skipped in a smoke run


STEPS = (
    Step("gates", 1, "the mount checks: the nine arms mounted on Beatrix, both arm seeds; all arms masked must equal the bare "
                     "model exactly, and this mount must equal the library's own route exactly", 10),
    Step("mount_read", 1, "the mount read: how much the arms change Beatrix's caption states (2,048 COCO captions, no pictures)",
         30),
    Step("port_check", 1, "the port check: 9 cells of the reference run recomputed on this card and compared", 30, smoke=False),
    Step("restart_test", 1, "the restart test: a small grid run whole, stopped and resumed, and in two halves; all must agree", 5),
    Step("stage1", 1, "stage 1 of the grid: Beatrix's states on every trunk (bare, two untrained controls, each arm mount, "
                      "framed captions, mood captions)", 40),
    Step("pictures_a", 1, "the picture test, stage A: 64 pictures that choose the caption form for stage B", 10, smoke=False),
    Step("grid_record", 2, "the grid of record: 612 cells mapping Beatrix's states into Anima's text space", 240),
    Step("pick_record", 2, "the pick: the cell carrying the most caption mood, its tied companion, the control cell", 10),
    Step("grid_mounts", 2, "the mount grids (144 cells per mount), their picks, the framed-caption column, the mount contrast",
         120),
    Step("export", 2, "the export: the picked cells' text states for the picture test (bare and mood captions)", 10),
    Step("publish_grids", 2, "the grids to the public data repo (scrubbed copies)", 5, smoke=False),
    Step("pictures_b", 3, "the picture test, stage B: about 900 pictures, Beatrix's relay beside the arm-mounted relays", 75,
         smoke=False),
)
RESTORE = {                           # fetched at a session's start when missing (globs relative to the workspace)
    1: ["markers/*.json", "reports/*", "mount_read/*", "out/s1_*.pt"],
    2: ["markers/*.json", "reports/*", "mount_read/*", "out/s1_*.pt", "out/stitch_*.json", "out/stitch_*_percaption.pt"],
    3: ["markers/*.json", "reports/*", "out/e029_export_*"],
}
SAVE = {                              # sent to the data store at a session's end
    1: ["markers/*.json", "logs/*", "reports/*", "reports/env/*", "mount_read/*", "out/s1_*.pt",
        "out/stitch_rehearsal_portgate*.json"],
    2: ["markers/*.json", "logs/*", "reports/*", "out/stitch_*.json", "out/stitch_*_percaption.pt", "out/stitch_*_cells_*/*",
        "out/e029_export_*"],
    3: ["markers/*.json", "logs/*", "reports/*"],
}
GRID_FILES = ["out/stitch_*_cells_*/*", "out/stitch_*.json", "out/stitch_*_percaption.pt", "logs/*.log"]
CHECKPOINT = {"stage1": ["out/s1_*.pt", "logs/*.log"], "grid_record": GRID_FILES, "grid_mounts": GRID_FILES}

_CELL = re.compile(r"cell (\d+)/(\d+)")
_LEFT = re.compile(r"about (\d+) s left")


def _dur(s: float) -> str:
    s = int(s)
    if s < 90:
        return f"{s} s"
    if s < 5400:
        return f"{round(s / 60)} min"
    return f"{s // 3600} h {round(s % 3600 / 60)} min"


def _clock() -> str:
    return time.strftime("%H:%M")


def _markers(smoke: bool) -> str:
    return os.path.join(settings.HOME, "markers", "smoke") if smoke else os.path.join(settings.HOME, "markers")


def _card() -> tuple[str, int, int]:
    """(name, total MiB, used MiB) of the first card, from nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.splitlines()[0]
        name, total, used = (x.strip() for x in out.rsplit(",", 2))
        return name, int(total), int(used)
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return "no card found", 0, 0


def grid_state(name: str) -> str:
    """'done' (the grid complete on disk), 'partial' (stopped; a rerun resumes it) or 'absent'."""
    for p in (os.path.join(settings.OUT_DIR, f"stitch_{name}_full.json"), os.path.join(settings.OUT_DIR, f"stitch_{name}.json")):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return "partial" if json.load(fh).get("partial") else "done"
    return "absent"


def resume_dir(name: str) -> str:
    """The cell folder of an earlier run of grid `name` holding the most finished cells (the newest among equals), or ''."""
    best, key = "", None
    for d in glob.glob(os.path.join(settings.OUT_DIR, f"stitch_{name}_cells_*")):
        k = (sum(f.endswith(".pt") for f in os.listdir(d)), os.path.getmtime(d))
        if k[0] and (key is None or k > key):
            best, key = d, k
    return best


def s1_forms(limit=0) -> dict:
    """{file: stage 1's batch form} for the stage-1 files on disk at this scale (limit 0: the full files; files before 0.3.0
    record none and are the 'equal' form). Reads only each file's small entries (the tensors are memory-mapped, not read)."""
    import torch
    suffix = f"_limit{limit}.pt" if limit else ".pt"
    out = {}
    for f in sorted(glob.glob(os.path.join(settings.OUT_DIR, "s1_*.pt"))):
        name = os.path.basename(f)
        if not name.endswith(suffix) or (not limit and "_limit" in name):
            continue
        d = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
        out[name] = (d.get("batching") or {"form": "equal"})["form"]
        del d
    return out


def stage1_jobs(mounts: list) -> list:
    """Stage 1's files in their order, as (trunk, mount, frame, mood, fatal): the bare trunk and the two untrained controls; the
    mood form of the trunk and of one control (not fatal: only the export's mood form needs them); each mount with its mood form;
    the framed column (the trunk, then each mount)."""
    jobs = [(s, None, None, False, True) for s in ("random:0", "random:1", str(STEP))]
    jobs += [(str(STEP), None, None, True, False), ("random:0", None, None, True, False)]
    for g in mounts:
        jobs += [(str(STEP), g, None, False, True), (str(STEP), g, None, True, False)]
    for f in FRAMES:
        jobs += [(str(STEP), None, f, False, True)] + [(str(STEP), g, f, False, True) for g in mounts]
    return jobs


def stage1_file(spec: str, mount=None, frame=None, mood=False, limit=0) -> str:
    """Stage 1's file name for a trunk ('245674' or 'random:0'), as the stage-1 program writes it."""
    base = f"random{spec.split(':')[1]}" if spec.startswith("random:") else f"step{spec}"
    return (f"s1_{'mood_' if mood else ''}{base}{'_' + mount if mount else ''}{'_f' + frame if frame else ''}"
            f"{f'_limit{limit}' if limit else ''}.pt")


class _Session:
    def __init__(self, smoke, store, card_fraction, ram_gb, card_free_mib, verbose):
        self.smoke, self.store, self.verbose, self.card_free_mib = smoke, store, verbose, card_free_mib
        self.limit = SMOKE_LIMIT if smoke else 0
        self.lim = f"_limit{SMOKE_LIMIT}" if smoke else ""
        self.cfg = ({"record": "smoke_m0", "mounts": "smoke_mounts", "frames": "smoke_frames"} if smoke
                    else {"record": "record", "mounts": "mounts", "frames": "frames"})
        self.logs = os.path.join(settings.HOME, "logs", "smoke") if smoke else os.path.join(settings.HOME, "logs")
        self.marks = _markers(smoke)
        self.tok = None
        self.note = ""                                  # the step's own count on progress lines ("file 4 of 15, <name>")
        self.card, total, _ = _card()
        if card_fraction is None:
            card_fraction = 0.90 if total >= 80_000 else 0.70
        if ram_gb is None:
            import psutil
            ram_gb = min(96, int(psutil.virtual_memory().total / 2 ** 30 * 0.5))
        self.env = dict(os.environ)
        self.env.update(ALEPHLLM_DIFFUSION_HOME=str(settings.HOME), HF_TOKEN="", PYTHONUNBUFFERED="1",
                        STITCH_MEM_FRACTION=f"{card_fraction:g}", BTX3_MEM_FRACTION=f"{card_fraction:g}",
                        STITCH_RAM_GB=f"{ram_gb:g}", STITCH_WALL_H="10", OMP_NUM_THREADS=str(min(16, os.cpu_count() or 1)))
        if sys.platform == "linux":
            self.env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        self.budget = f"card fraction {card_fraction:g}, host budget {ram_gb:g} GB"
        for d in (self.logs, self.marks, settings.OUT_DIR, settings.REPORT_DIR):
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------ output
    def say(self, msg: str) -> None:
        print(f"          {msg}", flush=True)

    def token(self) -> str:
        if self.tok is None:
            from . import storage
            self.tok = storage.token()
        return self.tok

    # ------------------------------------------------------------------ markers
    def marker(self, name: str) -> dict | None:
        p = os.path.join(self.marks, f"{name}.json")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    def mark(self, name: str, **info) -> None:
        rec = {"step": name, "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "card": self.card,
               "package": __version__, "commit": git_commit(repo_root()) if repo_root() else None, "smoke": self.smoke, **info}
        with open(os.path.join(self.marks, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)

    # ------------------------------------------------------------------ one program
    def card_free(self) -> None:
        _, _, used = _card()
        if used > self.card_free_mib:
            raise StepFailed(f"the card holds {used} MiB: another job is on it (one card, one job); nothing was started")

    def sh(self, log: str, module: str, *args, env=None, check=True, capture=False, fresh=False):
        """python -m <module> <args> as its own process, its output appended to logs/<log>.log (fresh: the log starts empty).
        Returns the exit code (and the output lines with capture); a nonzero exit raises StepFailed when check."""
        e = dict(self.env, **(env or {}))
        path = os.path.join(self.logs, f"{log}.log")
        cmd = [sys.executable, "-u", "-m", module, *map(str, args)]
        lines, state = [], {"last": "", "progress": None}
        with open(path, "w" if fresh else "a", encoding="utf-8") as fh:
            fh.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} python -m {module} {' '.join(map(str, args))}\n")
            fh.flush()
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace", bufsize=1, env=e)

            def pump():
                for line in p.stdout:
                    fh.write(line)
                    fh.flush()
                    state["last"] = line.rstrip()
                    if _LEFT.search(line) or _CELL.search(line):     # the newest line that says how far the step is
                        state["progress"] = (line.rstrip(), time.time())
                    if capture:
                        lines.append(line.rstrip())
                    if self.verbose:
                        print(line, end="", flush=True)

            th = threading.Thread(target=pump, daemon=True)
            th.start()
            t0, beat, pressed = time.time(), time.time() + BEAT_S, False
            while p.poll() is None:
                try:
                    time.sleep(1)
                    if not self.verbose and time.time() >= beat:
                        beat += BEAT_S
                        self.progress(time.time() - t0, state["last"], state["progress"])
                except KeyboardInterrupt:
                    if pressed:
                        self.say(f"left running unwatched (process {p.pid}); its log: {path}")
                        raise
                    pressed = True
                    stop()
                    self.say("stop pressed: a running grid stops at its next cell and keeps its cells (any other step runs to "
                             "its end); press stop again to leave this step running unwatched")
            th.join(timeout=60)
        rc = p.returncode
        if rc and check:
            self.show_tail(path)
            raise StepFailed(f"{module} {' '.join(map(str, args))} ended with exit {rc}; its log: {path}")
        return (rc, lines) if capture else rc

    def progress(self, spent: float, last: str, prog=None) -> None:
        """One progress line: the time in, and from the newest line that says how far the step is, the cell and the time left
        (less the time since that line was printed); without such a line, the step's last line."""
        parts = [f"{_dur(spent)} in"]
        line, at = prog if prog else (last, time.time())
        if m := _CELL.search(line):
            parts.append(f"cell {m[1]} of {m[2]}")
        note = getattr(self, "note", "")
        if m := _LEFT.search(line):
            parts.append(f"about {_dur(max(0, int(m[1]) - (time.time() - at)))} left" + (" in this file" if note else ""))
        if len(parts) == 1 and last:
            parts.append(last[:110])
        self.say((f"{note}: " if note else "") + ", ".join(parts))

    def show_tail(self, path: str, n: int = 25) -> None:
        with open(path, encoding="utf-8", errors="replace") as fh:
            tail = fh.read().splitlines()[-n:]
        print(f"          the last {len(tail)} lines of {path}:", flush=True)
        for x in tail:
            print(f"          | {x[:220]}", flush=True)

    # ------------------------------------------------------------------ inputs
    def need(self, names: list) -> None:
        missing = [n for n in names if not os.path.exists(os.path.join(settings.OUT_DIR, n))]
        if missing:
            raise StepFailed(f"missing in {settings.OUT_DIR}: {missing} (an earlier step makes them; on a new machine they come "
                             "back from the data store when it is on)")

    def restore(self, patterns: list) -> None:
        if self.store:
            from . import storage
            storage.restore(patterns, say=self.say)

    def stage1(self, spec: str, mount=None, frame=None, mood=False, fatal=True, label="", left=None):
        """One stage-1 file as its own process (kept when on disk). label: the step's count ("file 4 of 15"); left(seconds) -> the
        step's time left after this file. Returns the seconds it took (None when kept)."""
        name = stage1_file(spec, mount, frame, mood, self.limit)
        where = f" ({label})" if label else ""
        if os.path.exists(os.path.join(settings.OUT_DIR, name)):
            self.say(f"{name}: on disk, kept{where}")
            return None
        args = [spec] + ([self.limit] if self.limit else []) + ([f"--mount={mount}"] if mount else []) \
            + ([f"--frame={frame}"] if frame else []) + (["--mood"] if mood else [])
        self.note = f"{label}, {name}" if label else ""
        t0 = time.time()
        try:
            rc = self.sh(name[:-3], "alephllm_diffusion.stitch.s1", *args, check=fatal)
        finally:
            self.note = ""
        secs = time.time() - t0
        if rc:
            self.say(f"{name}: FAILED (logs/{name[:-3]}.log){where}; the export's mood form will stop, the bare export not")
        else:
            tail = f"; about {_dur(left(secs))} left in this step" if left else ""
            self.say(f"{name}: written in {_dur(secs)}{where}{tail}")
        return secs

    def mounts(self) -> list:
        """The mounts that get stage-1 files and grids: the nine-arm group on both seeds, plus the eight alone (gXA) when the
        mount read finds them changing caption states above ten times the repeat floor at a grid block (the plan's rule)."""
        p = os.path.join(settings.MOUNT_READ_OUT, f"mount_read_step{STEP}{'_smoke' if self.smoke else ''}.json")
        with open(p, encoding="utf-8") as fh:
            r = json.load(fh)
        moved = {s: [k for k, v in r[f"M8_{s}"]["write_over_10x_floor"].items() if v] for s in ("A", "B") if f"M8_{s}" in r}
        eight = any(moved.values())
        self.say("the eight stage arms change caption states above ten times the repeat floor at: "
                 + (", ".join(f"seed {s}: {v or 'no block'}" for s, v in moved.items())) + " -> "
                 + ("the eight get a grid of their own (gXA)" if eight else "the eight read as the bare model (no grid of their own)"))
        if self.smoke:
            return list(GROUPS)                                # the small grid configurations name their mounts themselves
        return list(GROUPS) + (["gXA"] if eight else [])

    def mount_env(self) -> tuple[list, list, dict]:
        mounts = (self.marker("stage1") or {}).get("mounts") or list(GROUPS)
        tags = [f"step{STEP}_{g}{self.lim}" for g in mounts]
        ftags = []
        for f in FRAMES:
            ftags += [f"step{STEP}_f{f}{self.lim}"] + [f"step{STEP}_{g}_f{f}{self.lim}" for g in mounts]
        return mounts, tags, {"STITCH_MOUNT_TAGS": ",".join(tags), "STITCH_FRAME_TAGS": ",".join(ftags)}

    def grid(self, log: str, name: str, env=None) -> None:
        """A grid run, resumed from its saved cells when an earlier run stopped; complete on return, or Stopped."""
        st = grid_state(name)
        if st == "done":
            self.say(f"the {name} grid is complete on disk: kept")
            return
        self.restore([f"out/stitch_{name}_cells_*/*"])
        env = dict(env or {})
        rd = resume_dir(name)
        if rd:
            env["STITCH_RESUME"] = rd
            self.say(f"resuming the {name} grid from {os.path.basename(rd)} "
                     f"({sum(f.endswith('.pt') for f in os.listdir(rd))} cells on file)")
        rc = self.sh(log, "alephllm_diffusion.stitch.s2", name, env=env, check=False)
        st = grid_state(name)
        if st == "partial":
            raise Stopped(f"the {name} grid stopped before its end (a stop, or its wall budget); run this session again: it "
                          "resumes from the saved cells")
        if rc or st != "done":
            self.show_tail(os.path.join(self.logs, f"{log}.log"))
            raise StepFailed(f"the {name} grid ended with exit {rc} and no complete grid on disk; a rerun resumes from its "
                             "saved cells")

    # ------------------------------------------------------------------ the steps
    def do_gates(self):
        self.sh("gates", "alephllm_diffusion.mount.gates", ",".join(GROUPS))
        with open(os.path.join(settings.REPORT_DIR, "mount_gates.json"), encoding="utf-8") as fh:
            g = json.load(fh)
        self.say("every exact gate passes" if not g["failed"] else f"FAILED: {g['failed']}")
        self.mark("gates", report="reports/mount_gates.json")

    def do_mount_read(self):
        self.sh("mount_read", "alephllm_diffusion.mount.read", ",".join(GROUPS), *(["--smoke=1"] if self.smoke else []))
        self.mark("mount_read")

    def do_port_check(self):
        for spec in ("random:0", "random:1", str(GATE_STEP)):
            self.stage1(spec)
        from .stitch.port_gate import GATE_KEYS
        self.sh("port_check_cells", "alephllm_diffusion.stitch.s2", "rehearsal", fresh=True,
                env={"STITCH_ONLY_CELLS": ",".join(GATE_KEYS), "STITCH_RUN_TAG": "_portgate"})
        rc = self.sh("port_check", "alephllm_diffusion.stitch.port_gate", "check",
                     os.path.join(self.logs, "port_check_cells.log"),
                     os.path.join(settings.OUT_DIR, "stitch_rehearsal_portgate_full.json"), check=False, fresh=True)
        if rc:
            self.show_tail(os.path.join(self.logs, "port_check.log"))
            raise StepFailed("THE PORT CHECK FAILED: this card's numbers differ from the reference run's beyond the registered "
                             "bar; nothing after it runs (logs/port_check.log names the cells and fields)")
        self.mark("port_check")

    def do_restart_test(self):
        self.need([stage1_file(s, limit=SMOKE_LIMIT) for s in ("212000", "random:0", "random:1")])
        self.sh("restart_test", "alephllm_diffusion.stitch.resume_test")
        self.mark("restart_test")

    def do_stage1(self):
        # one batch form per workspace: stage-1 files made in another form (another release, or the other setting) round
        # differently under bf16, so a workspace never mixes them (the grid refuses too)
        other = {f: form for f, form in s1_forms(self.limit).items() if form != settings.S1_BATCHING}
        if other:
            raise StepFailed(f"stage-1 files on disk were made with another batch form than this run's "
                             f"'{settings.S1_BATCHING}': {other}; start a new workspace, or set ALEPHLLM_DIFFUSION_S1_BATCHING to "
                             "their form to continue them")
        mounts = self.mounts()
        jobs = stage1_jobs(mounts)
        took = {True: [], False: []}                   # seconds per written file: mood files, state files

        def left(n):
            def est(secs):
                took[jobs[n][3]].append(secs)
                every = took[True] + took[False]
                rest = [j for j in jobs[n + 1:] if not os.path.exists(os.path.join(settings.OUT_DIR, stage1_file(*j[:4],
                                                                                                                   self.limit)))]
                return sum((sum(took[j[3]]) / len(took[j[3]])) if took[j[3]] else sum(every) / len(every) for j in rest)
            return est
        for n, (spec, mount, frame, mood, fatal) in enumerate(jobs):
            self.stage1(spec, mount=mount, frame=frame, mood=mood, fatal=fatal, label=f"file {n + 1} of {len(jobs)}",
                        left=left(n))
        self.mark("stage1", mounts=mounts)

    def pictures(self, stage: str, *extra):
        env = {"HF_TOKEN": self.token(), "HF_HOME": os.path.join(settings.ANIMA_DATA, "hf_cache"),
               "ANIMA_DIFFUSION_PIPE": settings.DP, "OMP_NUM_THREADS": "8"}
        self.sh(f"pictures_{stage.lower()}", "geolip_anima_trainer.relay_dp", "--stage", stage, *extra,
                "--data-root", settings.ANIMA_DATA, "--models-dir", settings.MODELS_DIR, env=env)

    def do_pictures_a(self):
        self.pictures("A")
        self.mark("pictures_a")

    def do_grid_record(self):
        self.need([stage1_file(s, limit=self.limit) for s in ("random:0", "random:1", str(STEP))])
        self.grid("grid_record", self.cfg["record"])
        self.mark("grid_record")

    def do_pick_record(self):
        rec = self.cfg["record"]
        self.sh("pick_record", "alephllm_diffusion.stitch.pick", rec)
        _, out = self.sh("control_cell", "alephllm_diffusion.stitch.export", rec, "--control-cell", capture=True, check=False)
        ctrl = next((x.split()[1:3] for x in out if x.startswith("CONTROL_CELL ")), None)
        if ctrl:
            key, relay = ctrl
            self.say(f"the pick is in the last-byte convention: its control cell {key} is computed outside the grid")
            rc = self.sh("control_outside", "alephllm_diffusion.stitch.s2", rec, check=False,
                         env={"STITCH_EXTRA_RELAYS": relay, "STITCH_ONLY_CELLS": key, "STITCH_RUN_TAG": "_control_outside"})
            if rc:
                self.say("the control's own run failed (logs/control_outside.log): the export labels the control unverified")
        self.mark("pick_record", control_outside=ctrl[0] if ctrl else None)

    def do_grid_mounts(self):
        mounts, tags, env = self.mount_env()
        self.env.update(env)
        self.need([stage1_file(str(STEP), mount=g, limit=self.limit) for g in mounts])
        self.grid("grid_mounts", self.cfg["mounts"])
        for g, t in zip(mounts, tags):
            self.sh(f"pick_mounts_{g}", "alephllm_diffusion.stitch.pick", self.cfg["mounts"], t)
        _, out = self.sh("frame_keys", "alephllm_diffusion.stitch.mounts", "--frame-keys", self.cfg["record"],
                         self.cfg["frames"], capture=True)
        keys = out[-1].strip() if out else ""
        self.say(f"the frame column: {len(keys.split(','))} cells (the pick and its companion in each framed trunk)")
        rc = self.sh("grid_frames", "alephllm_diffusion.stitch.s2", self.cfg["frames"], env={"STITCH_ONLY_CELLS": keys},
                     check=False)
        if rc:
            self.say("the frame column's run failed (logs/grid_frames.log): the contrast reports it as not read")
        rc = self.sh("mount_contrast", "alephllm_diffusion.stitch.mounts", self.cfg["record"], self.cfg["mounts"],
                     self.cfg["frames"], check=False)
        if rc:
            self.say("the mount contrast failed (logs/mount_contrast.log); the grids and picks stand")
        self.mark("grid_mounts", mounts=mounts, frame_column_ok=not rc)

    def do_export(self):
        _, tags, env = self.mount_env()
        env.update(STITCH_EXPORT_MOUNTS=tags[0], STITCH_EXPORT_MOUNTS_RUN=self.cfg["mounts"])
        rc = self.sh("export", "alephllm_diffusion.stitch.export", self.cfg["record"], env=env, check=False)
        if rc == 8:
            self.say("THE MOOD FORM FAILED (logs/export.log): the bare export stands; stage B stops only if stage A chose "
                     "the mood form")
        elif rc:
            self.show_tail(os.path.join(self.logs, "export.log"))
            raise StepFailed(f"the export's check failed (exit {rc}): the grids and picks stand; a rerun redoes only the export")
        self.mark("export", mood_form=(rc == 0))

    def do_publish_grids(self):
        env = {"HF_TOKEN": self.token()}
        self.sh("publish_record", "alephllm_diffusion.stitch.ship", "record", env=env)
        for run in ("mounts", "frames"):
            if self.sh(f"publish_{run}", "alephllm_diffusion.stitch.ship", run, env=env, check=False):
                self.say(f"the {run} grid did not publish (logs/publish_{run}.log)")
        self.mark("publish_grids")

    def do_pictures_b(self):
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(EXPERIMENTS_REPO, E029_META, token=self.token()), encoding="utf-8") as fh:
            form = json.load(fh)["stage_a"]["read"]["form_next"]
        export = os.path.join(settings.OUT_DIR, f"e029_export_record{'_mood' if form == 'mood' else ''}.safetensors")
        if not os.path.exists(export):
            raise StepFailed(f"stage A chose the {form} form, and its export {os.path.basename(export)} is not here")
        self.say(f"stage A chose the {form} caption form: {os.path.basename(export)}")
        self.pictures("B", "--export", export)
        self.mark("pictures_b", form=form, export=os.path.basename(export))

    # ------------------------------------------------------------------ the session
    def start(self, session: int, steps: list) -> None:
        from . import models, storage
        size = ("a smoke run at small scale" if self.smoke
                else f"about {_dur(60 * sum(s.minutes for s in steps))}")
        print(f"[{_clock()}] SESSION {session} on {self.card}: {len(steps)} steps, {size}; workspace {settings.HOME}; "
              f"{self.budget}", flush=True)
        rc, out = self.sh("environment", "alephllm_diffusion.environment", check=False, capture=True, fresh=True)
        for x in out:                                          # its own process: this one never touches the card
            self.say(x.strip())
        if rc:
            raise StepFailed("the environment differs from the pins (above); run the install cell again")
        env_dir = os.path.join(settings.REPORT_DIR, "env")              # the environment of record, kept with the session
        os.makedirs(env_dir, exist_ok=True)
        for fname, cmd in (("pip_freeze.txt", [sys.executable, "-m", "pip", "freeze"]), ("nvidia_smi.txt", ["nvidia-smi"])):
            try:
                with open(os.path.join(env_dir, fname), "w", encoding="utf-8") as fh:
                    fh.write(subprocess.run(cmd, capture_output=True, text=True).stdout)
            except OSError:
                pass
        if repo_root():
            rc = self.sh("tests", "pytest", str(repo_root() / "tests"), "-q", "-p", "no:cacheprovider", check=False, fresh=True,
                         env={"CUDA_VISIBLE_DEVICES": "-1"})
            rc2 = self.sh("tests", "geolip.alephllm.tests.test_arm_mount", check=False, env={"CUDA_VISIBLE_DEVICES": "-1"})
            if rc or rc2:
                self.show_tail(os.path.join(self.logs, "tests.log"))
                raise StepFailed("a test failed (logs/tests.log); nothing was run")
            self.say("the package's tests and the library's mount test pass (CPU)")
        if self.store:
            self.token()
            self.restore(RESTORE[session])
            storage.fetch_reference(tok=self.tok, say=self.say)
        elif storage.missing_reference():
            raise StepFailed(f"reference data missing and the data store is off: {storage.missing_reference()}")
        bad = storage.check_reference()
        if bad:
            raise StepFailed(f"reference files on disk differ from their recorded SHA-256: {bad} (delete them; the next start "
                             "fetches them again)")
        models.fetch(judge=not self.smoke, say=self.say)

    def finish(self, session: int) -> None:
        if self.store:
            from . import storage
            storage.save(SAVE[session], message=f"session {session}", tok=self.tok, say=self.say)


def _session_steps(session: int) -> list:
    return [s for s in STEPS if s.session == session]


def run(session: int, smoke: bool = False, store: bool | None = None, card_fraction: float | None = None,
        ram_gb: float | None = None, card_free_mib: int = 2000, verbose: bool = False) -> bool:
    """Run session 1, 2 or 3: every step not yet done, in order; True when the session completes. store: the private data store
    (default: on, off in a smoke run). card_fraction / ram_gb: the card's per-process memory fraction and the grid's host-memory
    budget (defaults from the machine). card_free_mib: a step starts only while the card holds at most this much (one card, one
    job). verbose: every output line of every step."""
    steps = _session_steps(session)
    assert steps, f"there is no session {session} (1, 2 or 3)"
    s = _Session(smoke, (not smoke) if store is None else store, card_fraction, ram_gb, card_free_mib, verbose)
    t_all = time.time()
    try:
        s.start(session, steps)
        for i, st in enumerate(steps, 1):
            head = f"[{_clock()}] step {i}/{len(steps)}"
            m = s.marker(st.name)
            if m:
                print(f"{head} done already ({m['utc']} UTC on {m['card']}): {st.about}", flush=True)
                continue
            if s.smoke and not st.smoke:
                print(f"{head} skipped in a smoke run: {st.about}", flush=True)
                continue
            print(f"{head} {st.about}" + ("" if s.smoke else f" (about {_dur(60 * st.minutes)})"), flush=True)
            s.card_free()
            t0 = time.time()
            ck = None
            if s.store and st.name in CHECKPOINT:
                from . import storage
                ck = storage.Checkpointer(CHECKPOINT[st.name], every=1800, tok=s.tok, say=s.say)
                ck.__enter__()
            try:
                getattr(s, f"do_{st.name}")()
            finally:
                if ck:
                    ck.__exit__(None, None, None)
            print(f"[{_clock()}] done in {_dur(time.time() - t0)}: {st.name}", flush=True)
    except (StepFailed, Stopped) as e:
        print(f"[{_clock()}] SESSION {session} STOPPED: {e}", flush=True)
        s.finish(session)
        return False
    s.finish(session)
    print(f"[{_clock()}] SESSION {session} COMPLETE in {_dur(time.time() - t_all)}", flush=True)
    summary(session, smoke=smoke)
    return True


def status(smoke: bool = False, fetch: bool = False) -> None:
    """Every step of the three sessions: done (when, on which card) or not yet. fetch: first fetch the markers of sessions run
    on other machines from the data store."""
    if fetch:
        from . import storage
        storage.restore(["markers/*.json"], say=lambda m: None)
    marks = _markers(smoke)
    for st in STEPS:
        p = os.path.join(marks, f"{st.name}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                m = json.load(fh)
            state = f"done {m['utc']} UTC on {m['card']}"
        else:
            state = "not yet"
        print(f"session {st.session}  {st.about.split(':')[0]:<22} {state}", flush=True)
    for name in ("record", "mounts"):
        rd = resume_dir(name)
        print(f"the {name} grid: {grid_state(name)}" + (f" (saved cells in {os.path.basename(rd)})" if rd else ""), flush=True)


def stop() -> None:
    """A running grid stops at its next cell boundary and saves its cells; running the session again resumes it."""
    os.makedirs(settings.OUT_DIR, exist_ok=True)
    with open(os.path.join(settings.OUT_DIR, "STOP"), "w", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("stop requested: the running grid stops at its next cell and keeps its cells", flush=True)


def summary(session: int, smoke: bool = False) -> None:
    """What the session found, in a few lines (each line is skipped when its file is missing)."""
    def load(*parts):
        try:
            with open(os.path.join(*parts), encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    rep, out = settings.REPORT_DIR, settings.OUT_DIR
    if session == 1:
        g = load(rep, "mount_gates.json")
        if g:
            print("the mount checks:", "every exact gate passes" if not g["failed"] else f"FAILED {g['failed']}")
            for grp, r in g["groups"].items():
                try:
                    t = {x["reading"]: x for x in r["reproduction_stored"]["table"]}
                    a, b = t["caption rows, arms off"], t["caption rows, all on"]
                    print(f"  {grp}: the stored caption read reproduced, arms off {a['reproduced']:.4f} -> all on "
                          f"{b['reproduced']:.4f} bits per byte (stored {a['stored']:.4f} -> {b['stored']:.4f})")
                except (KeyError, TypeError):
                    pass
        r = load(settings.MOUNT_READ_OUT, f"mount_read_step{STEP}{'_smoke' if smoke else ''}.json")
        if r:
            print("the mount read: how far each mount moves the caption states (largest block, relative to the bare model)")
            for k, v in r.items():
                if isinstance(v, dict) and "write_ratio" in v:
                    big = max(v["write_ratio"], key=v["write_ratio"].get)
                    print(f"  {k:<6} {v['write_ratio'][big]:.2e} at {big}; above ten times the repeat floor at "
                          f"{sum(v['write_over_10x_floor'].values())} of {len(v['write_over_10x_floor'])} grid blocks")
        files = sorted(f for f in os.listdir(out) if f.startswith("s1_")) if os.path.isdir(out) else []
        print(f"stage 1: {len(files)} files")
    elif session == 2:
        name = "smoke_m0" if smoke else "record"
        p = load(out if smoke else rep, f"pick_{name}.json")
        if p:
            comp = (p.get("companion") or {}).get("pick")
            print(f"the pick: {p['pick']['key']}" + (f"; its companion {comp['key']}" if comp and comp['key'] != p['pick']['key']
                                                     else ""))
        c = load(out if smoke else rep, "stitch_mounts_contrast.json" if not smoke else "stitch_smoke_mounts_contrast.json")
        if c:
            for col in ("bare", "framed"):
                for tm, by in (c.get(col) or {}).items():
                    x = by.get("pick of record") or {}
                    if "diff" in x:
                        print(f"  {col} captions, {tm}: mood size {x['mount_size']:.3f} vs the bare model {x['m0_size']:.3f}, "
                              f"difference {x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]")
        ex = sorted(f for f in os.listdir(out) if f.startswith("e029_export_")) if os.path.isdir(out) else []
        print(f"the export: {[f for f in ex if f.endswith('.safetensors')]}")
    elif session == 3:
        print(f"the pictures and their reads: {EXPERIMENTS_REPO}, {os.path.dirname(E029_META)}")
