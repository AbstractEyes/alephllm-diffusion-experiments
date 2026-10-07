"""alephllm_diffusion.stitch.s0 - stage 0 of the stitch instrument (the grid that maps Beatrix's states into Anima's
text-conditioning space; registered 2026-10-04, before any run): the captions and their token maps. CPU only, no model weights:
Anima's own tokenizers (Qwen3 and T5) from the diffusion-pipe fork's configs (settings.DP). Reads the COCO captions
(coco_caps_2x2048.json in settings.DATA_DIR) and anima-trainer's quality prefix and 32 scenes with their train / held-out split;
writes s0.pt to the output folder (settings.OUT_DIR), which stages 1 and 2 read.

FIT CAPTIONS: the 4,090 distinct COCO captions on disk + the grid's captions for the 64 TRAIN phrases on the 24 TRAIN scenes,
minus every COCO caption holding a held-out phrase's content word; every caption carries Anima's quality prefix. A ROW = one Qwen3
token of a fit caption (the caption's last token dropped: it has no byte after it), de-duplicated by the text up to and including
its closing byte.
EVAL: the grid's full captions "an illustration of {scene}, {phrase}." (seen phrases on the 8 held-out scenes; held-out phrases on
all 32), each with its phrase span, the ceiling token (the last Qwen3 token of the phrase) and the phrase's Qwen3 token count; and
per scene the slot base "an illustration of {scene}." with ONE appended Qwen3 position (the space token, which is also the empty
slot) and T5's bare space piece inserted before T5's end token (the slot's question).
THE MATCHED SPAN (fixed before any read): 86 phrases (the earlier 74-phrase set + twelve held-out fragment-split words), each in
one a-priori T5 class (FRAGMENT-SPLIT = the read of record, i.e. the registered primary read, 16 held-out phrases; MORPHEME-SPLIT;
WHOLE; MIXED), asserted against the tokenizer; per eval caption the phrase's T5 ids and question pieces (a bare space piece right
before them counted as the first), and the neutral filler of equal Qwen3 token count from the bank (single words for 1-3 tokens,
"X and Y" for longer counts and for "X and Y" phrases), its caption checked in place (the same Qwen3 length and phrase positions),
with its own T5 ids and pieces; one row per phrase token (' and' included): caption, Qwen3 position, closing byte, last byte,
token id, span.
Byte = character here: every caption is ASCII (asserted).
Usage: python -m alephllm_diffusion.stitch.s0
"""
import json
import os
import random
import re
import sys
import zlib

import torch

from .. import settings  # noqa: E402
from transformers import AutoTokenizer, T5TokenizerFast  # noqa: E402

from geolip_anima_trainer import anima_experiments as ax  # noqa: E402
from geolip_anima_trainer.sana_runner import HELD_OUT, SUBJECTS, TRAIN  # noqa: E402

OUT_DIR = settings.OUT_DIR
COCO = os.path.join(settings.DATA_DIR, "coco_caps_2x2048.json")
PREFIX = ax.PREFIX
FREE_PIECE = 3                                           # T5's bare space piece (anima-trainer e027 form C; checked: id 3, unk 2)
qwen = AutoTokenizer.from_pretrained(os.path.join(settings.DP, "configs", "qwen3_06b"), local_files_only=True)
t5 = T5TokenizerFast(vocab_file=os.path.join(settings.DP, "configs", "t5_old", "spiece.model"),
                     tokenizer_file=os.path.join(settings.DP, "configs", "t5_old", "tokenizer.json"))
SPACE = qwen(" ", add_special_tokens=False)["input_ids"]
assert SPACE == [220] and t5.convert_ids_to_tokens(FREE_PIECE) == "\u2581", (SPACE, t5.convert_ids_to_tokens(FREE_PIECE))

# the earlier 74-phrase set (its original construction, verbatim)
PHRASES = {
    ("up", "train"): ["cheerful and upbeat", "joyful and uplifting", "happy and bright", "sunny and cheerful",
                      "lighthearted and warm", "bright and hopeful"],
    ("up", "heldout"): ["merry and glad", "elated", "jubilant and gleeful", "blissful"],
    ("down", "train"): ["gloomy and downbeat", "somber and melancholy", "sad and dark", "bleak and grey",
                        "sorrowful and heavy", "dreary and dull"],
    ("down", "heldout"): ["mournful", "forlorn and desolate", "dismal", "despairing and grim"],
    ("neutral", "train"): ["an ordinary scene", "plain", "everyday", "neutral"],
    ("neutral", "heldout"): ["typical", "unremarkable"],
}
PROBE = ["happy", "joyful", "cheerful", "glad", "merry", "elated", "jubilant", "blissful", "delighted", "bright", "sunny", "upbeat",
         "uplifting", "hopeful", "lighthearted", "playful", "ecstatic", "content", "thrilled", "gleeful", "buoyant", "radiant", "jolly",
         "festive", "carefree", "euphoric", "exuberant", "optimistic", "warm", "lively", "sad", "gloomy", "somber", "melancholy",
         "mournful", "forlorn", "desolate", "dismal", "grim", "bleak", "dreary", "sorrowful", "depressed", "miserable", "despairing",
         "downcast", "glum", "morose", "heavy", "dark", "grey", "tearful", "lonely", "hopeless", "wistful", "doleful", "woeful",
         "joyless", "cheerless", "bitter"]
rows = [(c, s, p) for (c, s), ps in PHRASES.items() for p in ps]
held_words = {w for c, s, p in rows if s == "heldout" for w in p.replace(" and ", " ").split()}
train_text = {p for c, s, p in rows}
rows += [("up" if PROBE.index(w) < 30 else "down", "train", w) for w in PROBE if w not in held_words and w not in train_text]
assert len(rows) == 74, len(rows)
# twelve held-out fragment-split words, chosen by tokenization and polarity only, appended (indices 74-85; 0-73 unchanged)
NEW_HELD = {"up": ["blithe", "jaunty", "jovial", "giddy", "chirpy", "exultant"],
            "down": ["despondent", "dejected", "sullen", "lugubrious", "brooding", "anguished"]}
rows += [(c, "heldout", w) for c, ws in NEW_HELD.items() for w in ws]
held_words |= {w for ws in NEW_HELD.values() for w in ws}
assert len(rows) == 86, len(rows)
phrases = [{"class": c, "split": s, "text": p} for c, s, p in rows]
held_re = re.compile(r"\b(" + "|".join(sorted(held_words)) + r")\b", re.IGNORECASE)
# THE CLASSES (registered a priori from the T5 tokenization; never moved by a read): FRAGMENT-SPLIT = every content word splits
# into pieces that carry no meaning of their own (the read of record); MORPHEME-SPLIT = split, and some piece is a real word
# belonging to the mood meaning; WHOLE = T5 keeps every content word whole (the low-answer-share class; not a negative control
# at the adapter's output: its answer share measured .30); MIXED = the rest
MORPHEME = {"blissful", "mournful", "unremarkable", "gloomy and downbeat", "upbeat", "uplifting", "lighthearted", "buoyant",
            "carefree", "sorrowful", "downcast", "tearful", "hopeless", "joyless", "cheerless"}
FRAGMENT = {"elated", "jubilant and gleeful", "forlorn and desolate", "dismal", "somber and melancholy", "ecstatic", "jolly",
            "euphoric", "exuberant", "gloomy", "somber", "melancholy", "bleak", "dreary", "depressed", "glum", "morose", "wistful",
            "doleful", "woeful"} | {w for ws in NEW_HELD.values() for w in ws}
MIXED = {"merry and glad", "despairing and grim", "lighthearted and warm", "sorrowful and heavy", "dreary and dull", "cheerful and upbeat",
         "joyful and uplifting", "bleak and grey"}           # registered counts: train 3 up + 3 down, held out 2; asserted below
# THE NEUTRAL BANK by Qwen3 token count (single words for counts 1-3, "X and Y" pairs for longer counts and for "X and Y"
# phrases; never a repeated token; checked in place per caption): the neutral fillers of an earlier interchange test + their kin
BANK = {(1, False): "neutral", (2, False): "workaday", (3, False): "nondescript", (3, True): "plain and ordinary",
        (4, True): "plain and workaday", (5, True): "ordinary and nondescript", (6, True): "workaday and nondescript",
        (7, True): "nondescript and unexceptional"}


def full_caption(scene, phrase):
    return PREFIX + f"an illustration of {scene}, {phrase}."


def qmap(text):
    e = qwen(text, return_offsets_mapping=True, add_special_tokens=False)
    return e["input_ids"], [tuple(o) for o in e["offset_mapping"]]


def fold_of(text):
    return zlib.crc32(text.encode("utf-8")) % 5


# ---- fit captions -----------------------------------------------------------------------------------------------------
coco_all = json.load(open(COCO, encoding="utf-8"))
coco = sorted({c.strip() for v in coco_all.values() for c in v})
n_coco_raw = len(coco)
coco = [c for c in coco if not held_re.search(c)]
random.Random(0).shuffle(coco)                            # the nested saturation subsets take prefixes of this order
fit = [{"text": PREFIX + c, "source": "coco", "coco_rank": i} for i, c in enumerate(coco)]
fit += [{"text": full_caption(SUBJECTS[i], p["text"]), "source": "grid", "coco_rank": -1, "scene": i, "phrase": j}
        for i in TRAIN for j, p in enumerate(phrases) if p["split"] == "train"]
row_list, seen = [], {}
for ci, f in enumerate(fit):
    assert f["text"].isascii(), f["text"]
    ids, offs = qmap(f["text"])
    f["qwen_ids"], f["fold"] = ids, fold_of(f["text"])
    for t in range(len(ids) - 1):                         # the last token has no byte after it
        a, e = offs[t]
        if e <= a:
            continue
        key = f["text"][: e + 1]
        if key in seen:                                   # a repeated byte prefix (the quality prefix, a scene's opening) counts once
            continue
        seen[key] = len(row_list)
        row_list.append((ci, t, e, e - 1, ids[t], a, e))
rows_t = torch.tensor(row_list, dtype=torch.long)         # caption, token, closing byte, last byte, token id, span start, span end
src = torch.tensor([0 if fit[c]["source"] == "coco" else 1 for c in rows_t[:, 0].tolist()])
print(f"fit captions: {len(fit)} ({len(coco)} COCO of {n_coco_raw} distinct, {len(fit) - len(coco)} grid); rows {len(row_list)} "
      f"(COCO {int((src == 0).sum())}, grid {int((src == 1).sum())}); distinct row tokens {len(set(rows_t[:, 4].tolist()))}")

# ---- the classes and the fillers (the matched span) -------------------------------------------------------------------
def t5_span(text, ps, pe):
    """T5 ids of the caption (with its end token) and the positions of the pieces covering [ps, pe), a bare space piece right
    before them counted as the first piece (the phrase's questions)."""
    e = t5(text, return_offsets_mapping=True)
    tids, toff = e["input_ids"], e["offset_mapping"]
    pos = [k for k, (a, b) in enumerate(toff) if a < pe and b > ps]
    if pos and pos[0] > 0 and tids[pos[0] - 1] == FREE_PIECE:
        pos = [pos[0] - 1] + pos
    assert tids[-1] == t5.eos_token_id
    return tids, pos


def content_whole(text):
    return all(len(t5(" " + w, add_special_tokens=False)["input_ids"]) == 1 for w in text.split() if w not in ("and", "an", "a"))


for p in phrases:
    p["cls"] = ("FRAGMENT" if p["text"] in FRAGMENT else "MORPHEME" if p["text"] in MORPHEME else
                "MIXED" if p["text"] in MIXED else "WHOLE")
    if p["cls"] == "WHOLE":                               # the a-priori table: everything else must be whole by the tokenizer
        assert content_whole(p["text"]), p["text"]
    else:
        assert not content_whole(p["text"]), p["text"]
assert sum(p["cls"] == "FRAGMENT" and p["split"] == "heldout" for p in phrases) == 16, "the read of record must hold 16 phrases"
assert all(sum(p["cls"] == "FRAGMENT" and p["split"] == "heldout" and p["class"] == c for p in phrases) == 8 for c in ("up", "down"))
cnt = {(c, s): sum(p["cls"] == c and p["split"] == s for p in phrases) for c in ("WHOLE", "MIXED") for s in ("train", "heldout")}
assert cnt == {("WHOLE", "train"): 30, ("WHOLE", "heldout"): 1, ("MIXED", "train"): 6, ("MIXED", "heldout"): 2}, cnt


def filler_for(phrase, n_tok):
    pair = " and " in phrase or n_tok > 3
    return BANK[(n_tok, pair)]


# ---- eval captions and slot bases -------------------------------------------------------------------------------------
evals, eval_tok = [], []                                  # all 86 phrases on all 32 scenes: the centring means need every phrase
for si, scene in enumerate(SUBJECTS):                     # on each scene; seen phrases on train scenes are a nuisance estimate only
    for j, p in enumerate(phrases):
        text = full_caption(scene, p["text"])
        ids, offs = qmap(text)
        ps = text.rindex(", " + p["text"] + ".") + 2
        pe = ps + len(p["text"])
        toks = [k for k, (a, e) in enumerate(offs) if a < pe and e > ps]
        tids, tpos = t5_span(text, ps, pe)
        fill = filler_for(p["text"], len(toks))
        ftext = full_caption(scene, fill)
        fids, foffs = qmap(ftext)
        fps = ftext.rindex(", " + fill + ".") + 2
        ftoks = [k for k, (a, e) in enumerate(foffs) if a < fps + len(fill) and e > fps]
        assert len(ftoks) == len(toks) and ftoks == toks and len(fids) == len(ids), (p["text"], fill, toks, ftoks)
        ftids, ftpos = t5_span(ftext, fps, fps + len(fill))
        assert len(tpos) >= 1 and len(ftpos) >= 1
        ei = len(evals)
        for k in toks:                                    # the matched span: one row per phrase token, ' and' included
            eval_tok.append((ei, k, offs[k][1], offs[k][1] - 1, ids[k], offs[k][0], offs[k][1]))
        evals.append({"scene": si, "phrase": j, "text": text, "qwen_ids": ids, "p_start": ps, "p_end": pe,
                      "close_byte": pe, "last_byte": pe - 1, "t_ceil": toks[-1], "phrase_toks": toks,
                      "phrase_tok_ids": [ids[k] for k in toks], "n_tok": len(toks), "held_phrase": p["split"] == "heldout",
                      "held_scene": si in HELD_OUT, "read": p["split"] == "heldout" or si in HELD_OUT, "cls": p["cls"],
                      "t5_ids": tids, "t5_span": tpos, "filler": fill, "filler_qwen_ids": fids, "filler_t5_ids": ftids,
                      "filler_t5_span": ftpos, "filler_is_phrase": fill == p["text"]})
bases = []
for si, scene in enumerate(SUBJECTS):
    text = PREFIX + f"an illustration of {scene}."
    ids, _ = qmap(text)
    t5_ids = t5(text)["input_ids"]
    assert t5_ids[-1] == t5.eos_token_id
    t5_ids = t5_ids[:-1] + [FREE_PIECE, t5.eos_token_id]
    bases.append({"text": text, "qwen_ids": ids + SPACE, "slot_n": len(ids), "t5_ids": t5_ids, "q_m": len(t5_ids) - 2})
off = [b["q_m"] - b["slot_n"] for b in bases]
print(f"eval captions: {len(evals)} (read: held-out phrases {sum(e['held_phrase'] for e in evals)}, seen phrases on held-out "
      f"scenes {sum(e['read'] and not e['held_phrase'] for e in evals)}; nuisance only {sum(not e['read'] for e in evals)}); "
      f"single-token phrases among the read {sum(e['n_tok'] == 1 and e['read'] for e in evals)}; slot offset T5 - Qwen3 positions "
      f"{min(off)}..{max(off)}")
print("THE CLASSES (a priori; train / held out): " + "; ".join(
    f"{c} {sum(p['cls'] == c and p['split'] == 'train' for p in phrases)} / {sum(p['cls'] == c and p['split'] == 'heldout' for p in phrases)}"
    for c in ("FRAGMENT", "MORPHEME", "WHOLE", "MIXED")))
print("THE NEUTRAL BANK (Qwen3 tokens, pair shape): " + "; ".join(f"{n}{' pair' if pr else ''} '{w}'" for (n, pr), w in sorted(BANK.items())))
print("EACH PHRASE'S FILLER (class, Qwen3 tokens):")
seen_f = {}
for e in evals:
    seen_f.setdefault(e["phrase"], (e["filler"], e["n_tok"]))
for j, p in enumerate(phrases):
    print(f"  [{j}] {p['split']:7s} {p['class']:7s} {p['cls']:8s} {p['text']!r} ({seen_f[j][1]}) -> {seen_f[j][0]!r}")
eval_tok_t = torch.tensor(eval_tok, dtype=torch.long)    # eval caption, Qwen3 position, closing byte, last byte, token id, span
print("THE FILLER FOR 'neutral' IS 'neutral' ITSELF: that caption's ceiling effect is zero by construction and its answer share "
      f"undefined (printed as n/a); neutral phrases stay out of every centring reference ({sum(e['filler_is_phrase'] for e in evals)} "
      "captions)")
print(f"matched-span rows: {len(eval_tok)} phrase tokens over {len(evals)} eval captions")
os.makedirs(OUT_DIR, exist_ok=True)
torch.save({"fit": fit, "rows": rows_t, "evals": evals, "eval_tok": eval_tok_t, "bases": bases, "phrases": phrases,
            "held_words": sorted(held_words), "space_id": SPACE[0], "free_piece": FREE_PIECE, "prefix": PREFIX,
            "subjects": SUBJECTS, "held_out": HELD_OUT, "train": TRAIN, "n_coco_raw": n_coco_raw, "bank": BANK},
           os.path.join(OUT_DIR, "s0.pt"))
print("wrote", os.path.join(OUT_DIR, "s0.pt"))
