"""The caption pack's text frame: one fixed field order, a plain label per line. The caption arm was trained on rows rendered
this way, so its reads render captions the same way."""

STRATA = ["hair", "face", "skin", "body", "pose", "clothing", "accessory", "color", "action", "abstract_quality", "scene_level"]


def render_row(row):
    """The proposed frame: one fixed field order, a plain label per line. Every later line restates what earlier lines hold."""
    out = []
    if row.get("tags"):
        out.append("tags: " + row["tags"])
    if row.get("caption"):                      # the source's own words (the web's alt text, the COCO caption) come first
        out.append("caption: " + row["caption"])
    if row.get("short"):
        out.append("short: " + row["short"])
    for st in STRATA:
        v = (row.get("strata") or {}).get(st)
        if v and st not in ("abstract_quality", "action"):
            out.append(f"{st.replace('_', ' ')}: " + ", ".join(v))
    sc = row.get("scene") or {}
    bits = [f"{k} {sc[k]}" for k in ("setting", "mood", "layout") if sc.get(k) and sc.get(k) != "unknown"]   # the style vote is noisy
    if bits:
        out.append("scene: " + "; ".join(bits))
    if row.get("photo"):
        out.append("photo: " + row["photo"])
    if row.get("structure"):
        out.append("structure: " + row["structure"])
    if row.get("medium"):
        out.append("medium: " + row["medium"])
    if row.get("long"):
        out.append("long: " + row["long"])
    if row.get("rating"):                       # the information at the end: the source's rating word, a classification target
        out.append("rating: " + row["rating"])
    return "\n".join(out)
