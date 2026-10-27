"""Rule-based room classification with explicit evidence. Scores are heuristic rule strengths, NOT calibrated probabilities."""
import re
import numpy as np

UNIT_ID = re.compile(r"\b0?\d/D\d/\d+/(\d+|K|TU)\b", re.I)
TYPE_CODE = re.compile(r"type\s*([bk]|tu)\s*\d{3}", re.I)


def match_keywords(text, classes):
    t = re.sub(r"\s+", "", text.lower())   # OCR drops/adds spaces ("(TypeB703)", "Tutor Flat")
    for name, spec in classes.items():
        for kw in spec.get("keywords", []):
            if re.sub(r"\s+", "", kw) in t:
                return name, kw
    return None, None


def elongation(poly):
    mrr = poly.minimum_rotated_rectangle
    xs = np.array(mrr.exterior.coords)
    a, b = np.linalg.norm(xs[1] - xs[0]), np.linalg.norm(xs[2] - xs[1])
    return max(a, b) / max(min(a, b), 1e-6), min(a, b)


def classify_room(poly, texts, n_circles, classes, pt2_per_m2):
    """texts: OCR strings whose centre lies inside the room."""
    area_m2 = poly.area / pt2_per_m2
    el, short_pt = elongation(poly)
    ev, flags = [], []
    units = sorted({m.group(0).upper() for t in texts for m in UNIT_ID.finditer(t)})
    codes = [m.group(0) for t in texts for m in TYPE_CODE.finditer(t)]
    joined = " | ".join(texts)

    # 1. explicit text labels (strongest evidence)
    cls, kw = None, None
    for t in texts:
        c, k = match_keywords(t, {n: s for n, s in classes.items() if n != "communal_or_shared_space"})
        if c in ("stair", "lift", "shaft_or_riser", "storage", "laundry_or_utility"):
            cls, kw = c, k; break
    if cls is None:
        c, k = match_keywords(joined, {n: s for n, s in classes.items() if n in ("bedroom", "kitchen")})
        if c:
            cls, kw = c, k
    if cls is None:
        c, k = match_keywords(joined, classes)
        if c:
            cls, kw = c, k
    if cls:
        ev.append(f"ocr:{kw}")
        score = 0.9
    else:
        score = None

    # 2. fixtures / geometry when there is no label
    if cls is None:
        if n_circles >= 1 and 800 <= poly.area <= 4500:
            cls, score = "bathroom", 0.5; ev.append(f"round_fixtures:{n_circles}")
        elif el >= 4.0 and area_m2 >= 3:
            cls, score = "corridor_or_hallway", 0.6; ev.append(f"elongated:{el:.1f}")
        elif area_m2 >= 40:
            cls, score = "communal_or_shared_space", 0.4; ev.append(f"large_area_m2:{area_m2:.0f}")
        elif 6 <= area_m2 <= 30 and short_pt >= 60:
            cls, score = "unknown", None; ev.append("room_sized_no_label")
        else:
            cls, score = "unknown", None; ev.append("no_evidence")

    if cls == "bedroom" and len(units) > 1:
        flags.append(f"possibly_merged:{len(units)}_unit_ids")
    if cls == "bedroom" and not (6 <= area_m2 <= 35):
        flags.append("unusual_area_for_bedroom")
    if len(units) > 1 and cls != "bedroom":
        flags.append(f"multiple_unit_ids:{len(units)}")
    review = score is None or score < 0.8 or bool(flags)
    return dict(type=cls, heuristic_score=score, evidence=ev, flags=flags, unit_ids=units, type_codes=codes,
                approx_area_m2=round(area_m2, 1), elongation=round(el, 2), review_required=review)
