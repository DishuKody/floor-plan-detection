"""Pure raster room segmentation (no OCR, no file I/O) so that it can be tested on synthetic plans with known ground truth.

Root causes of the v3 raster inaccuracy that this module addresses (measured against expected.png, see review_v4.md):
  1. bold text survived the wall-thickness filter -> text became 'wall' -> holes / wiggly outlines inside rooms and false rooms on titles & logos;
  2. the plan footprint was derived from ALL ink (text included) -> the title / logo box became 'rooms';
  3. door and opening gaps were not closed -> rooms were split by a geodesic (curved) partition instead of by the straight wall / threshold line;
  4. elliptical opening rounded wall corners; polygons were taken half a pixel inside the mask.
"""
from dataclasses import dataclass
import numpy as np
import cv2
from shapely.geometry import Polygon
from skimage.graph import MCP_Geometric
from skimage.morphology import skeletonize


@dataclass
class PageTransform:
    """image px <-> PDF points <-> page-render px for an image placed axis-aligned at `rect` (x0, y0, x1, y1 in pt)."""
    rect: tuple
    W: int
    H: int
    scale: float            # page-render px per pt

    @property
    def sx(self):
        return (self.rect[2] - self.rect[0]) / self.W

    @property
    def sy(self):
        return (self.rect[3] - self.rect[1]) / self.H

    def img_to_pt(self, p):
        p = np.asarray(p, float)
        return p * [self.sx, self.sy] + [self.rect[0], self.rect[1]]

    def pt_to_img(self, p):
        p = np.asarray(p, float)
        return (p - [self.rect[0], self.rect[1]]) / [self.sx, self.sy]

    def img_to_page_px(self, p):
        return self.img_to_pt(p) * self.scale

    def page_px_to_img(self, p):
        return self.pt_to_img(np.asarray(p, float) / self.scale)


# ------------------------------------------------------------------------------------------------------------- walls
def _odd(n):
    return int(n) | 1


def thick_walls(gray, text_boxes=(), open_frac=0.0075, close_k=3, pad=3, min_wall_len_k=0.0):
    """Otsu -> close (joins dithered / hatched wall texture) -> square-kernel opening (keeps square wall corners exact)
    -> drop text-sized blobs lying inside OCR text boxes (bold glyphs survive the thickness filter)."""
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if t < 8 or t > 247:                       # a purely binary (0/255) image makes Otsu return a degenerate threshold: use mid-grey
        t = 128.0
    ink = (gray < t).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    k = max(5, int(round(gray.shape[1] * open_frac)))
    walls = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    removed = 0
    n0, lab0, st0, _ = cv2.connectedComponentsWithStats(walls, connectivity=8)       # thick blobs smaller than ~4 wall widths are fixtures, not walls
    for i in range(1, n0 if min_wall_len_k > 0 else 1):
        if max(st0[i, 2], st0[i, 3]) < min_wall_len_k * k:
            walls[lab0 == i] = 0
            removed += 1
    if len(text_boxes):
        n, lab, st, _ = cv2.connectedComponentsWithStats(walls, connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            for (x0, y0, x1, y1) in text_boxes:
                # the whole blob lies inside a text box (so it is a glyph cluster, not a wall that merely passes under a label)
                if x >= x0 - pad and y >= y0 - pad and x + w <= x1 + pad and y + h <= y1 + pad:
                    walls[lab == i] = 0
                    removed += 1
                    break
    ink_nt = ink.copy()
    for (x0, y0, x1, y1) in text_boxes:                        # text is not structure: also remove it from the thin-ink layer
        sl = (slice(max(0, int(y0) - pad), int(y1) + pad), slice(max(0, int(x0) - pad), int(x1) + pad))
        ink_nt[sl] = np.where(walls[sl] > 0, ink_nt[sl], 0)
    return dict(ink=ink, ink_nt=ink_nt, walls=walls, otsu=float(t), k=k, text_blobs_removed=removed)


def plan_footprint(ink_nt, walls, wall_share=0.03, thin_diag_frac=0.1, close_frac=0.12, rectilinear=True):
    """Outline of the drawing = filled hull of the wall cluster (+ large thin outlines attached to it, e.g. veranda frames).
    Title text / logo boxes are NOT part of it because they are not connected to the walls."""
    H, W = walls.shape
    thin = cv2.bitwise_and(ink_nt, cv2.bitwise_not(cv2.dilate(walls, np.ones((3, 3), np.uint8))))
    n, lab, st, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    near_walls = cv2.dilate(walls, np.ones((9, 9), np.uint8))
    big = np.zeros_like(walls)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if np.hypot(w, h) >= thin_diag_frac * W and (near_walls[lab == i] > 0).any():     # a large thin outline counts only if it is attached to the walls
            big[lab == i] = 1                                                             # (veranda frames yes; a logo box standing near the plan no)
    cluster = cv2.bitwise_or(walls, big)
    k = _odd(max(9, round(close_frac * W)))
    shape = cv2.MORPH_RECT if rectilinear else cv2.MORPH_ELLIPSE            # a square kernel keeps the hull edges straight / axis-aligned
    pad = k + 2                                                             # pad first: morphology treats the outside of the image as 'set',
    cluster_p = cv2.copyMakeBorder(cluster, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)   # which would fuse walls near the edge into a frame
    closed = cv2.morphologyEx(cluster_p, cv2.MORPH_CLOSE, cv2.getStructuringElement(shape, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    walls_p = cv2.copyMakeBorder(walls, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    tot = max(int(walls.sum()), 1)
    keep = np.zeros_like(closed)
    for i in range(1, n):
        if walls_p[lab == i].sum() >= wall_share * tot:
            keep[lab == i] = 1
    inv = (keep == 0).astype(np.uint8)
    ff = inv.copy()
    cv2.floodFill(ff, np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), np.uint8), (0, 0), 2)
    ff = ff[pad:-pad, pad:-pad]
    return (ff != 2).astype(np.uint8)


# ------------------------------------------------------------------------------------------------------- gap closure
def _neighbours(sk):
    return cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk.astype(np.uint8)


def wall_endpoints(walls, k, min_branch_k=2.0, dir_len_k=3.0, rectilinear_only=True, snap_deg=15):
    """Skeleton end points of the wall mask with a local outward direction; short spurs (< min_branch_k * wall thickness) are pruned."""
    sk = skeletonize(walls > 0)
    nb = _neighbours(sk)
    eps = np.argwhere(sk & (nb == 1))
    H, W = sk.shape
    out = []
    for y, x in eps:
        path = [(x, y)]
        prev, cur = None, (x, y)
        junction = False
        for _ in range(int(dir_len_k * k) + 2):
            cx, cy = cur
            nxt = [(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy) and 0 <= cx + dx < W and 0 <= cy + dy < H
                   and sk[cy + dy, cx + dx] and (cx + dx, cy + dy) != prev and (cx + dx, cy + dy) not in path]
            if not nxt:
                break
            if len(nxt) > 1 or nb[nxt[0][1], nxt[0][0]] >= 3:
                junction = True
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
        if junction and len(path) < min_branch_k * k:
            continue                                  # a spur of a jagged wall edge, not a real wall end
        if len(path) < k:
            continue
        P = np.array(path, float)
        d = P[0] - P[min(len(P) - 1, int(len(P) * 0.6))]
        d = d / (np.linalg.norm(d) + 1e-9)
        snapped = False
        for ax in (np.array([1.0, 0]), np.array([0, 1.0])):               # rectilinear walls: snap to the axis within snap_deg
            for sgn in (1, -1):
                if np.dot(d, sgn * ax) > np.cos(np.deg2rad(snap_deg)):
                    d, snapped = sgn * ax, True
        if rectilinear_only and not snapped:
            continue                                                      # a skeleton end that does not run along an axis is not a wall end
        out.append((np.array([x, y], float), d))
    return out


def close_wall_gaps(walls, k, W, door_frac=0.11, pair_frac=0.27, rectilinear_only=True, corner_closure=True):
    """Extend every wall end along its own axis until it meets a wall:
       * gap <= door_frac*W: any wall hit closes it (door threshold);
       * gap  > door_frac*W (up to pair_frac*W): only if it terminates at another wall END (facing collinear end, or the corner of a
         perpendicular wall) - i.e. a wall line continued across an opening up to the next wall corner.
    Returns (closed wall mask, list of closure dicts). Parallel walls are never bridged (rays run along the wall axis only)."""
    H, Wd = walls.shape
    eps = wall_endpoints(walls, k, rectilinear_only=rectilinear_only)
    closed = walls.copy()
    lines = []
    if not eps:
        return closed, lines
    E = np.array([e for e, _ in eps])
    D = np.array([d for _, d in eps])
    max_gap = pair_frac * W
    dtw = cv2.distanceTransform(walls.astype(np.uint8), cv2.DIST_L2, 3)
    for i, (e, d) in enumerate(eps):
        half = float(dtw[int(e[1]), int(e[0])])              # the skeleton end lies half a wall thickness inside the end cap
        t0 = half + 1.5
        hit = None
        for t in np.arange(t0, max_gap, 1.0):
            p = e + d * t
            x, y = int(round(p[0])), int(round(p[1]))
            if not (0 <= x < Wd and 0 <= y < H):
                break
            if walls[y, x]:
                hit = (t, p)
                break
        if hit is None:
            continue
        t, p = hit
        gap = t - t0
        if gap < 2.0:
            continue                                         # touches / overlaps the wall body: nothing to close
        kind = "door_threshold"
        if gap > door_frac * W:
            rel = E - p
            lat = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
            near_end = (np.linalg.norm(rel, axis=1) <= k + 4) & (lat <= k + 4)
            facing = near_end & (np.einsum("ij,j->i", D, d) < -0.8)
            if not near_end.any():
                continue
            if not facing.any() and not corner_closure:
                continue
            kind = "opening_between_wall_ends" if facing.any() else "opening_to_wall_corner"
        a = e + d * half          # convention: a virtual boundary across an opening lies on the wall centreline (threshold), real walls keep their faces
        cv2.line(closed, (int(round(a[0])), int(round(a[1]))), (int(round(p[0])), int(round(p[1]))), 1, max(3, int(round(half))))   # ~half the wall thickness: the opening boundary runs along the wall CENTRELINE (see review_v4.md)
        lines.append(dict(p0=[round(float(a[0]), 1), round(float(a[1]), 1)], p1=[round(float(p[0]), 1), round(float(p[1]), 1)],
                          gap_px=round(float(gap), 1), kind=kind))
    return closed, lines


# ------------------------------------------------------------------------------------------------------- polygons
def mask_to_polygon(mask, simplify=0.8, min_hole_area=150, grow=0.5):
    """Pixel-exact outline: contour of the mask pushed out by half a pixel (contours run through pixel centres), then a light
    simplification. Holes are kept when >= min_hole_area px^2."""
    m = np.pad((mask > 0).astype(np.uint8), 1)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    tops = [i for i, h in enumerate(hier[0]) if h[3] == -1]
    big = max(tops, key=lambda i: cv2.contourArea(cnts[i]))
    holes = [c[:, 0, :].astype(float) - 1 for i, (c, h) in enumerate(zip(cnts, hier[0])) if h[3] == big and cv2.contourArea(c) >= min_hole_area]
    poly = Polygon(cnts[big][:, 0, :].astype(float) - 1, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.geom_type != "Polygon":
            poly = max(poly.geoms, key=lambda g: g.area)
    if grow:
        poly = poly.buffer(grow, join_style=2)
    poly = poly.simplify(simplify, preserve_topology=True)
    return poly if (poly.geom_type == "Polygon" and not poly.is_empty) else None


# ------------------------------------------------------------------------------------------------------- partition
def _geodesic_partition(free_comp, seeds_xy, alpha=40.0):
    dt = cv2.distanceTransform(free_comp.astype(np.uint8), cv2.DIST_L2, 3)
    cost = np.full(free_comp.shape, np.inf)
    cost[free_comp] = 1.0 + alpha / (dt[free_comp] + 1.0)
    best = np.full(free_comp.shape, np.inf)
    lab = np.zeros(free_comp.shape, np.int32)
    for i, (x, y) in enumerate(seeds_xy, 1):
        cum, _ = MCP_Geometric(cost, fully_connected=True).find_costs([(y, x)])
        upd = cum < best
        best[upd] = cum[upd]
        lab[upd] = i
    lab[~np.isfinite(best)] = 0
    return lab


def segment_rooms(gray, seeds, text_boxes=(), cfg=None):
    """seeds: list of dict(xy=(x, y) in image px, cls=str, text=str). Returns dict(rooms, walls, closed, footprint, closures, info).
    Every room: dict(poly=shapely Polygon in image px, seed_ids=[...], kind='single'|'split'|'unlabelled', open_boundary_frac)."""
    cfg = cfg or {}
    H, W = gray.shape
    tw = thick_walls(gray, text_boxes, cfg.get("open_frac", 0.0075), min_wall_len_k=cfg.get("min_wall_len_k", 0.0))
    walls, k = tw["walls"], tw["k"]
    footprint = plan_footprint(tw["ink_nt"], walls, close_frac=cfg.get("footprint_close_frac", 0.12), rectilinear=cfg.get("rectilinear_walls", True))
    closed, closures = close_wall_gaps(walls, k, W, cfg.get("door_gap_frac", 0.11), cfg.get("pair_gap_frac", 0.27), cfg.get("rectilinear_walls", True), cfg.get("corner_closure", True))
    free = (footprint > 0) & (closed == 0)
    n, comp, st, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=4)
    # snap seeds into free space
    placed, outside = [], []
    for i, s in enumerate(seeds):
        x, y = int(round(s["xy"][0])), int(round(s["xy"][1]))
        if not (0 <= x < W and 0 <= y < H):
            continue
        if not free[y, x]:
            y0, x0 = max(0, y - 14), max(0, x - 14)
            ys, xs = np.nonzero(free[y0:y + 15, x0:x + 15])
            if not len(xs):
                outside.append((i, x, y))
                continue
            j = np.argmin((xs + x0 - x) ** 2 + (ys + y0 - y) ** 2)
            x, y = x0 + int(xs[j]), y0 + int(ys[j])
        placed.append((i, x, y, int(comp[y, x])))
    by_comp = {}
    for i, x, y, c in placed:
        by_comp.setdefault(c, []).append((i, x, y))
    rooms = []
    for c, ss in by_comp.items():
        m = comp == c
        if len(ss) == 1:
            rooms.append(dict(mask=m, seed_ids=[ss[0][0]], kind="single", open_boundary_frac=0.0))
            continue
        lab = _geodesic_partition(m, [(x, y) for _, x, y in ss], cfg.get("neck_penalty", 40.0))
        for j, (i, x, y) in enumerate(ss, 1):
            mj = lab == j
            # fraction of this room's boundary that lies in free space shared with another seed's basin (no wall there)
            edge = mj & ~cv2.erode(mj.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            nb = cv2.dilate(((lab > 0) & (lab != j)).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            rooms.append(dict(mask=mj, seed_ids=[i], kind="split", open_boundary_frac=round(float((edge & nb).sum()) / max(int(edge.sum()), 1), 2)))
    # labels outside the footprint (e.g. a veranda bounded only by a thin / dashed outline): bound the region by ALL non-text ink with the
    # dash gaps bridged, accept it only if it is enclosed (does not touch the image border) and is a plausible size
    if outside:
        kb = _odd(max(9, round(cfg.get("outline_bridge_frac", 0.03) * W)))
        barrier = cv2.morphologyEx(cv2.bitwise_or(tw["ink_nt"], closed), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kb, kb)))
        n2, comp2, st2, _ = cv2.connectedComponentsWithStats(((barrier == 0) & (footprint == 0)).astype(np.uint8), connectivity=4)
        used = set()
        for i, x, y in outside:
            c2 = int(comp2[y, x])
            if c2 == 0 or c2 in used:
                continue
            x0_, y0_, w_, h_, a_ = st2[c2]
            if x0_ == 0 or y0_ == 0 or x0_ + w_ >= W or y0_ + h_ >= H or a_ > cfg.get("max_outside_room_frac", 0.15) * H * W:
                continue
            used.add(c2)
            m2 = comp2 == c2
            m2 = cv2.dilate(m2.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (closed == 0)     # give back the bridged outline pixels
            rooms.append(dict(mask=m2, seed_ids=[i], kind="bounded_by_thin_outline", open_boundary_frac=0.0))
    min_a = cfg.get("min_unlabelled_area_frac", 0.0008) * H * W
    seeded = set(by_comp)
    for c in range(1, n):
        if c in seeded:
            continue
        x, y, w, h, a = st[c]
        if a < min_a or x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        m = comp == c
        if cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3).max() < 8:
            continue
        # an unlabelled space whose boundary is mostly the OUTLINE HULL (not a wall) is outside space (recess, yard) - its shape is an
        # artefact of the hull, so it is not reported; a closet on the plan edge that is walled in on all sides is kept
        m8 = m.astype(np.uint8)
        edge = (m8 > 0) & ~cv2.erode(m8, np.ones((3, 3), np.uint8)).astype(bool)
        to_outside = cv2.dilate((footprint == 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        on_wall = cv2.dilate(closed, np.ones((7, 7), np.uint8)) > 0
        open_frac = float((edge & to_outside & ~on_wall).sum()) / max(int(edge.sum()), 1)
        if open_frac > cfg.get("max_open_hull_frac", 0.15):
            continue
        rooms.append(dict(mask=m, seed_ids=[], kind="unlabelled", open_boundary_frac=0.0))
    out = []
    for r in rooms:
        poly = mask_to_polygon(r["mask"], cfg.get("simplify_px", 0.8), cfg.get("min_hole_area_px", 150))
        if poly is None or poly.area < 120:
            continue
        r["poly"] = poly
        out.append(r)
    return dict(rooms=out, walls=walls, closed=closed, footprint=footprint, closures=closures, ink=tw["ink"], info=dict(
        otsu=tw["otsu"], wall_kernel_px=k, text_blobs_removed=tw["text_blobs_removed"], n_closures=len(closures),
        footprint_frac=round(float(footprint.mean()), 3)))
