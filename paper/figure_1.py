"""Render the overview using the bus edit and matched-photo answer transitions."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from analysis.paths import ROOT, INPUT, output_dir

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath
from matplotlib.path import Path as MPath
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, PathPatch
from matplotlib.colors import to_hex

from paper import plot_style as ps  # noqa: E402

INK = getattr(ps, 'INK', '#20374C')
MAIN = getattr(ps, 'MAIN', '#3188B8')
LATER = getattr(ps, 'LATER', '#199A86')
CORAL = getattr(ps, 'CORAL', '#E77965')
MUTED = getattr(ps, 'MUTED', '#526677')
LATER_INK = getattr(ps, 'LATER_INK', LATER)  # darker teal for small later-entry text
MAIN_INK = '#1F6E99'   # darker blue for small main-entry text (bars and lines keep MAIN)
EDGE = '#8DA3B5'       # every grey line and frame
BAND = '#EEF3F7'       # the one light fill: the language-model band
TOKEN = '#ABB8C4'      # the one mid grey: text tokens and the control-key chip
GRID = '#D5DDE5'
LAYER_FILL = GRID      # the language model's layer tiles reuse the gridline grey
LAYER_STYLES = ('numbered', 'labelled', 'filled', 'digits')  # model layout 'current' only. numbered (default): L1, L2, L3;
#                      labelled: the rotated word 'layer' in each tile; filled: plain tiles;
#                      digits (preview): 1, 2, 3 in the tiles, otherwise as numbered
DEFAULT_LAYER_STYLE = 'numbered'
WORD_CLEAR = 1.5       # min gap from a tile word to the tile's top and bottom (inner stroke)
LABEL_CLEAR = 1.5      # min gap from a tile label (L1, L2, L3) to the tile's left and right (inner stroke)
MODEL_LAYOUTS = ('row2', 'current')   # row2 (default, installed): 'Layer 1', 'Layer 2'; current: three tiles
DEFAULT_MODEL_LAYOUT = 'row2'
ROW2_LABEL_LR = 2.0    # row2: min gap from a tile word's visible glyphs to the inner stroke, left and right
ROW2_LABEL_TB = 1.5    # row2: min gap above the caps and below the descender
IMG_TOKEN = '#CFE3F0'  # main image tokens (MAIN tint)

W, H = 495.0, 160.0
CAP = 0.716            # Arial cap height per pt of font size

# Line system
LW_HAIR, LW_CTRL, LW_REG, LW_EMPH = 0.5, 0.6, 0.8, 1.1
LW_EDIT = 1.3          # M11: the edit arrow class (merged layout only): the arrow(s) between the photographs
ALLOWED_LW = (LW_HAIR, LW_CTRL, LW_REG, LW_EMPH, LW_EDIT)
HEADS = {LW_REG: (2.6, 2.4), LW_EMPH: (3.2, 3.0), LW_EDIT: (4.4, 4.0)}   # line weight -> (length, width) of the head
BASE_HEADS = {HEADS[LW_REG], HEADS[LW_EMPH]}   # heads every layout uses; the edit head only in the merged layout
TIP_GAP = 0.6          # tip to the target's outer stroke edge
HEAD_TUCK = 0.2        # the shaft ends this far under its own head (hidden; avoids a seam)
CORAL_DASH = (2.4, 1.4)
FRAME_MAX = LW_REG     # heaviest thumbnail frame; marks are inset against it on every thumbnail
MARK_INSET = 1.5       # min gap between a mark's outer stroke and the frame's inner stroke
ALLOWED_FONTS = (8.5, 7.5, 6.5, 6.0)
THUMB = 56.0
ZONE_B_DX = 9.0       # panel b drawn this far right of its original place (see docstring)

# Zoomed crop of the edited object beside each zone a thumbnail (spec key 'zoom'). The crop is
# canvas pixels of the same d_a / d_b images the thumbnails show, embedded at native pixels; it
# holds the whole changed region (d_a != d_b: x 161-258, y 160-187) with a 3 px margin.
ZOOM_FRAME = 'solid'   # 'solid' (thin 0.8 pt coral frame, chosen) or 'dashed' (the 1.1 pt coral marker)
ZOOM_PERIODS = 15      # dashed frame only: width in whole coral dash periods (height follows the crop)
ZOOM_GAP = 3.5         # solid frame: thumbnail outer stroke to zoom outer left edge (width follows)
ZOOM_RIGHT = None      # outer right edge of the zoom frames; None = ink end of heading a
ZOOM_LABEL_GAP = 3.0   # 'Source a/b' baseline to the zoom frame's outer top edge
ZOOM_CLEAR = 1.0       # min gap from zoom frames and leaders to text and to unrelated marks

# ---------------------------------------------------------------- data
# Reference photograph cell: P1, layout M0, full background (alpha = 1), raw strength,
# direct generation, each model's common complete-photograph set. Counts are target
# answers that change toward b (blue); no change runs from b to a in these cells.
NATURAL = ROOT / 'datasets/natural-selectivity-v12'
COUNTS = {}

VARIANTS = {
    'photo': {
        'stem': 'fig1_overview',
        'title_a': 'Two versions of one photo',
        'title_merged': 'One photo, a different version at each entry',
        'manifest': NATURAL / 'manifest.jsonl',
        # Visual Genome 2390681 = Flickr "River Brent at Brentford", Maxwell Hamilton, CC BY 2.0.
        # (The earlier hydrant photo vg-2406297-edit-0 is CC BY-NC-SA 2.0 and is not used.)
        'intervention': 'vg-2390681-edit-0',
        # Square crop of the 384 x 384 canvas (pixel_half_open_xyxy) holding the bus on the
        # bridge (edited, box [156,159,261,192]) and the boat below it (control, [140,235,257,335]);
        # the bottom edge is the end of the photo (letterbox starts at y = 336).
        'crop': (112, 150, 298, 336),
        'object': 'bus',
        'mark_control': True,
        'q_target': 'What color is the bus?',
        'q_control': 'What color is the boat?',
        'xlabel': 'Target answers turned blue (%)',
        'credit': ('Photo: River Brent at Brentford, Maxwell Hamilton,',
                   'CC BY 2.0, cropped and recolored'),
        # Main keeps the original photograph; later gets it with the bus's tokens from b.
        'main_source': 'm0_s100',
        'main_label': 'original photo',
        'later_label': r'original, bus from $\mathit{b}$',
        # merged (M6F): the label on the transformation arrow between the two photographs
        # M9: plain words, no source letters: what the edit does to the bus
        'edit_label': 'bus recolored',
        # M11B: the label of the control arrow (merged, control_arrow True)
        'control_label': 'boat unchanged',
        # merged (M9): chart header in plain words (the abc layout keeps 'route_header')
        'route_header_merged': 'Object recolored only at',
        # merged (M6G): what the later entries receive, set after their name like the main label
        'later_sub': 'edited photo',
        # Panel c header over the per-model rows; each row is one condition.
        # The bars pool all 128 photograph edits, not this one bus.
        'route_header': r'Edited object from $\mathit{b}$ only at',
        # Zoomed crop of the bus beside each zone a thumbnail (canvas pixels, half-open xyxy).
        'zoom': True,
        'zoom_canvas': (158, 157, 262, 190),
    },

}


def pct(c, n):
    """Round half up, as in the manuscript's 5%, 63%, 65%, 74%."""
    return int(100 * c / n + 0.5)


def load_counts():
    data = json.loads((INPUT / 'components.json').read_text())['p1_components']
    for label, model in [('DeepStack', 'deepstack7b'), ('Qwen8', 'qwen8b'), ('Qwen32', 'qwen32b')]:
        rows = data[model]['common_complete_counts']
        COUNTS[label] = {entry: (rows[entry+'_target']['forward'], rows[entry+'_target']['query_pairs'])
                         for entry in ('main', 'later')}


def load_item(spec):
    rows = [json.loads(line) for line in spec['manifest'].open()]
    hits = [r for r in rows if r['intervention_id'] == spec['intervention']]
    changed = next(r for r in hits if r['query_role'] == 'changed' and r['query_type'] == 'identity')
    control = next((r for r in hits if r['query_role'] != 'changed' and r['query_type'] == 'identity'), None)
    root = spec['manifest'].parent
    pa, pb = root / changed['source_images']['d_a'], root / changed['source_images']['d_b']
    x0, y0, x1, y1 = spec['crop']
    img_a = np.asarray(Image.open(pa).convert('RGB'))[y0:y1, x0:x1]
    img_b = np.asarray(Image.open(pb).convert('RGB'))[y0:y1, x0:x1]
    box = changed.get('changed_bbox_by_layout', {}).get('M0') or changed['donor_roi_bbox_xyxy']
    box = (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0)  # crop pixel coordinates
    cbox = (control or {}).get('queried_bbox_by_layout', {}).get('M0') if spec.get('mark_control') else None
    if cbox:
        cbox = (cbox[0] - x0, cbox[1] - y0, cbox[2] - x0, cbox[3] - y0)
    info = {'intervention_id': spec['intervention'], 'query_target': changed['query'],
            'query_control': control['query'] if control else None,
            'd_a': str(pa), 'd_b': str(pb),
            'sha256_d_a': hashlib.sha256(pa.read_bytes()).hexdigest(),
            'sha256_d_b': hashlib.sha256(pb.read_bytes()).hexdigest(),
            'crop_xyxy': list(spec['crop']), 'edited_box_crop_xyxy': list(box),
            'control_box_crop_xyxy': list(cbox) if cbox else None,
            'source': changed.get('source')}
    img_main = img_a
    if spec.get('main_source'):
        pm = root / changed['source_images'][spec['main_source']]
        img_main = np.asarray(Image.open(pm).convert('RGB'))[y0:y1, x0:x1]
        # Drawing d_b as the later input must equal "main with the object's box from b".
        outside = np.ones(img_main.shape[:2], bool)
        outside[box[1]:box[3], box[0]:box[2]] = False
        assert np.array_equal(img_main[outside], img_b[outside]), 'main and b differ outside the box'
        info.update({'main_source': str(pm),
                     'sha256_main_source': hashlib.sha256(pm.read_bytes()).hexdigest()})
    return img_main, img_a, img_b, box, cbox, info


def load_zoom(spec, info):
    """Zoom crops of d_a and d_b (same files as the thumbnails, native pixels, no resampling),
    and the canvas bbox of the pixels that differ between them (half-open xyxy)."""
    zx0, zy0, zx1, zy1 = spec['zoom_canvas']
    full_a = np.asarray(Image.open(info['d_a']).convert('RGB'))
    full_b = np.asarray(Image.open(info['d_b']).convert('RGB'))
    diff = np.any(full_a != full_b, axis=2)
    ys, xs = np.nonzero(diff)
    changed = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return full_a[zy0:zy1, zx0:zx1], full_b[zy0:zy1, zx0:zx1], changed


def seg_box_gap(p0, p1, b, n=4000):
    """Smallest distance (pt) from the segment p0-p1 to the box b (0 when they touch or cross)."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = np.asarray(p0, float) * (1 - t) + np.asarray(p1, float) * t
    dx = np.maximum.reduce([b[0] - pts[:, 0], np.zeros(n), pts[:, 0] - b[2]])
    dy = np.maximum.reduce([b[1] - pts[:, 1], np.zeros(n), pts[:, 1] - b[3]])
    return float(np.min(np.hypot(dx, dy)))


# ---------------------------------------------------------------- drawing helpers
def setup_rc():
    ps.apply_paper_style()
    plt.rcParams.update({
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold', 'mathtext.bfit': 'Arial:italic:bold',
        'pdf.fonttype': 42, 'axes.unicode_minus': True,
        # Dash lengths are given in points, not in multiples of the line width.
        'lines.scale_dashes': False,
        # Lines end where they are told to (matplotlib's default cap projects lw/2).
        'lines.solid_capstyle': 'butt', 'lines.dash_capstyle': 'butt',
    })


def cap(size):
    return CAP * size


def text(ax, x, y, s, size=7.5, color=INK, weight='normal', ha='left', va='baseline', **kw):
    return ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, **kw)


def ink_box(s, size, weight='normal'):
    """Visible-glyph bbox (pt) of s set at the pen origin (0, 0) on the baseline."""
    bb = TextPath((0, 0), s, prop=FontProperties(size=size, weight=weight)).get_extents()
    return bb.x0, bb.y0, bb.x1, bb.y1


def ink_centred(ax, cx, cy, s, size, color=INK, **kw):
    """Text whose visible glyphs (not the font's line box) are centred on (cx, cy)."""
    x0, y0, x1, y1 = ink_box(s, size)
    return text(ax, cx - (x0 + x1) / 2, cy - (y0 + y1) / 2, s, size=size, color=color,
                ha='left', va='baseline', **kw)


def heading(ax, x, base, letter, title):
    text(ax, x, base, letter, size=8.5, weight='bold')
    return text(ax, x + 8.8, base, title, size=8.5, weight='bold')


def line(ax, xs, ys, color=EDGE, lw=LW_REG, zorder=2):
    ln = Line2D(xs, ys, color=color, lw=lw, solid_capstyle='butt', zorder=zorder)
    ax.add_line(ln)
    return ln


def photo(ax, img, x, y, w, h, frame, lw):
    ax.imshow(img, extent=(x, x + w, y, y + h), interpolation='none', zorder=2)
    fr = Rectangle((x, y), w, h, fill=False, lw=lw, edgecolor=frame, joinstyle='miter', zorder=3)
    ax.add_patch(fr)
    return fr


def dashed_box(ax, x0, y0, w, h, color=CORAL, lw=LW_EMPH, zorder=4):
    """Coral dashed rectangle. Sides are whole dash periods, so each corner carries one centred
    L-shaped dash (1.2 pt per arm); the path starts and ends mid-gap, so no dash is split."""
    on, off = CORAL_DASH
    g = on / 2 + off / 2                       # lower-left corner to the middle of the first gap
    verts = [(x0 + g, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h), (x0, y0), (x0 + g, y0)]
    p = PathPatch(MPath(verts, [MPath.MOVETO] + [MPath.LINETO] * 5), fill=False, lw=lw, edgecolor=color,
                  linestyle=(on / 2 + g, CORAL_DASH), joinstyle='miter', capstyle='butt', zorder=zorder)
    ax.add_patch(p)
    return p


def snap_span(lo, hi, lo_min, hi_max):
    """Resize [lo, hi] to a whole number of dash periods inside [lo_min, hi_max].
    A side that was clamped stays where it is; otherwise the span keeps its centre."""
    period = sum(CORAL_DASH)
    n = max(2, int(round((hi - lo) / period)))
    while n > 2 and n * period > hi_max - lo_min + 1e-9:
        n -= 1
    span = n * period
    if lo <= lo_min + 1e-9:
        new = lo_min
    elif hi >= hi_max - 1e-9:
        new = hi_max - span
    else:
        new = (lo + hi) / 2 - span / 2
    new = min(max(new, lo_min), hi_max - span)
    return new, new + span


def mark(ax, box, img_shape, x, y, w, h, color, lw, pad, dashed):
    """Box around an object (crop pixels -> figure points), kept MARK_INSET inside the frame."""
    ph, pw = img_shape[:2]
    sx, sy = w / pw, h / ph
    bx0, by0, bx1, by1 = box
    inset = FRAME_MAX / 2 + MARK_INSET + lw / 2
    rx0 = max(x + bx0 * sx - pad, x + inset)
    rx1 = min(x + bx1 * sx + pad, x + w - inset)
    ry0 = max(y + h - by1 * sy - pad, y + inset)
    ry1 = min(y + h - by0 * sy + pad, y + h - inset)
    if dashed:
        rx0, rx1 = snap_span(rx0, rx1, x + inset, x + w - inset)
        ry0, ry1 = snap_span(ry0, ry1, y + inset, y + h - inset)
        return dashed_box(ax, rx0, ry0, rx1 - rx0, ry1 - ry0, color, lw)
    p = Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False, lw=lw, edgecolor=color,
                  joinstyle='miter', zorder=4)
    ax.add_patch(p)
    return p


def rounded_path(pts, r):
    """Poly-line through pts with circular elbows of radius r at the interior vertices."""
    pts = [np.asarray(p, float) for p in pts]
    verts, codes = [pts[0]], [MPath.MOVETO]
    for p0, p1, p2 in zip(pts, pts[1:], pts[2:]):
        l0, l1 = np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1)
        u0, u1 = (p1 - p0) / l0, (p2 - p1) / l1
        turn = np.arccos(np.clip(np.dot(u0, u1), -1.0, 1.0))
        if r <= 0 or turn < 1e-6:
            verts.append(p1)
            codes.append(MPath.LINETO)
            continue
        t = min(r * np.tan(turn / 2), l0 / 2, l1 / 2)      # tangent length
        rr = t / np.tan(turn / 2)
        k = 4.0 / 3.0 * np.tan(turn / 4) * rr              # cubic handle for a circular arc
        a, b = p1 - u0 * t, p1 + u1 * t
        verts += [a, a + u0 * k, b - u1 * k, b]
        codes += [MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
    verts.append(pts[-1])
    codes.append(MPath.LINETO)
    return MPath(np.array(verts), codes)


def arrow(ax, reg, name, pts, color, lw, target, radius=0.0, zorder=5, start_inset=0.0):
    """Shaft (poly-line, optional rounded elbows) plus a separate solid head at pts[-1].

    pts[-1] is the tip. The head is a filled triangle with no outline, sized by the line
    weight (HEADS); the shaft stops at the head's base (tucked HEAD_TUCK under it).
    start_inset: how much of the shaft's start is hidden under a same-coloured stroke.
    """
    pts = [np.asarray(p, float) for p in pts]
    tip = pts[-1]
    d = (tip - pts[-2]) / np.linalg.norm(tip - pts[-2])
    hl, hw = HEADS[lw]
    base = tip - hl * d
    shaft = PathPatch(rounded_path(pts[:-1] + [tip - (hl - HEAD_TUCK) * d], radius), fill=False,
                      lw=lw, edgecolor=color, capstyle='butt', joinstyle='round', zorder=zorder)
    ax.add_patch(shaft)
    n = np.array([-d[1], d[0]])
    head = Polygon([tip, base + n * hw / 2, base - n * hw / 2], closed=True, facecolor=color,
                   edgecolor='none', lw=0, zorder=zorder + .5)
    ax.add_patch(head)
    seg = np.linalg.norm(base - pts[-2])
    reg['arrows'].append({'name': name, 'head': head, 'shaft': shaft, 'dir': d, 'target': target, 'pts': pts,
                          'lw': lw, 'visible_last_segment': float(seg - (start_inset if len(pts) == 2 else 0))})
    return head


def outer_bbox(art, r=None):
    """Data-space (= pt) bbox of an artist including half its visible stroke."""
    if isinstance(art, Line2D):
        xs, ys = (np.asarray(v, float) for v in art.get_data())
        h = art.get_linewidth() / 2
        return (xs.min() - h, ys.min() - h, xs.max() + h, ys.max() + h)
    if isinstance(art, matplotlib.text.Text):
        bb = art.get_window_extent(r)
        k = 72.0 / art.figure.dpi
        return (bb.x0 * k, bb.y0 * k, bb.x1 * k, bb.y1 * k)
    if isinstance(art, matplotlib.image.AxesImage):
        x0, x1, y0, y1 = art.get_extent()
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    bb = art.get_path().get_extents(art.get_patch_transform())
    h = art.get_linewidth() / 2 if has_stroke(art) else 0.0
    return (bb.x0 - h, bb.y0 - h, bb.x1 + h, bb.y1 + h)


def has_stroke(art):
    if isinstance(art, Line2D):
        return art.get_linestyle() not in ('None', 'none', '', ' ') and art.get_linewidth() > 0
    ec = art.get_edgecolor()
    return art.get_linewidth() > 0 and ec is not None and ec[3] > 0


def sep(b1, b2):
    """Separation of two boxes (negative when they overlap); inflate-by-t test: sep < t."""
    return max(b2[0] - b1[2], b1[0] - b2[2], b2[1] - b1[3], b1[1] - b2[3])


def check_text_layout(fig):
    """Report text that leaves the canvas or overlaps other text (boxes in points)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = 72.0 / fig.dpi
    boxes = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_text() or not t.get_visible():
            continue
        bb = t.get_window_extent(r)
        boxes.append((t.get_text(), bb.x0 * k, bb.y0 * k, bb.x1 * k, bb.y1 * k))
    issues = []
    for s, x0, y0, x1, y1 in boxes:
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            issues.append(f'outside canvas: {s!r} ({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})')
    for i, (s, x0, y0, x1, y1) in enumerate(boxes):
        for s2, u0, v0, u1, v1 in boxes[i + 1:]:
            if min(x1, u1) - max(x0, u0) > .3 and min(y1, v1) - max(y0, v0) > .3:
                issues.append(f'overlap: {s!r} / {s2!r}')
    for msg in issues:
        print('layout:', msg, file=sys.stderr)
    return issues


# ---------------------------------------------------------------- acceptance checks
def tip_clearance(tip, d, target):
    """Distance from the tip to the target's outer stroke edge, along the arrow direction."""
    h = target.get_linewidth() / 2
    if isinstance(target, Circle):
        return float(np.hypot(*(tip - np.asarray(target.center))) - (target.radius + h)), True
    bb = target.get_path().get_extents(target.get_patch_transform())
    if abs(d[0]) < 1e-9 and d[1] < 0:
        return float(tip[1] - (bb.y1 + h)), bb.x0 <= tip[0] <= bb.x1
    if abs(d[0]) < 1e-9 and d[1] > 0:
        return float((bb.y0 - h) - tip[1]), bb.x0 <= tip[0] <= bb.x1
    if abs(d[1]) < 1e-9 and d[0] > 0:
        return float((bb.x0 - h) - tip[0]), bb.y0 <= tip[1] <= bb.y1
    if abs(d[1]) < 1e-9 and d[0] < 0:
        return float(tip[0] - (bb.x1 + h)), bb.y0 <= tip[1] <= bb.y1
    raise ValueError('non-orthogonal arrow into a box')


def run_checks(fig, ax, reg, zones):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text() and t.get_visible()]
    out = {}

    # 1. Arrowhead bounding boxes against blocks, slot, nodes, boxes, frames, other heads, text.
    obstacles = [(n, outer_bbox(a)) for n, a in reg['obstacles']]
    obstacles += [(f'text {t.get_text()!r}', outer_bbox(t, r)) for t in texts]
    rows, ok1 = [], True
    for a in reg['arrows']:
        hb = outer_bbox(a['head'])
        others = obstacles + [(f"head {b['name']}", outer_bbox(b['head']))
                              for b in reg['arrows'] if b is not a]
        g, who = min(((sep(hb, ob), n) for n, ob in others), key=lambda v: v[0])
        good = g >= 0.2
        ok1 &= good
        rows.append({'arrow': a['name'], 'head_bbox': [round(v, 2) for v in hb],
                     'nearest': who, 'gap_pt': round(g, 3), 'pass': good})
    out['1_head_bboxes'] = {'pass': ok1, 'tolerance_pt': 0.2, 'heads': rows}

    # 2. Tip clearance 0.6 +- 0.05, measured from the head polygon's tip vertex.
    rows, ok2 = [], True
    for a in reg['arrows']:
        tip = np.asarray(a['head'].get_xy()[0], float)
        c, aimed = tip_clearance(tip, a['dir'], a['target'])
        good = abs(c - TIP_GAP) <= 0.05 and aimed
        ok2 &= good
        rows.append({'arrow': a['name'], 'tip': [round(v, 3) for v in tip], 'clearance_pt': round(c, 3),
                     'on_target': bool(aimed), 'pass': good})
    out['2_tip_clearance'] = {'pass': ok2, 'target_pt': TIP_GAP, 'arrows': rows}

    # 3. Stroke weights, head geometries, font sizes.
    weights = {}
    for art in list(ax.patches) + list(ax.lines) + list(ax.collections):
        if art.get_visible() and has_stroke(art):
            lw = round(float(art.get_linewidth()), 3)
            weights[lw] = weights.get(lw, 0) + 1
    heads, pairs_ok = set(), True
    for a in reg['arrows']:
        v = np.asarray(a['head'].get_xy(), float)[:3]
        geo = (round(float(np.linalg.norm(v[0] - (v[1] + v[2]) / 2)), 2), round(float(np.linalg.norm(v[1] - v[2])), 2))
        heads.add(geo)
        pairs_ok &= geo == HEADS[a['lw']] and has_stroke(a['head']) is False
    fonts = sorted({round(float(t.get_fontsize()), 2) for t in texts})
    # M11: the allowed weights and heads now include the edit arrow class (LW_EDIT and its head). Every
    # head is one of HEADS and matches its arrow's weight; the heads in use are exactly the required set
    # (abc: the 0.8 and 1.1 heads, as before; merged: those plus the edit head); and the edit weight
    # appears only on the edit arrows' shafts (no other stroke is drawn at LW_EDIT).
    required_heads = reg.get('required_heads', BASE_HEADS)
    edit_ids = {id(a['shaft']) for a in reg['arrows'] if a['lw'] == LW_EDIT}
    edit_named = all(a['name'] in reg.get('edit_arrow_names', ()) for a in reg['arrows'] if a['lw'] == LW_EDIT)
    edit_only = edit_named and all(id(art) in edit_ids for art in list(ax.patches) + list(ax.lines) + list(ax.collections)
                                   if art.get_visible() and has_stroke(art)
                                   and abs(float(art.get_linewidth()) - LW_EDIT) < 1e-9)
    ok_w = set(weights) <= set(ALLOWED_LW) and edit_only
    ok_h = heads == set(required_heads) and set(required_heads) <= set(HEADS.values()) and pairs_ok
    ok_f = set(fonts) <= set(ALLOWED_FONTS)
    out['3_weights_heads_fonts'] = {
        'pass': ok_w and ok_h and ok_f,
        'stroke_weights_pt': {str(k): v for k, v in sorted(weights.items())}, 'weights_pass': ok_w,
        'edit_weight_only_on_edit_arrows': bool(edit_only),
        'required_heads_pt': sorted(required_heads),
        'head_geometries_pt': sorted(heads), 'heads_pass': ok_h,
        'font_sizes_pt': fonts, 'fonts_pass': ok_f}

    # 4. Text layout.
    issues = check_text_layout(fig)
    # Rotated words inside the layer tiles: inside the tile's inner stroke edge, with at least
    # WORD_CLEAR to its top and bottom (window extent of the rotated text, in points).
    words = []
    for word, blk in reg['tile_words']:
        wb = outer_bbox(word, r)
        tb = blk.get_path().get_extents(blk.get_patch_transform())
        hs = blk.get_linewidth() / 2
        inner = (tb.x0 + hs, tb.y0 + hs, tb.x1 - hs, tb.y1 - hs)
        top_gap, bot_gap = inner[3] - wb[3], wb[1] - inner[1]
        left_gap, right_gap = wb[0] - inner[0], inner[2] - wb[2]
        good = min(top_gap, bot_gap) >= WORD_CLEAR - 1e-6 and min(left_gap, right_gap) >= 0
        if not good:
            issues.append(f'tile word {word.get_text()!r} at x={tb.x0:.1f}: top {top_gap:.2f}, '
                          f'bottom {bot_gap:.2f}, left {left_gap:.2f}, right {right_gap:.2f} pt')
        words.append({'tile_x': round(float(tb.x0), 2), 'word_bbox': [round(v, 2) for v in wb],
                      'top_gap_pt': round(float(top_gap), 3), 'bottom_gap_pt': round(float(bot_gap), 3),
                      'left_gap_pt': round(float(left_gap), 3), 'right_gap_pt': round(float(right_gap), 3),
                      'pass': bool(good)})
    # Tile labels (current: L1, L2, L3; row2: Layer 1, Layer 2): inside the tile's inner stroke edge. For
    # current, at least LABEL_CLEAR to its left and right, tested on the union of the text's window
    # extent and its visible-glyph box; row2 has its own rule below.
    labels = []
    for lab, blk in reg['tile_labels']:
        wb = outer_bbox(lab, r)
        lx0, ly0 = lab.get_position()
        g = ink_box(lab.get_text(), lab.get_fontsize())
        ib = (lx0 + g[0], ly0 + g[1], lx0 + g[2], ly0 + g[3])
        ub = (min(wb[0], ib[0]), min(wb[1], ib[1]), max(wb[2], ib[2]), max(wb[3], ib[3]))
        tb = blk.get_path().get_extents(blk.get_patch_transform())
        hs = blk.get_linewidth() / 2
        inner = (tb.x0 + hs, tb.y0 + hs, tb.x1 - hs, tb.y1 - hs)
        left_gap, right_gap = ub[0] - inner[0], inner[2] - ub[2]
        top_gap, bot_gap = inner[3] - ub[3], ub[1] - inner[1]
        ink_dx = (ib[0] + ib[2]) / 2 - (tb.x0 + tb.x1) / 2
        ink_dy = (ib[1] + ib[3]) / 2 - (tb.y0 + tb.y1) / 2
        if reg.get('model_layout') == 'row2':
            # Same rule as 'red or blue?': the caps are centred on the row axis (cap line and baseline
            # equidistant from it) and the visible glyphs are centred horizontally on the tile. Visible
            # glyphs at least ROW2_LABEL_LR from the inner stroke left and right; at least ROW2_LABEL_TB
            # above the caps (cap line or glyph top, whichever is higher) and below the descender; the
            # text's own box (advance width, incl. the '1' side bearing) must still stay inside the stroke.
            row_axis = (tb.y0 + tb.y1) / 2
            cap_dy = (ly0 + cap(lab.get_fontsize()) / 2) - row_axis
            top_ink = max(ib[3], ly0 + cap(lab.get_fontsize()))
            gl, gr = ib[0] - inner[0], inner[2] - ib[2]
            gt, gb = inner[3] - top_ink, ib[1] - inner[1]
            good = (min(gl, gr) >= ROW2_LABEL_LR - 1e-6 and min(gt, gb) >= ROW2_LABEL_TB - 1e-6
                    and min(left_gap, right_gap, top_gap, bot_gap) >= 0
                    and abs(cap_dy) <= 0.01 and abs(ink_dx) <= 0.01)
            row2_gaps = {'glyph_left_pt': round(float(gl), 3), 'glyph_right_pt': round(float(gr), 3),
                         'above_caps_pt': round(float(gt), 3), 'below_descender_pt': round(float(gb), 3),
                         'cap_centre_offset_pt': round(float(cap_dy), 3)}
        else:
            good = min(left_gap, right_gap) >= LABEL_CLEAR - 1e-6 and min(top_gap, bot_gap) >= 0
            row2_gaps = {}
        if not good:
            issues.append(f'tile label {lab.get_text()!r} at x={tb.x0:.1f}: left {left_gap:.2f}, '
                          f'right {right_gap:.2f}, top {top_gap:.2f}, bottom {bot_gap:.2f} pt')
        labels.append({'label': lab.get_text(), 'tile_x': round(float(tb.x0), 2),
                       'text_bbox': [round(v, 2) for v in wb], 'glyph_bbox': [round(v, 2) for v in ib],
                       'left_gap_pt': round(float(left_gap), 3), 'right_gap_pt': round(float(right_gap), 3),
                       'top_gap_pt': round(float(top_gap), 3), 'bottom_gap_pt': round(float(bot_gap), 3),
                       'glyph_centre_offset_pt': [round(float(ink_dx), 3), round(float(ink_dy), 3)],
                       **row2_gaps, 'pass': bool(good)})
    out['4_text_layout'] = {'pass': not issues, 'issues': issues, 'tile_words': words,
                            'tile_word_min_top_bottom_pt': WORD_CLEAR, 'tile_labels': labels,
                            'tile_label_min_left_right_pt': LABEL_CLEAR,
                            'row2_rule': ({'basis': 'caps centred on the row axis, glyphs centred horizontally; visible glyphs '
                                                    'clear the inner stroke; text box inside inner stroke',
                                           'min_left_right_pt': ROW2_LABEL_LR, 'min_top_bottom_pt': ROW2_LABEL_TB}
                                          if reg.get('model_layout') == 'row2' else None)}

    # 5. Bare chain line beside every node; marks inside their frames.
    row_obst = [outer_bbox(a) for _, a in reg['row_obstacles']]
    rows, ok5 = [], True
    for node in reg['nodes']:
        cx, cy = node.center
        ro = node.radius + node.get_linewidth() / 2
        segs = []
        for ln in reg['chain']:
            xs, ys = (np.asarray(v, float) for v in ln.get_data())
            if np.allclose(ys, cy):
                segs.append((xs.min(), xs.max()))
        sides = {}
        for side, edge in (('left', cx - ro), ('right', cx + ro)):
            cover = [s for s in segs if s[0] - 1e-9 <= edge <= s[1] + 1e-9]
            if not cover:
                sides[side] = 0.0
                continue
            if side == 'left':
                lim = max([b[2] for b in row_obst if b[2] <= edge + 1e-9 and b[1] <= cy <= b[3]] + [min(s[0] for s in cover)])
                sides[side] = round(float(edge - lim), 3)
            else:
                lim = min([b[0] for b in row_obst if b[0] >= edge - 1e-9 and b[1] <= cy <= b[3]] + [max(s[1] for s in cover)])
                sides[side] = round(float(lim - edge), 3)
        good = min(sides.values()) >= 3.0 - 1e-6
        ok5 &= good
        rows.append({'node_x': round(cx, 2), 'bare_left_pt': sides['left'], 'bare_right_pt': sides['right'], 'pass': good})
    mrows = []
    for name, box, frame in reg['marks']:
        b = outer_bbox(box)
        fb = frame.get_path().get_extents(frame.get_patch_transform())
        fh = frame.get_linewidth() / 2
        inner = (fb.x0 + fh, fb.y0 + fh, fb.x1 - fh, fb.y1 - fh)
        m = min(b[0] - inner[0], b[1] - inner[1], inner[2] - b[2], inner[3] - b[3])
        good = m >= MARK_INSET - 1e-6
        ok5 &= good
        mrows.append({'mark': name, 'inside_pt': round(float(m), 3), 'pass': good})
    out['5_nodes_and_marks'] = {'pass': ok5, 'nodes': rows, 'marks': mrows}

    # 6. Zooms (zone a): native pixels from the thumbnail's own image, undistorted, inside their
    #    frames, equal and aligned; frames and leaders clear of all text and of every other mark.
    if reg.get('layout') == 'merged':
        out['6_no_crops'] = check_no_crops(reg, texts, ax)
    elif reg.get('zooms'):
        zi = reg['zoom_info']
        tboxes = [(f'text {t.get_text()!r}', outer_bbox(t, r)) for t in texts]
        others_all = [(n, outer_bbox(a)) for n, a in reg['obstacles']]
        for a in reg['arrows']:
            others_all += [(f"head {a['name']}", outer_bbox(a['head'])), (f"shaft {a['name']}", outer_bbox(a['shaft']))]
        zrows, ok6 = [], True
        cx0, cy0, cx1, cy1 = zi['canvas_xyxy']
        ch0 = zi['changed_xyxy']
        holds_change = cx0 <= ch0[0] and cy0 <= ch0[1] and ch0[2] <= cx1 and ch0[3] <= cy1
        ok6 &= holds_change
        frames = []
        for z in reg['zooms']:
            nm, im, zf = z['name'], z['image'], z['frame']
            arr = im.get_array()
            x0, x1, y0, y1 = im.get_extent()
            sx, sy = (x1 - x0) / arr.shape[1], (y1 - y0) / arr.shape[0]
            k0, k1 = zi['crop_xyxy'][:2]
            same_pixels = bool(np.array_equal(np.asarray(arr), z['thumb_img'][cy0 - k1:cy1 - k1, cx0 - k0:cx1 - k0]))
            native = tuple(arr.shape[:2]) == (cy1 - cy0, cx1 - cx0) and im.get_interpolation() == 'none'
            undistorted = abs(sx - sy) < 1e-9
            fb = zf.get_path().get_extents(zf.get_patch_transform())
            fh = zf.get_linewidth() / 2
            inner = (fb.x0 + fh, fb.y0 + fh, fb.x1 - fh, fb.y1 - fh)
            img_in = min(x0 - inner[0], y0 - inner[1], inner[2] - x1, inner[3] - y1)
            fo = outer_bbox(zf)
            frames.append(fb)
            own = {f'zoom frame {nm}'}
            g_txt, w_txt = min(((sep(fo, b), n) for n, b in tboxes), key=lambda v: v[0])
            g_obj, w_obj = min(((sep(fo, b), n) for n, b in others_all if n not in own), key=lambda v: v[0])
            # Leaders: start on the edit box corner, end on the zoom frame corner; they may cross only
            # their own thumbnail's frame and photo.
            allowed = own | {f'edited box source {nm}', f'frame source {nm}'}
            lrows = []
            for ln, (p0, p1) in zip(z['leaders'], z['leader_ends']):
                xs, ys = ln.get_data()
                bb = z['box'].get_path().get_extents()
                on_ends = (np.allclose([xs[0], ys[0]], p0) and np.allclose([xs[1], ys[1]], p1)
                           and any(np.allclose(p0, c) for c in ((bb.x1, bb.y1), (bb.x1, bb.y0)))
                           and any(np.allclose(p1, c) for c in ((fb.x0, fb.y1), (fb.x0, fb.y0))))
                lt, wt = min(((seg_box_gap(p0, p1, b), n) for n, b in tboxes), key=lambda v: v[0])
                lo, wo = min(((seg_box_gap(p0, p1, b), n) for n, b in others_all if n not in allowed),
                             key=lambda v: v[0])
                lgood = on_ends and lt >= ZOOM_CLEAR and lo >= ZOOM_CLEAR
                lrows.append({'from': [round(v, 3) for v in p0], 'to': [round(v, 3) for v in p1],
                              'on_box_and_frame_corners': bool(on_ends),
                              'nearest_text': wt, 'text_gap_pt': round(lt, 3),
                              'nearest_mark': wo, 'mark_gap_pt': round(lo, 3), 'pass': bool(lgood)})
            good = (same_pixels and native and undistorted and img_in >= -1e-6 and g_txt >= ZOOM_CLEAR
                    and g_obj >= ZOOM_CLEAR and all(l_['pass'] for l_ in lrows))
            ok6 &= good
            zrows.append({'zoom': nm, 'same_pixels_as_thumbnail_source': same_pixels, 'native_pixels': bool(native),
                          'undistorted': bool(undistorted), 'image_inside_frame_pt': round(float(img_in), 3),
                          'frame_outer_bbox': [round(v, 2) for v in fo],
                          'nearest_text': w_txt, 'text_gap_pt': round(g_txt, 3),
                          'nearest_mark': w_obj, 'mark_gap_pt': round(g_obj, 3), 'leaders': lrows, 'pass': bool(good)})
        same_size = all(abs((f.x1 - f.x0) - (frames[0].x1 - frames[0].x0)) < 1e-9 and
                        abs((f.y1 - f.y0) - (frames[0].y1 - frames[0].y0)) < 1e-9 for f in frames)
        aligned = all(abs(f.x0 - frames[0].x0) < 1e-9 for f in frames)
        ok6 &= same_size and aligned
        out['6_zooms'] = {'pass': bool(ok6), 'min_clear_pt': ZOOM_CLEAR, 'crop_holds_changed_region': bool(holds_change),
                          'same_size': bool(same_size), 'left_aligned': bool(aligned),
                          'magnification_vs_thumbnail': round(reg['zooms'][0]['scale_pt_per_px'] / zi['thumb_scale_pt_per_px'], 3),
                          'zooms': zrows}

    if reg.get('layout') == 'merged':
        out.update(merged_checks(fig, ax, reg, zones, texts, r))
        out['all_pass'] = all(out[k]['pass'] for k in out if k[0].isdigit())
        for k in out:
            if k[0].isdigit() and not out[k]['pass']:
                print(f'CHECK FAILED: {k}', file=sys.stderr)
        return out

    # Further measurements from the specification (reported; not among the five checks).
    dash = sorted({tuple(round(float(v), 3) for v in p._dash_pattern[1]) for p in reg['dashed']})
    dash_lw = sorted({round(float(p.get_linewidth()), 3) for p in reg['dashed']})
    ext = {z: [outer_bbox(a, r) for a in arts] for z, arts in zones.items()}
    allb = [b for bs in ext.values() for b in bs]
    margin = min(min(b[0], b[1], W - b[2], H - b[3]) for b in allb)
    zc_left = min(b[0] for b in ext['c'])
    zb_right = max(b[2] for b in ext['b'])
    title_a = outer_bbox(reg['title_a'], r)
    letter_b = outer_bbox(reg['letter_b'], r)
    extra = {
        'coral_dash_rendered_on_off_pt': dash, 'coral_dash_line_weights_pt': dash_lw,
        'coral_dash_items': len(reg['dashed']),
        'canvas_margin_min_pt': round(margin, 2),
        'title_a_end_to_letter_b_pt': round(letter_b[0] - title_a[2], 2),
        'zone_b_to_c_content_gap_pt': round(zc_left - zb_right, 2),
        'top_lines_pt': {k: round(outer_bbox(v, r)[3] if not isinstance(v, float) else v, 2)
                         for k, v in reg['top_lines'].items()},
        'credit_below_axis_title_pt': (round(outer_bbox(reg['xlabel'], r)[1] - outer_bbox(reg['credit'], r)[3], 2)
                                       if reg.get('credit') is not None else None),
        'main_arrow_visible_shaft_pt': round(next(a['visible_last_segment'] for a in reg['arrows']
                                                  if a['name'] == 'main entry'), 2),
    }
    out['measurements'] = extra
    out['all_pass'] = all(out[k]['pass'] for k in out if k[0].isdigit())
    for k in out:
        if k[0].isdigit() and not out[k]['pass']:
            print(f'CHECK FAILED: {k}', file=sys.stderr)
    return out


# ---------------------------------------------------------------- figure


# ---------------------------------------------------------------- merged layout (variant M3)
# Two panels instead of three: (a) the setup, each photograph shown once and as large as the
# 160 pt height allows, side by side over the model row, with the bus's switch from source a to b
# as a zoomed crop pair called out from the later photo's edit box; (b) the result bars.
# Selected with --layout merged (default); --layout abc draws the earlier three-panel figure.
LAYOUTS = ('merged', 'abc')
DEFAULT_LAYOUT = 'merged'
PHOTO_MIN = 80.0       # merged: minimum side of each photograph (pt); M9B: 80 (M9base 68), M9B and M9F draw 89.5 pt, M10 84 pt
TEXT_CLEAR = 1.0       # merged: min gap from visible glyphs to marks, frames, blocks and lines they do not sit in
BLOCK_WHITE = 12.0     # merged (M12; M11B 6): min white between the target block's lowest ink and the control block's highest
BOTTOM_WHITE = 4.0     # merged (M11B): min white from the control block's lowest ink to the photos' bottom frame line
ARROW_COLOURS = {      # merged (M12): M3 'arrow_colours' -> (bus arrow, boat arrow), shaft and head
    'coral_muted': (CORAL, MUTED),   # M12A (default): the edit in the coral of the edit marks; nothing changes at the boat
    'muted': (MUTED, MUTED),         # M12B: both in the muted grey of the secondary text
    'ink': (INK, INK),               # M11B's colouring, for comparison builds only
}
MID_CLEAR = 8.0        # merged (M9F): min gap from the middle column's ink (keys, glyphs) to both photo frames
GROUP_GAP = 12.0       # merged: min gap between the setup panel's content and the chart's content
M3 = {
    # M9B (from M9base): nothing under the band; each question sits in the middle column with the key
    # of the object it asks about; the band drops to the canvas bottom, its label moves under the row,
    # and the slot and the main photo move left so the photographs grow from 72 to 89.5 pt.
    'photo': 84.0,               # side of each photograph (pt); the crops are square (M9base 72, M9F 89.5; M10: 84 so the questions stay 7.5 pt)
    'main_left': 3.4,            # M9B: main photo's left image edge; its frame's outer stroke at 3.0 = left_x
    'top_outer': 132.6,          # photographs' outer stroke on the common top line
    'label_base': 136.0,         # baseline of the one-line entry labels above the photographs
    'label_sep': 4.0,            # entry name ink end to its sub-label's pen start
    'left_x': 3.0,               # heading a, the band and the main photo's frame start here
    'band_y0': 3.0,              # M9B: band bottom edge (M9base: row_y0 - band_pad = 27)
    'lm_base': 6.2,              # M9B: baseline of the one-line band label and the token label under the row
    'row_y0': 13.5, 'row_h': 18.0, 'band_pad': 4.0,   # band_pad: band top above the row (M9base also below)
    'tile_w': 35.0, 'node_gap': 15.0,   # M9B: tiles 35 wide (M9base 28) so the slot-to-Layer-1 arrow stays short
    'n_layers': 3,               # M7: tiles 'Layer 1'..'Layer n', each followed by its circled-plus node
    'ans_right': 293.5,          # the answer box's right edge (setup's right end; chart heading at chart_x0)
    'ans_w': 46.0,
    'bus_dy': 6.0,               # later bar this far above the row's top
    'credit_base': (11.1, 4.1),  # M9B: credit baselines, fixed (M9base derived them from the question baselines)
    'edit_lw': LW_EDIT,          # M11: the edit arrow's stroke (LW_EDIT 1.3 pt, head HEADS[LW_EDIT]; 1.1 = LW_EMPH for a trial)
    'arrow_colours': 'coral_muted',  # M12: colours of the two arrows between the photographs (ARROW_COLOURS):
                                 # coral_muted (M12A, default): bus arrow CORAL, boat arrow MUTED; muted (M12B): both MUTED
    'ctrl_dy': 0.0,              # M12: shift of the boat arrow (and its block) off the boat's centre line (pt, trials only; check 11 then fails)
    'q_size': 7.5,               # M10: the questions in the middle column, in ink, at the ordinary label size (M9F 6.5); 8 pt to each frame with 84 pt photos
    'mid_lead': (7.5, 8.5),      # M10: key row baseline to role line baseline, role line to question baseline (as M9B, for 7.5 pt questions; M9F 8.0, 7.5)
    'n_text_tokens': 5,          # M9F: grey text tokens in the slot (M9B 3); 'question' fits under them
    'lm_pad': 3.5,               # M9F: 'Language model' glyphs end this far before the band's right edge
    'defs_size': 6.5,            # M8: 'source a: red' / 'source b: blue', set as sub-lines of the 'edited object' key
    'defs_lead': 7.5,            # M8: leading of those sub-lines (from the key label's baseline)
    'key_gap': 2.5,              # transform arrow's stroke to the 'edited object' row's ink (key or glyphs; M9B 2.0)
    'edit_gap': 1.8,             # transform arrow's stroke to the bottom ink of its label 'bus recolored'
    'chart_x0': 293.5,           # M7b: panel b's heading x on the answer box's right edge (M7 296, M6G 306); axis end chart_x0 + 64.5 + chart_full
    'chart_full': 122.5,         # M7b: chart width of 100 % (M7 120, M6G 110), raised by what chart_x0 lost (axis ends at 480.5)
}


def glyph_box(t, r):
    """Visible-glyph box (pt): the text's window extent horizontally, its ink vertically."""
    wb = outer_bbox(t, r)
    if t.get_rotation() or t.get_va() != 'baseline':
        return wb
    g = ink_box(t.get_text(), t.get_fontsize(), t.get_fontweight())
    base = t.get_position()[1]
    return (wb[0], base + g[1], wb[2], base + g[3])


def line_segments(ax, reg):
    """Every drawn stroke as straight segments: Line2D pieces and arrow shafts (pts to the head base)."""
    segs = []
    for ln in ax.lines:
        if not ln.get_visible():
            continue
        xs, ys = (np.asarray(v, float) for v in ln.get_data())
        for i in range(len(xs) - 1):
            segs.append((f'line {id(ln)}', (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), ln.get_linewidth(), ln))
    for a in reg['arrows']:
        pts = [np.asarray(p, float) for p in a['pts']]
        hl = HEADS[a['lw']][0]
        pts = pts[:-1] + [pts[-1] - hl * a['dir']]
        for p0, p1 in zip(pts, pts[1:]):
            segs.append((f"shaft {a['name']}", tuple(p0), tuple(p1), a['lw'], a['shaft']))
    return segs


def stroke_gap(p0, p1, lw, b):
    """Gap (pt) from a butt-capped stroke of width lw along p0-p1 to the box b."""
    (x0, y0), (x1, y1) = p0, p1
    if abs(y1 - y0) < 1e-9:
        return sep((min(x0, x1), y0 - lw / 2, max(x0, x1), y0 + lw / 2), b)
    if abs(x1 - x0) < 1e-9:
        return sep((x0 - lw / 2, min(y0, y1), x0 + lw / 2, max(y0, y1)), b)
    return seg_box_gap(p0, p1, b, n=600) - lw / 2


def largest_empty_rect(boxes, segs, region, step=0.5):
    """Largest axis-aligned rectangle (pt) inside region that no box or stroke touches."""
    x0, y0, x1, y1 = region
    nx, ny = int(round((x1 - x0) / step)), int(round((y1 - y0) / step))
    occ = np.zeros((ny, nx), bool)
    def fill(b):
        i0 = max(0, int(np.floor((b[0] - x0) / step))); i1 = min(nx, int(np.ceil((b[2] - x0) / step)))
        j0 = max(0, int(np.floor((b[1] - y0) / step))); j1 = min(ny, int(np.ceil((b[3] - y0) / step)))
        if i1 > i0 and j1 > j0:
            occ[j0:j1, i0:i1] = True
    for b in boxes:
        fill(b)
    for p0, p1, lw in segs:
        n = max(2, int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / (step / 2)) + 1)
        for t in np.linspace(0, 1, n):
            x, y = p0[0] * (1 - t) + p1[0] * t, p0[1] * (1 - t) + p1[1] * t
            fill((x - lw / 2, y - lw / 2, x + lw / 2, y + lw / 2))
    best, where = 0.0, None
    hgt = np.zeros(nx, int)
    for j in range(ny):
        hgt = np.where(occ[j], 0, hgt + 1)
        stack = []
        for i in range(nx + 1):
            hh = hgt[i] if i < nx else 0
            start = i
            while stack and stack[-1][1] >= hh:
                s, sh = stack.pop()
                area = sh * (i - s)
                if area > best:
                    best, where = area, (s, j - sh + 1, i, j + 1)
                start = s
            stack.append((start, hh))
    if where is None:
        return None
    i0, j0, i1, j1 = where
    return {'xywh': [round(x0 + i0 * step, 1), round(y0 + j0 * step, 1),
                     round((i1 - i0) * step, 1), round((j1 - j0) * step, 1)],
            'area_pt2': round(best * step * step, 1)}


LABEL_OWN_RATIO = 0.5  # merged: a crop label's gap to its own crop at most this share of its gap to the other crop or the arrow


def check_no_crops(reg, texts, ax):
    """6 (merged, M5). No crops: the only rasters are the two photographs; no coral line anywhere (no
    leader), and the only coral-stroked patches are the dashed edit marks and the legend key (no crop
    frame); no text 'switch'. M9: no text says 'source' and none sets an italic a or b (the figure uses plain
    words: 'bus recolored', 'Object recolored only at'). The colours named in the figure ('red or blue?',
    'turned blue') are true for this item: in the pixels where d_a and d_b differ, d_a is red (R > B + 20)
    and d_b blue (B > R + 20).
    M12: the bus's edit arrow may be coral (arrow_colours coral_muted, M12A). Exactly the arrows whose
    expected colour (reg['edit_arrow_colours']) is coral may then carry coral: their shaft (stroke) and
    head (fill), and in M12A that is only the bus arrow. Every other coral stroke is still forbidden (no
    coral Line2D at all, no other coral-stroked patch than the dashed marks and the key), and no other
    patch is filled coral."""
    photo_ids = {id(p[1]) for p in reg['photos']}
    rasters_ok = len(ax.images) == 2 and {id(im) for im in ax.images} == photo_ids
    coral = to_hex(CORAL).lower()
    coral_lines = [ln for ln in ax.lines if ln.get_visible() and to_hex(ln.get_color()).lower() == coral]
    dashed_ids = {id(d) for d in reg['dashed']}
    coral_arrows = [a for a in reg['arrows']
                    if to_hex(reg.get('edit_arrow_colours', {}).get(a['name'], '#000000')).lower() == coral]
    shaft_ids = {id(a['shaft']) for a in coral_arrows}
    head_ids = {id(a['head']) for a in coral_arrows}
    coral_patches = [p for p in ax.patches if p.get_visible() and p.get_linewidth() > 0
                     and to_hex(p.get_edgecolor()).lower() == coral]
    patches_ok = ({id(p) for p in coral_patches} == dashed_ids | shaft_ids
                  and len(coral_patches) == len(dashed_ids) + len(shaft_ids))
    coral_fills = [p for p in ax.patches if p.get_visible() and p.get_fill()
                   and to_hex(p.get_facecolor()).lower() == coral and p.get_facecolor()[3] > 0]
    fills_ok = {id(p) for p in coral_fills} == head_ids and len(coral_fills) == len(head_ids)
    want = 1 if reg.get('arrow_colours') == 'coral_muted' else 0
    arrows_ok = (len(coral_arrows) == want
                 and all(a['name'] == 'edit (main to later)' for a in coral_arrows)
                 and all(to_hex(a['shaft'].get_edgecolor()).lower() == coral
                         and to_hex(a['head'].get_facecolor()).lower() == coral for a in coral_arrows))
    no_switch = not any('switch' in t.get_text().lower() for t in texts)
    # M9: no source letters in the merged figure: no text says 'source', none sets an italic a or b
    lettered = [t.get_text() for t in texts if t.get_visible() and ('source' in t.get_text().lower()
                or r'\mathit{a}' in t.get_text() or r'\mathit{b}' in t.get_text())]
    defs_ok = not lettered and not reg['defs']
    rows = [{'texts_with_source_or_italic_letter': lettered, 'pass': bool(defs_ok)}]
    col = reg['defs_colour']
    colour_ok = col['a_red'] and col['b_blue']
    ok = (rasters_ok and not coral_lines and patches_ok and fills_ok and arrows_ok and no_switch and defs_ok
          and colour_ok)
    return {'pass': bool(ok), 'rasters': len(ax.images), 'only_the_two_photos': bool(rasters_ok),
            'coral_lines': len(coral_lines),
            'coral_patches_are_only_marks_key_and_coral_edit_arrow_shaft': bool(patches_ok),
            'coral_patches': len(coral_patches), 'coral_fills_are_only_coral_edit_arrow_heads': bool(fills_ok),
            'coral_arrows': [a['name'] for a in coral_arrows], 'coral_arrows_expected': want,
            'coral_arrows_ok': bool(arrows_ok),
            'no_switch_text': bool(no_switch), 'definitions': rows,
            'colour_words_true': bool(colour_ok), 'colour': col}


def merged_checks(fig, ax, reg, zones, texts, r):
    out = {}
    # 7. Photographs: each shown once, at least PHOTO_MIN, native pixels of their files, undistorted,
    #    equal and top-aligned, framed exactly; the only other rasters are the two bus crops.
    rows, ok7 = [], True
    ext = []
    for name, im, fr, src_key, src_img in reg['photos']:
        arr = np.asarray(im.get_array())
        x0, x1, y0, y1 = im.get_extent()
        w_, h_ = x1 - x0, y1 - y0
        native = tuple(arr.shape[:2]) == tuple(src_img.shape[:2]) and im.get_interpolation() == 'none'
        same = bool(np.array_equal(arr, src_img))
        undist = abs(w_ / arr.shape[1] - h_ / arr.shape[0]) < 1e-9
        fb = fr.get_path().get_extents(fr.get_patch_transform())
        framed = np.allclose([fb.x0, fb.y0, fb.x1, fb.y1], [x0, y0, x1, y1])
        good = native and same and undist and framed and min(w_, h_) >= PHOTO_MIN - 1e-9
        ok7 &= good
        ext.append((x0, y0, x1, y1))
        rows.append({'photo': name, 'source': src_key, 'size_pt': [round(w_, 2), round(h_, 2)],
                     'pixels': list(arr.shape[:2]), 'pt_per_px': round(w_ / arr.shape[1], 4),
                     'same_pixels_as_source': same, 'native_pixels': bool(native), 'undistorted': bool(undist),
                     'frame_on_image_edge': bool(framed), 'pass': bool(good)})
    full = [im for im in ax.images if tuple(np.asarray(im.get_array()).shape[:2]) == tuple(reg['photos'][0][4].shape[:2])]
    distinct = not np.array_equal(np.asarray(reg['photos'][0][1].get_array()), np.asarray(reg['photos'][1][1].get_array()))
    once = len(full) == 2 and distinct and len(ax.images) == 2   # M5: no crop rasters
    equal = abs((ext[0][2] - ext[0][0]) - (ext[1][2] - ext[1][0])) < 1e-9 and abs((ext[0][3] - ext[0][1]) - (ext[1][3] - ext[1][1])) < 1e-9
    top_al = abs(ext[0][3] - ext[1][3]) < 1e-9
    ok7 &= once and equal and top_al
    out['7_photos'] = {'pass': bool(ok7), 'min_side_pt': PHOTO_MIN, 'each_photo_once': bool(once),
                       'rasters': len(ax.images), 'equal_size': bool(equal), 'top_aligned': bool(top_al), 'photos': rows}

    # 8. Routing: every arrow orthogonal; main enters on the main photo's centre line and the middle
    #    image token; the later trunk leaves the later photo's centre and the bus is symmetric about it.
    rows, ok8 = [], True
    for a in reg['arrows']:
        pts = [np.asarray(p, float) for p in a['pts']]
        orth = all(abs(p1[0] - p0[0]) < 1e-9 or abs(p1[1] - p0[1]) < 1e-9 for p0, p1 in zip(pts, pts[1:]))
        ok8 &= orth
        rows.append({'arrow': a['name'], 'orthogonal': bool(orth)})
    g = reg['route']
    main = next(a for a in reg['arrows'] if a['name'] == 'main entry')
    mp = [np.asarray(p, float) for p in main['pts']]
    (mx0, my0, mx1, my1), (lx0, ly0, lx1, ly1) = ext
    main_ok = (abs(mp[0][0] - (mx0 + mx1) / 2) < 1e-9 and abs(mp[0][0] - g['main_token_c']) < 1e-9
               and abs(mp[0][1] - my0) < 1e-9 and to_hex(main['head'].get_facecolor()) == MAIN.lower())
    # M6B: the bar is no longer symmetric about the trunk (the answer box must end inside the panel).
    # M6F: the trunk leaves the later photo's bottom at its centre and comes straight down into node 2
    # (node 2's centre on the trunk, its outer stroke under the later photo); the branch to node 1 goes
    # left, and node 1 (outer stroke) lies under the gap between the photos, left of the later frame.
    # M7: n_layers nodes (3). The last node is the trunk's: centred on it, its outer stroke under the
    # later photo, the drop straight down the trunk. Every other node lies under the gap between the
    # photos, left of the trunk; node 1 is reached by the branch's rounded elbow (three points, the
    # middle one on node 1's x at bus_y), every node in between by a straight drop that starts on the
    # branch's centre line (a T). Exactly one drop per node, each with its own LATER head.
    tx, ty = (np.asarray(v, float) for v in reg['trunk'].get_data())
    nn = reg['route']['n_layers']
    bus_y = reg['route']['bus_y']
    nodes_x = [n.center[0] for n in reg['nodes']]
    n_out = [n.radius + n.get_linewidth() / 2 for n in reg['nodes']]
    fl_ = outer_bbox(reg['photos'][1][2])
    fm_ = outer_bbox(reg['photos'][0][2])
    last_under = lx0 <= nodes_x[-1] - n_out[-1] and nodes_x[-1] + n_out[-1] <= lx1
    rest_gap = all(fm_[2] <= x_ - o_ and x_ + o_ <= fl_[0] for x_, o_ in zip(nodes_x[:-1], n_out[:-1]))
    by_name = {a['name']: a for a in reg['arrows'] if a['name'].startswith('later drop')}
    dl = by_name.get(f'later drop (node {nn})')
    dl_straight = dl is not None and len(dl['pts']) == 2 and np.allclose([p_[0] for p_ in dl['pts']], tx[0])
    d1 = by_name.get('later drop (node 1)')
    d1_elbow = (d1 is not None and len(d1['pts']) == 3 and abs(d1['pts'][1][0] - nodes_x[0]) < 1e-9
                and abs(d1['pts'][0][1] - bus_y) < 1e-9 and abs(d1['pts'][1][1] - bus_y) < 1e-9)
    tees = []
    for k in range(1, nn - 1):
        a_ = by_name.get(f'later drop (node {k + 1})')
        tees.append(a_ is not None and len(a_['pts']) == 2 and abs(a_['pts'][0][0] - nodes_x[k]) < 1e-9
                    and abs(a_['pts'][1][0] - nodes_x[k]) < 1e-9 and abs(a_['pts'][0][1] - bus_y) < 1e-9
                    and d1 is not None and d1['pts'][1][0] < nodes_x[k] < d1['pts'][0][0])
    increasing = all(b_ > a_ for a_, b_ in zip(nodes_x, nodes_x[1:]))
    trunk_ok = (np.allclose(tx, (lx0 + lx1) / 2) and abs(ty.max() - ly0) < 1e-9 and len(nodes_x) == nn
                and abs(nodes_x[-1] - tx[0]) < 1e-9 and dl_straight and d1_elbow and all(tees) and increasing
                and nodes_x[0] < tx[0] and last_under and rest_gap)
    drops = [a for a in reg['arrows'] if a['name'].startswith('later drop')]
    drops_ok = (len(drops) == nn and sorted(id(a['target']) for a in drops) == sorted(id(n) for n in reg['nodes'])
                and all(to_hex(a['head'].get_facecolor()) == LATER.lower() for a in drops))
    # the branch's stroke clears the photographs' bottom edge (outer stroke) and the band's top
    band_top = reg['band'].get_y() + reg['band'].get_height()
    branch_clear = {'to_photo_bottom_pt': round(min(fm_[1], fl_[1]) - (bus_y + LW_EMPH / 2), 3),
                    'to_band_top_pt': round((bus_y - LW_EMPH / 2) - band_top, 3)}
    trunk_ok &= min(branch_clear.values()) > 0
    # M6B: the transformation arrow starts on the main photo's frame (outer stroke), ends on the later
    # photo's frame, and runs on the bus's line: inside both edit boxes' vertical span.
    t = reg['transform']
    tp = [np.asarray(p, float) for p in t['pts']]
    eb = {n.split(', ')[1]: outer_bbox(b) for n, b, _ in reg['marks'] if n.startswith('edited')}
    f_main = outer_bbox(reg['photos'][0][2])
    # M6F: its label 'bus from b' sits centred over the shaft (within 0.05 pt), above it, and wholly
    # inside the gap between the two frames.
    elb = glyph_box(reg['edit_label'], r)
    eli = ink_box(reg['edit_label'].get_text(), reg['edit_label'].get_fontsize())
    el_c = reg['edit_label'].get_position()[0] + (eli[0] + eli[2]) / 2     # centre of the visible glyphs
    f_later = outer_bbox(reg['photos'][1][2])
    # M11: the label's visible glyphs clear the shaft's stroke (top edge) by at least TEXT_CLEAR (was: label
    # bottom above the shaft's centre line), and the arrow is of the edit class: weight M3['edit_lw'], its
    # head HEADS[edit_lw]. M12: the label row is the label with its key symbol to its left, centred as one
    # unit (key's outer left edge to the label's last glyph) over the shaft, inside the gap; its lowest ink
    # (glyphs or key) clears the stroke by at least TEXT_CLEAR; shaft and head are in the arrow's expected
    # colour (reg['edit_arrow_colours'], from M3 'arrow_colours'; M11: INK) and the label stays in INK.
    def edit_arrow_ok(a_, lab_, key_, spans_):
        p_ = [np.asarray(q_, float) for q_ in a_['pts']]
        lb_ = glyph_box(lab_, r)
        kb_ = outer_bbox(key_)
        l_end_ = lab_.get_position()[0] + ink_box(lab_.get_text(), lab_.get_fontsize())[2]   # last glyph's ink
        row_c_ = (kb_[0] + l_end_) / 2
        gap_ = min(lb_[1], kb_[1]) - (p_[0][1] + a_['lw'] / 2)
        lab_ok_ = (abs(row_c_ - (f_main[2] + f_later[0]) / 2) < 0.05 and gap_ >= TEXT_CLEAR - 1e-6
                   and f_main[2] < kb_[0] and kb_[2] < lb_[0] and lb_[2] < f_later[0]
                   and to_hex(lab_.get_color()).lower() == INK.lower())
        want_ = to_hex(reg['edit_arrow_colours'][a_['name']]).lower()
        style_ok_ = (abs(a_['lw'] - M3['edit_lw']) < 1e-9 and to_hex(a_['head'].get_facecolor()) == want_
                     and to_hex(a_['shaft'].get_edgecolor()) == want_)
        geo_ok_ = (len(p_) == 2 and abs(p_[0][1] - p_[1][1]) < 1e-9 and abs(p_[0][0] - f_main[2]) < 1e-9
                   and a_['target'] is reg['photos'][1][2] and p_[1][0] > p_[0][0]
                   and all(b[1] < p_[0][1] < b[3] for b in spans_))
        return bool(lab_ok_ and style_ok_ and geo_ok_), {
            'y': round(float(p_[0][1]), 3), 'x': [round(float(p_[0][0]), 3), round(float(p_[1][0]), 3)],
            'stroke_pt': a_['lw'], 'head_pt': list(HEADS[a_['lw']]), 'colour': to_hex(a_['head'].get_facecolor()),
            'expected_colour': want_,
            'label': lab_.get_text(), 'label_row_ink_gap_to_stroke_pt': round(float(gap_), 3),
            'label_row_centre_dev_pt': round(float(row_c_ - (f_main[2] + f_later[0]) / 2), 3),
            'inside_object_spans': [[round(b[1], 2), round(b[3], 2)] for b in spans_],
            'label_ok': bool(lab_ok_), 'edit_class_colour': bool(style_ok_), 'frame_to_frame_in_span': bool(geo_ok_)}
    transform_ok, t_rep = edit_arrow_ok(t, reg['edit_label'], reg['mid']['target']['key'], list(eb.values()))
    # M11B: the control arrow passes the same test on the boat's line (inside both control boxes' vertical
    # span) with its label 'boat unchanged'. M12: it is always drawn, so its absence fails the check.
    ca = reg.get('control_arrow')
    if ca is not None:
        cbs = [outer_bbox(b) for n, b, _ in reg['marks'] if n.startswith('control')]
        control_ok, c_rep = edit_arrow_ok(ca['arrow'], ca['label'], reg['mid']['control']['key'], cbs)
    else:
        control_ok, c_rep = False, None
    ok8 &= main_ok and trunk_ok and drops_ok and transform_ok and control_ok
    out['8_routing'] = {'pass': bool(ok8), 'arrows': rows, 'main_on_photo_and_token_centre': bool(main_ok),
                        'trunk_on_photo_centre_into_last_node_others_under_gap_tees_elbow': bool(trunk_ok),
                        'n_layers': nn, 'nodes_x': [round(v, 2) for v in nodes_x], 'branch_clearance': branch_clear,
                        'later_drops_into_nodes': bool(drops_ok),
                        'transform_frame_to_frame_on_bus_line_labelled': bool(transform_ok), 'transform': t_rep,
                        'control_arrow_frame_to_frame_on_boat_line_labelled': None if ca is None else bool(control_ok),
                        'control_arrow': c_rep}

    # 9. Clearance: visible glyphs keep TEXT_CLEAR from every mark, frame, block, node, bar and stroke
    #    they are not designed to sit in; the setup and the chart keep GROUP_GAP apart.
    obst = [(n, outer_bbox(a)) for n, a in reg['obstacles']]
    obst += [(f"head {a['name']}", outer_bbox(a['head'])) for a in reg['arrows']]
    segs = line_segments(ax, reg)
    rows, ok9 = [], True
    for t in texts:
        gb = glyph_box(t, r)
        inside = reg['text_inside'].get(id(t), set())
        cands = [(sep(gb, b), n) for n, b in obst if n not in inside]
        cands += [(stroke_gap(p0, p1, lw, gb), n) for n, p0, p1, lw, _ in segs]
        gap, who = min(cands, key=lambda v: v[0])
        good = gap >= TEXT_CLEAR - 1e-6
        ok9 &= good
        if not good or gap < 2.0:
            rows.append({'text': t.get_text(), 'nearest': who, 'gap_pt': round(gap, 3), 'pass': bool(good)})
    a_right = max(outer_bbox(a, r)[2] for a in zones['a'])
    b_left = min(outer_bbox(a, r)[0] for a in zones['b'])
    # M7: the setup (answer box at the bottom) and the chart (heading at the top) now overlap in x, so the
    # 1-D gap b_left - a_right no longer describes the layout. Two-dimensional rule: the smallest Euclidean
    # distance between the box of any setup ink (texts by their visible glyphs, photos, frames, boxes,
    # lines, heads, the band) and the box of any chart ink (heading, header, names, bars, axis, ticks,
    # tick labels, axis label; not the credit) is at least GROUP_GAP.
    def ink_bb(a):
        return glyph_box(a, r) if isinstance(a, matplotlib.text.Text) else outer_bbox(a, r)
    def bdist(b1, b2):
        dx = max(0.0, b2[0] - b1[2], b1[0] - b2[2])
        dy = max(0.0, b2[1] - b1[3], b1[1] - b2[3])
        return float(np.hypot(dx, dy)) if (dx > 0 or dy > 0) else float(sep(b1, b2))
    def label(a):
        if isinstance(a, matplotlib.text.Text):
            return f'text {a.get_text()!r}'
        return f'{type(a).__name__} {[round(v, 1) for v in outer_bbox(a, r)]}'
    sa = [(a, ink_bb(a)) for a in zones['a']]
    sb = [(a, ink_bb(a)) for a in reg['chart_ink']]
    pairs = sorted(((bdist(b1, b2), a1, a2) for a1, b1 in sa for a2, b2 in sb), key=lambda v: v[0])
    gg, ga, gb_ = pairs[0]
    ok9 &= gg >= GROUP_GAP - 1e-6
    closest = [{'setup': label(a1), 'chart': label(a2), 'distance_pt': round(d_, 2)} for d_, a1, a2 in pairs[:4]]
    # white space between the later photo's right edge and the nearest chart ink at photo height
    flb = outer_bbox(reg['photos'][1][2])
    at_h = [b for _, b in sb if b[1] < flb[3] and b[3] > flb[1] and b[0] > flb[2]]
    photo_ws = min(b[0] for b in at_h) - flb[2] if at_h else None
    out['9_clearance'] = {'pass': bool(ok9), 'min_text_clear_pt': TEXT_CLEAR, 'tight_or_failing_texts': rows,
                          'setup_to_chart_min_distance_2d_pt': round(gg, 2), 'closest_setup_chart_pairs': closest,
                          'min_group_gap_pt': GROUP_GAP,
                          'later_photo_to_chart_at_photo_height_pt': round(photo_ws, 2) if photo_ws is not None else None,
                          'x_overlap_setup_right_minus_chart_left_pt': round(a_right - b_left, 2)}

    # 10. Required wording present (labels, questions, legend, chart, credit).
    have = {t.get_text() for t in texts}
    need = reg['required_text']
    missing = [s for s in need if s not in have]
    credit_ok = ' '.join(reg['credit_lines']) == reg['credit_sentence']
    lm_ok = ' '.join(t.get_text() for t in reg['lm_texts']) == 'Language model'   # M6B: band label in two lines
    out['10_required_text'] = {'pass': not missing and credit_ok and lm_ok, 'required': len(need), 'missing': missing,
                               'band_label_reads_Language_model': bool(lm_ok),
                               'credit_word_for_word': bool(credit_ok), 'credit_lines': reg['credit_lines']}

    # 11. M9B, adapted in M9F and M12: each question beside its object, nothing under the band. M12: each
    #     block in the middle column has four elements: the label row (key symbol, 3 pt, label), centred as
    #     one unit on the gap's centre (key's outer left edge to the label's last glyph, 0.05 pt); the arrow,
    #     on its object's centre line (bus: the edit box's centre; boat: the control box's centre; 0.05 pt;
    #     M11B: the boat arrow was not on the boat's line); the role line and the question, each centred
    #     (visible glyphs, 0.05 pt). The label row's lowest ink sits edit_gap above the arrow's stroke and the
    #     role line's top ink key_gap under it (0.05 pt), the question lead2 under the role line; the key
    #     lies inside its object's vertical span; all the block's ink keeps MID_CLEAR (8 pt) from both
    #     frames. The target block lies wholly above the control block with at least BLOCK_WHITE (12 pt;
    #     M11B 6) of white between them, and the control block's lowest ink lies at least BOTTOM_WHITE (4 pt)
    #     above the photos' bottom frame line (the stroke's top edge). The token label 'question' is centred
    #     (visible glyphs, 0.05 pt) on the grey text tokens and its glyphs lie within their run, clear of the
    #     image tokens, below the slot and inside the band. M9F: the band label 'Language model' sits in the
    #     band's lower right corner (glyphs end lm_pad before the band's right edge, below the row, inside
    #     the band). No setup ink lies below the band's bottom edge.
    m = reg['mid']
    cx_, col_w_, gx0_, gx1_ = m['column']
    gc_ = m['gap_centre']
    rows11, ok11 = [], True
    spans = {}
    for nm in ('target', 'control'):
        blk = m.get(nm)
        if blk is None or blk.get('arrow') is None:
            ok11 = False
            rows11.append({'block': nm, 'missing': True, 'pass': False})
            continue
        kb = outer_bbox(blk['key'])
        lab = blk['label']
        lgb = glyph_box(lab, r)
        gbs = [glyph_box(t_, r) for t_ in blk['texts']]
        l_end = lab.get_position()[0] + ink_box(lab.get_text(), lab.get_fontsize())[2]   # last glyph's ink
        devs = [(kb[0] + l_end) / 2 - gc_]                         # label row as one unit
        for t_ in blk['texts']:
            gi_ = ink_box(t_.get_text(), t_.get_fontsize())
            devs.append(t_.get_position()[0] + (gi_[0] + gi_[2]) / 2 - gc_)
        centred = all(abs(d_) <= 0.05 for d_ in devs)
        allb = [kb, lgb] + gbs
        side = (min(b_[0] for b_ in allb) - gx0_, gx1_ - max(b_[2] for b_ in allb))
        inside = min(side) >= MID_CLEAR - 1e-6
        top_ = max(b_[3] for b_ in allb)
        bot_ = min(b_[1] for b_ in allb)
        spans[nm] = (bot_, top_)
        a_ = blk['arrow']
        ay = float(a_['pts'][0][1])
        centre_ = m['bus_c'] if nm == 'target' else m['boat_c']
        span_ = m['bus_span'] if nm == 'target' else m['boat_span']
        on_line = abs(ay - centre_) <= 0.05
        above = min(kb[1], lgb[1]) - (ay + a_['lw'] / 2)            # label row's lowest ink above the stroke
        under = (ay - a_['lw'] / 2) - gbs[0][3]                     # stroke to the role line's top ink
        lead_ = blk['texts'][0].get_position()[1] - blk['texts'][1].get_position()[1]
        gaps_ok = (abs(above - M3['edit_gap']) <= 0.05 and abs(under - M3['key_gap']) <= 0.05
                   and abs(lead_ - M3['mid_lead'][1]) <= 1e-6)
        order_in = lgb[1] > ay > gbs[0][3] and gbs[0][1] > gbs[1][3]    # label, arrow, role, question
        key_in = span_[0] <= kb[1] and kb[3] <= span_[1]
        good = centred and inside and on_line and gaps_ok and order_in and key_in
        ok11 &= good
        rows11.append({'block': nm, 'label_row': [('dashed key' if nm == 'target' else 'white-box key'), lab.get_text()],
                       'lines': [t_.get_text() for t_ in blk['texts']], 'centred_on_gap': bool(centred),
                       'centre_dev_pt': [round(d_, 3) for d_ in devs],
                       'side_clearance_pt': [round(v_, 2) for v_ in side], 'inside_gap_mid_clear': bool(inside),
                       'arrow_y': round(ay, 3), 'object_centre_y': round(centre_, 3),
                       'arrow_on_object_centre_line': bool(on_line),
                       'label_row_ink_above_stroke_pt': round(float(above), 3),
                       'role_top_ink_under_stroke_pt': round(float(under), 3), 'role_to_question_lead_pt': round(lead_, 3),
                       'gaps_ok': bool(gaps_ok), 'order_label_arrow_role_question': bool(order_in),
                       'key_inside_object_span': bool(key_in), 'ink_y': [round(bot_, 2), round(top_, 2)],
                       'pass': bool(good)})
    bot_line_ = reg['photos'][0][1].get_extent()[2] + LW_REG / 2
    if len(spans) == 2:
        white_between = spans['target'][0] - spans['control'][1]
        white_below = spans['control'][0] - bot_line_
        order_ok = (spans['target'][0] > spans['control'][1] and white_between >= BLOCK_WHITE - 1e-6
                    and white_below >= BOTTOM_WHITE - 1e-6)
    else:
        white_between = white_below = float('nan')
        order_ok = False
    qt = reg['q_token_label']
    qg = glyph_box(qt, r)
    tq0, tq1 = reg['text_tokens_x']
    iq0, _ = reg['image_tokens_x']
    slot_b = outer_bbox(dict(reg['obstacles'])['slot'])
    band_b = outer_bbox(reg['band'])
    qi_ = ink_box(qt.get_text(), qt.get_fontsize())
    q_x0, q_x1 = qt.get_position()[0] + qi_[0], qt.get_position()[0] + qi_[2]   # visible glyphs
    q_c = (q_x0 + q_x1) / 2
    q_within = tq0 - 1e-6 <= q_x0 and q_x1 <= tq1 + 1e-6 and q_x1 < iq0
    q_ok = (abs(q_c - (tq0 + tq1) / 2) <= 0.05 and q_within and qg[3] < slot_b[1] - TEXT_CLEAR
            and qg[1] > band_b[1] and qg[0] > band_b[0])
    lm_t = reg['lm_texts'][0]
    lmg = glyph_box(lm_t, r)
    lmi = ink_box(lm_t.get_text(), lm_t.get_fontsize())
    lm_end = lm_t.get_position()[0] + lmi[2]
    row_bottom = min(outer_bbox(b_)[1] for _, b_ in reg['row_obstacles'])
    lm_corner = (abs(lm_end - (reg['band_x1'] - M3['lm_pad'])) <= 0.05 and lmg[3] < row_bottom - TEXT_CLEAR
                 and lmg[1] > band_b[1] and lm_end < band_b[2])
    lowest = min(outer_bbox(a, r)[1] if not isinstance(a, matplotlib.text.Text) else glyph_box(a, r)[1]
                 for a in zones['a'])
    nothing_below = lowest >= band_b[1] - 1e-6
    ok11 &= order_ok and q_ok and lm_corner and nothing_below
    out['11_questions_beside_objects'] = {'pass': bool(ok11), 'blocks': rows11, 'target_above_control': bool(order_ok),
                                          'white_target_to_control_pt': round(float(white_between), 3),
                                          'white_control_to_photo_bottom_line_pt': round(float(white_below), 3),
                                          'min_white_between_blocks_pt': BLOCK_WHITE,
                                          'min_white_below_control_pt': BOTTOM_WHITE,
                                          'min_side_clearance_pt': MID_CLEAR,
                                          'question_label_on_text_tokens_in_band': bool(q_ok),
                                          'question_label_glyphs_x': [round(q_x0, 2), round(q_x1, 2)],
                                          'text_tokens_x': [round(tq0, 2), round(tq1, 2)], 'image_tokens_x0': round(iq0, 2),
                                          'band_label_in_lower_right_corner': bool(lm_corner),
                                          'band_label_glyphs': [round(v_, 2) for v_ in lmg],
                                          'lowest_setup_ink_y': round(lowest, 2), 'band_bottom_y': round(band_b[1], 2),
                                          'nothing_below_band': bool(nothing_below)}

    # Measurements (reported).
    dash = sorted({tuple(round(float(v), 3) for v in p._dash_pattern[1]) for p in reg['dashed']})
    allb = [outer_bbox(a, r) for z in zones.values() for a in z]
    margin = min(min(b[0], b[1], W - b[2], H - b[3]) for b in allb)
    region = (reg['panel_a_box'][0], reg['panel_a_box'][1], a_right, reg['panel_a_box'][3])
    boxes = [glyph_box(t, r) for t in texts if t in zones['a']]
    boxes += [outer_bbox(a) for a in zones['a'] if isinstance(a, (Rectangle, Circle, Polygon, PathPatch,
                                                                  FancyBboxPatch, matplotlib.image.AxesImage))
              and a is not reg['band']]
    lsegs = [(p0, p1, lw) for n, p0, p1, lw, art in segs if art in zones['a']]
    out['measurements'] = {
        'coral_dash_rendered_on_off_pt': dash, 'coral_dash_items': len(reg['dashed']),
        'canvas_margin_min_pt': round(margin, 2),
        'photo_side_pt': round(ext[0][2] - ext[0][0], 2),
        'top_lines_pt': {k: round(outer_bbox(v, r)[3] if not isinstance(v, float) else v, 2)
                         for k, v in reg['top_lines'].items()},
        'credit_below_axis_title_pt': (round(outer_bbox(reg['xlabel'], r)[1] - outer_bbox(reg['credit'], r)[3], 2)
                                       if reg['credit'] in zones['b'] else None),
        'main_arrow_visible_shaft_pt': round(main['visible_last_segment'], 2),
        'setup_right_pt': round(a_right, 2), 'chart_left_pt': round(b_left, 2),
        'largest_empty_rect_in_setup': largest_empty_rect(boxes, lsegs, region),
        'largest_empty_rect_above_band': largest_empty_rect(boxes, lsegs, (region[0], reg['band'].get_y() + reg['band'].get_height(), region[2], region[3])),
        # M9B: the same, left of the later photo's right edge (the strip right of it is the white space to the chart)
        'largest_empty_rect_left_of_later_photo_right': largest_empty_rect(boxes, lsegs, (region[0], region[1], outer_bbox(reg['photos'][1][2])[2], region[3])),
    }
    return out


def build_merged(variant, outdir, dpi):
    spec = VARIANTS[variant]
    if not (spec.get('zoom') and spec.get('main_source')):
        raise SystemExit('the merged layout is defined for the photo variant only')
    L = M3
    img_main, img_a, img_b, box, cbox, info = load_item(spec)
    fig = plt.figure(figsize=(W / 72, H / 72), dpi=600)  # dpi only sets measurement precision
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    aspect = img_a.shape[0] / img_a.shape[1]
    reg = {'arrows': [], 'obstacles': [], 'row_obstacles': [], 'nodes': [], 'chain': [],
           'marks': [], 'dashed': [], 'top_lines': {}, 'tile_words': [], 'tile_labels': [],
           'model_layout': 'row2', 'layout': 'merged', 'photos': [], 'zooms': [], 'text_inside': {}}
    zones, seen = {}, set(ax.get_children())

    def close_zone(z):
        nonlocal seen
        now = ax.get_children()
        zones[z] = [a for a in now if a not in seen and a.get_visible()]
        seen = set(now)

    head_base, top = 149.3, 138.0          # top: panel b's chart header line (unchanged)
    period = sum(CORAL_DASH)
    kw_, kh_, ch = 2 * period, period, 5.6
    key_text_dx = kw_ + LW_EMPH + 3.0       # key outer left edge to legend text
    geo = {'layout': 'merged (M6F)', 'params': dict(L)}
    lx = L['left_x']
    P = L['photo']
    ph = P * aspect

    # ---- x positions. Left (M9B): the main photo starts at main_left (frame on left_x); its centre sets the
    #      slot's middle image token, and the slot follows (no band label at the left any more).
    #      Right: the answer box ends at ans_right; the row is laid out backwards from it to node 1.
    main_x = L['main_left']
    main_c = main_x + P / 2
    main_tok_c = main_c
    tok, pitch, split = 4.6, 5.4, 1.6
    n_txt = int(L['n_text_tokens'])   # M9F: 5 grey text tokens (M9B 3), so the label 'question' fits under them
    in_x0 = main_tok_c - (tok / 2 + pitch) - split - n_txt * pitch - 2.5   # M9B (3 tokens): main_tok_c - 28
    in_x1 = main_tok_c + 10.0
    nr, bare, clear, dot_r, dot_p = 3.5, 3.2, 1.5, .65, 2.3
    bw, gn = L['tile_w'], L['node_gap']
    ans_x1 = L['ans_right']
    ans_x0 = ans_x1 - L['ans_w']
    resume = ans_x0 - 7.5
    d0 = resume - clear - dot_r - 2 * dot_p
    line_end = d0 - dot_r - clear
    n_lay = int(L['n_layers'])
    nlast = line_end - bare - nr - LW_REG / 2
    nodes_xs = [nlast - (n_lay - 1 - k) * (bw + gn) for k in range(n_lay)]   # M7: node 1 .. node n
    n1x = nodes_xs[0]
    later_c = nlast                  # M6F/M7: the later trunk comes straight down into the last node
    later_x = later_c - P / 2
    gap = later_x - (main_x + P)
    py1 = L['top_outer'] - LW_REG / 2
    py0 = py1 - ph
    heading(ax, lx, head_base, 'a', spec['title_merged'])
    reg['title_a'] = ax.texts[-1]

    # Photographs, each once: main gets the original (m0_s100); later gets d_b, which equals the
    # original with the bus's box from b (asserted in load_item).
    fm = photo(ax, img_main, main_x, py0, P, ph, MAIN, LW_REG)
    im_main = ax.images[-1]
    fl = photo(ax, img_b, later_x, py0, P, ph, LATER, LW_REG)
    im_later = ax.images[-1]
    reg['photos'] = [('main', im_main, fm, spec['main_source'], img_main), ('later', im_later, fl, 'd_b', img_b)]
    reg['obstacles'] += [('frame main', fm), ('frame later', fl)]
    reg['top_lines']['photographs (outer stroke)'] = py1 + LW_REG / 2
    ebox, cboxes = {}, {}
    for nm, xx, fr in (('later', later_x, fl), ('main', main_x, fm)):
        eb = mark(ax, box, img_b.shape, xx, py0, P, ph, CORAL, LW_EMPH, 2.0, True)
        ebox[nm] = eb
        reg['marks'].append((f'edited, {nm}', eb, fr))
        reg['dashed'].append(eb)
        reg['obstacles'].append((f'edited box {nm}', eb))
        cb = mark(ax, cbox, img_b.shape, xx, py0, P, ph, 'white', LW_CTRL, 1.2, False)
        cboxes[nm] = cb
        reg['marks'].append((f'control, {nm}', cb, fr))
        reg['obstacles'].append((f'control box {nm}', cb))

    # ---- entry labels: one line above each photograph, left-aligned on its frame: the entry's name
    #      (bold, entry colour), then what that entry receives
    lb = L['label_base']
    t_main = text(ax, main_x, lb, 'Main entry', weight='bold', color=MAIN_INK)
    text(ax, main_x + ink_box('Main entry', 7.5, 'bold')[2] + L['label_sep'], lb, spec['main_label'])
    text(ax, later_x, lb, 'Later entries', weight='bold', color=LATER_INK)
    # M6G: 'edited photo' mirrors 'original photo'; the edit itself ('bus from b') is on the arrow
    text(ax, later_x + ink_box('Later entries', 7.5, 'bold')[2] + L['label_sep'], lb, spec['later_sub'])
    reg['top_lines']['entry labels (cap line)'] = lb + cap(7.5)

    # ---- model row: input slot under the main photo; both nodes under the later photo
    row_y0, row_h = L['row_y0'], L['row_h']
    row_c = row_y0 + row_h / 2
    band_y0, band_y1 = L['band_y0'], row_y0 + row_h + L['band_pad']   # M9B: band reaches down to band_y0
    slot = FancyBboxPatch((in_x0, row_y0), in_x1 - in_x0, row_h, boxstyle='round,pad=0,rounding_size=2.5',
                          facecolor='white', edgecolor=EDGE, lw=LW_REG, zorder=3)
    ax.add_patch(slot)
    img_l = [main_tok_c - tok / 2 + (i - 1) * pitch for i in range(3)]
    txt_l = [img_l[0] - split - (n_txt - i) * pitch for i in range(n_txt)]
    for xl in txt_l:
        ax.add_patch(Rectangle((xl, row_c - tok / 2), tok, tok, facecolor=TOKEN, edgecolor='none', lw=0, zorder=4))
    for xl in img_l:
        ax.add_patch(Rectangle((xl, row_c - tok / 2), tok, tok, facecolor=IMG_TOKEN, edgecolor=MAIN,
                               lw=LW_HAIR, zorder=4))
    q_tok_c = txt_l[1] + tok / 2
    blocks, nodes = [], []
    for i, nx in enumerate(nodes_xs):
        bx = nx - gn / 2 - bw
        blk = FancyBboxPatch((bx, row_y0), bw, row_h, boxstyle='round,pad=0,rounding_size=2.0',
                             facecolor=LAYER_FILL, edgecolor=EDGE, lw=LW_REG, zorder=3)
        ax.add_patch(blk)
        blocks.append(blk)
        node = Circle((nx, row_c), nr, facecolor='white', edgecolor=LATER, lw=LW_REG, zorder=4)
        ax.add_patch(node)
        arm = 2.0
        line(ax, [nx - arm, nx + arm], [row_c, row_c], LATER, LW_REG, zorder=5)
        line(ax, [nx, nx], [row_c - arm, row_c + arm], LATER, LW_REG, zorder=5)
        reg['chain'].append(line(ax, [bx + bw, nx], [row_c, row_c]))
        nodes.append(node)
        tile_text = f'Layer {i + 1}'
        gx0, _, gx1, _ = ink_box(tile_text, 6.5)
        lab = text(ax, bx + bw / 2 - (gx0 + gx1) / 2, row_c - cap(6.5) / 2, tile_text, size=6.5, zorder=4)
        reg['tile_labels'].append((lab, blk))
        reg['text_inside'][id(lab)] = {f'block {i + 1}'}
    b0 = blocks[0].get_x()
    if b0 - in_x1 < 8.0:
        raise SystemExit(f'slot to layer 1 only {b0 - in_x1:.1f} pt')
    for blk, node in zip(blocks[1:], nodes[:-1]):
        reg['chain'].append(line(ax, [node.center[0], blk.get_x()], [row_c, row_c]))
    arrow(ax, reg, 'slot to block 1', [(in_x1, row_c), (b0 - LW_REG / 2 - TIP_GAP, row_c)], EDGE, LW_REG, blocks[0],
          start_inset=LW_REG / 2)
    reg['chain'].append(line(ax, [nodes[-1].center[0], line_end], [row_c, row_c]))
    dots = []
    for i in range(3):
        dot = Circle((d0 + i * dot_p, row_c), dot_r, facecolor=EDGE, edgecolor='none', lw=0, zorder=4)
        ax.add_patch(dot)
        dots.append(dot)
    band_x1 = d0 + 2 * dot_p + dot_r + 3.5
    ans = FancyBboxPatch((ans_x0, row_y0), ans_x1 - ans_x0, row_h, boxstyle='round,pad=0,rounding_size=2.5',
                         facecolor='white', edgecolor=EDGE, lw=LW_REG, zorder=3)
    ax.add_patch(ans)
    arrow(ax, reg, 'into answer box', [(resume, row_c), (ans_x0 - LW_REG / 2 - TIP_GAP, row_c)], EDGE, LW_REG, ans)
    t_ans = text(ax, (ans_x0 + ans_x1) / 2, row_c - cap(7.5) / 2, 'red or blue?', ha='center')
    reg['text_inside'][id(t_ans)] = {'answer box'}
    band = FancyBboxPatch((lx, band_y0), band_x1 - lx, band_y1 - band_y0, boxstyle='round,pad=0,rounding_size=4',
                          facecolor=BAND, edgecolor='none', lw=0, zorder=1)
    ax.add_patch(band)
    reg['band'] = band
    # M9B: under the row, on one baseline inside the band: 'question' centred (visible glyphs) under the
    #      text tokens (the question is the model's text input), and the band label 'Language model' on one
    #      line, centred under the layers (Layer 1's left edge to the last node's right edge)
    tq0, tq1 = txt_l[0], txt_l[-1] + tok
    reg['text_tokens_x'] = (tq0, tq1)
    reg['image_tokens_x'] = (img_l[0], img_l[-1] + tok)
    qi = ink_box('question', 6.5)
    reg['q_token_label'] = text(ax, (tq0 + tq1) / 2 - (qi[0] + qi[2]) / 2, L['lm_base'], 'question', size=6.5, color=MUTED)
    # M9F: the band label sits in the band's lower right corner (visible glyphs end lm_pad before the band's
    #      right edge, the dots' own margin), so it labels the band, not the layer it would be centred under
    li = ink_box('Language model', 6.5)
    reg['lm_texts'] = [text(ax, band_x1 - L['lm_pad'] - li[2], L['lm_base'], 'Language model',
                            size=6.5, color=MUTED)]
    reg['band_x1'] = band_x1
    reg['obstacles'] += [('slot', slot), ('answer box', ans)] + [(f'block {i + 1}', b) for i, b in enumerate(blocks)]
    reg['obstacles'] += [(f'node {i + 1}', n) for i, n in enumerate(nodes)] + [(f'dot {i + 1}', d) for i, d in enumerate(dots)]
    reg['row_obstacles'] = [('slot', slot), ('answer box', ans)] + [('block', b) for b in blocks] + [('dot', d) for d in dots]
    reg['nodes'] = nodes

    # ---- entry routes: main straight down into its middle image token; later one trunk from the photo's
    #      centre onto a bar that drops into both nodes
    slot_top_tip = row_y0 + row_h + LW_REG / 2 + TIP_GAP
    arrow(ax, reg, 'main entry', [(main_c, py0), (main_c, slot_top_tip)], MAIN, LW_EMPH, slot,
          start_inset=LW_REG / 2)
    node_tip = row_c + nr + LW_REG / 2 + TIP_GAP
    bus_y = row_y0 + row_h + L['bus_dy']
    # M6F/M7: the trunk runs straight from the later photo's centre into the last node (its lower part is
    # the drop arrow into that node, tucked LW_EMPH/2 under the trunk); a branch leaves the trunk's left
    # stroke edge at bus_y, runs left and ends in a rounded elbow into node 1. Every node in between gets
    # its own drop, which starts on the branch's centre line (butt cap, so its top half hides under the
    # branch's stroke: a T). All nodes but the last sit under the gap left of the later photo.
    reg['trunk'] = line(ax, [later_c, later_c], [py0, bus_y - LW_EMPH / 2], LATER, LW_EMPH, zorder=5)
    arrow(ax, reg, f'later drop (node {n_lay})', [(later_c, bus_y), (later_c, node_tip)], LATER, LW_EMPH, nodes[-1])
    for k in range(n_lay - 2, 0, -1):
        nx_ = nodes_xs[k]
        arrow(ax, reg, f'later drop (node {k + 1})', [(nx_, bus_y), (nx_, node_tip)], LATER, LW_EMPH, nodes[k])
    arrow(ax, reg, 'later drop (node 1)', [(later_c - LW_EMPH / 2, bus_y), (n1x, bus_y), (n1x, node_tip)],
          LATER, LW_EMPH, nodes[0], radius=2.0)
    reg['route'] = {'main_token_c': main_tok_c, 'n_layers': n_lay, 'bus_y': bus_y}

    # M9B: no questions under the band any more; they sit in the middle column (below)

    # ---- between the photographs: the edit, top to bottom.
    #      (1) the transformation: one arrow on the bus's centre line, from the main photo's frame to the
    #          later photo's frame (main's version becomes later's version by this edit);
    #      (2) the edited-object key right under it, still within the bus's height;
    #      (3) the two sources in words, a and b each once;
    #      (4) the control-object key on the boat's centre line.
    #      All left-aligned on one column, the column centred in the gap.
    _, _, changed = load_zoom(spec, info)        # canvas box of the pixels that differ between d_a and d_b
    full_a = np.asarray(Image.open(info['d_a']).convert('RGB'))
    full_b = np.asarray(Image.open(info['d_b']).convert('RGB'))
    dmask = np.any(full_a != full_b, axis=2)
    mean_a, mean_b = full_a[dmask].mean(axis=0), full_b[dmask].mean(axis=0)
    reg['defs_colour'] = {'changed_pixels': int(dmask.sum()), 'changed_xyxy': list(changed),
                          'mean_rgb_d_a': [round(float(v), 1) for v in mean_a],
                          'mean_rgb_d_b': [round(float(v), 1) for v in mean_b],
                          'a_red': bool(mean_a[0] > mean_a[2] + 20), 'b_blue': bool(mean_b[2] > mean_b[0] + 20)}
    # M8: the two sources are stated as what the edited object looks like in each ('source a: red',
    # 'source b: blue'), as sub-lines of the 'edited object' key in the key's own type (6.5 pt, muted),
    # left-aligned on the key label. The earlier free-standing block 'source a: bus red' / 'source b: bus
    # blue' (7.5 pt, between the keys) is gone.
    # M9: the merged figure names no sources: the arrow says 'bus recolored', and 'source a' / 'source b'
    # appear nowhere (their letters collided with the panel letters and needed a definition of their own).
    defs_txt = ()
    # M12: each object's block in the middle column holds four elements, top to bottom: a label row (the
    # object's key symbol, 3 pt, then its label 'bus recolored' / 'boat unchanged', 7.5 pt ink), centred as
    # one unit on the gap's centre; an arrow from the main photo's frame to the later photo's frame on the
    # object's centre line; the role line ('Target question' / 'Control question', 6.5 pt muted); the
    # question (q_size, ink). The label row's lowest ink (glyphs or key) sits edit_gap above the arrow's
    # stroke and the role line's top ink key_gap under it (the M11B label and key-row gaps). The M11B key
    # lines 'edited object' / 'control object' are gone: the symbol beside each label names its mark.
    q_sz = L['q_size']
    lead2 = L['mid_lead'][1]           # role line baseline to question baseline (M10: 8.5)
    lab_sz = 7.5
    blocks_txt = {'target': (spec['edit_label'], 'Target question', spec['q_target']),
                  'control': (spec['control_label'], 'Control question', spec['q_control'])}
    col_w = max(max(key_text_dx + ink_box(k_, lab_sz)[2], ink_box(r_, 6.5)[2], ink_box(q_, q_sz)[2])
                for k_, r_, q_ in blocks_txt.values())
    gx0 = main_x + P + LW_REG / 2
    gx1 = later_x - LW_REG / 2
    gc = (gx0 + gx1) / 2              # M9F: every line of the middle column is centred on the gap's centre
    cx = gc - col_w / 2               # left end of the widest line (reported)

    def centred_x(s_, size_):
        """Pen x that centres the visible glyphs of s_ on the gap's centre."""
        g_ = ink_box(s_, size_)
        return gc - (g_[0] + g_[2]) / 2

    def key_row_x(s_, size_=6.5):
        """Outer left edge of a key whose row (key, 3 pt, label glyphs) is centred on the gap's centre."""
        g_ = ink_box(s_, size_)
        return gc - (key_text_dx + g_[2]) / 2
    mb = ebox['main'].get_path().get_extents()
    bus_c = (mb.y0 + mb.y1) / 2
    cbb = cboxes['main'].get_path().get_extents(cboxes['main'].get_patch_transform())
    boat_c = (cbb.y0 + cbb.y1) / 2
    t_tip = gx1 - TIP_GAP
    # M11: the edit arrow class: edit_lw (LW_EDIT, 1.3 pt) with its own head (HEADS[LW_EDIT]).
    # M12: its colour comes from 'arrow_colours' (ARROW_COLOURS): coral_muted (M12A, default) draws the bus
    # arrow in the coral of the edit marks and the boat arrow in the muted grey of the secondary text;
    # muted (M12B) draws both in that grey. The labels stay in ink.
    elw = L['edit_lw']
    if elw not in HEADS:
        raise SystemExit(f'edit_lw {elw} has no head class')
    if L['arrow_colours'] not in ARROW_COLOURS:
        raise SystemExit(f"arrow_colours must be one of {sorted(ARROW_COLOURS)}")
    col_t, col_c = ARROW_COLOURS[L['arrow_colours']]
    bot_line = py0 + LW_REG / 2                                    # photos' bottom frame stroke, top edge
    kd_half = kh_ / 2 + LW_EMPH / 2                                # dashed key: half its outer height
    mid = {}
    for nm, y_, colour, aname in (('target', bus_c, col_t, 'edit (main to later)'),
                                  ('control', boat_c + L['ctrl_dy'], col_c, 'control (main to later)')):
        lab_s, role, q_ = blocks_txt[nm]
        arrow(ax, reg, aname, [(gx0, y_), (t_tip, y_)], colour, elw, fl)
        a_ = reg['arrows'][-1]
        # label row: key centred on the label's cap-height middle; its lowest ink edit_gap above the stroke
        k_half = kd_half if nm == 'target' else ch / 2
        low_rel = min(ink_box(lab_s, lab_sz)[1], cap(lab_sz) / 2 - k_half)
        base = y_ + elw / 2 + L['edit_gap'] - low_rel
        ky_ = base + cap(lab_sz) / 2
        kx_ = key_row_x(lab_s, lab_sz)
        if nm == 'target':
            key_ = dashed_box(ax, kx_ + LW_EMPH / 2, ky_ - kh_ / 2, kw_, kh_)
            reg['dashed'].append(key_)
            reg['obstacles'].append(('edited key', key_))
        else:
            cw = kw_ + LW_EMPH
            key_ = Rectangle((kx_, ky_ - ch / 2), cw, ch, facecolor=TOKEN, edgecolor='none', lw=0, zorder=4)
            wi = 1.4
            wbox = Rectangle((kx_ + wi, ky_ - ch / 2 + wi), cw - 2 * wi, ch - 2 * wi, fill=False, lw=LW_CTRL,
                             edgecolor='white', joinstyle='miter', zorder=5)
            ax.add_patch(key_)
            ax.add_patch(wbox)
            reg['obstacles'].append(('control key', key_))
        lab_t = text(ax, kx_ + key_text_dx, base, lab_s, size=lab_sz)
        role_base = y_ - elw / 2 - L['key_gap'] - ink_box(role, 6.5)[3]
        role_t = text(ax, centred_x(role, 6.5), role_base, role, size=6.5, color=MUTED)
        q_t = text(ax, centred_x(q_, q_sz), role_base - lead2, q_, size=q_sz)
        mid[nm] = {'key': key_, 'label': lab_t, 'arrow': a_, 'y': y_, 'colour': colour,
                   'texts': [role_t, q_t], 'base': role_base - lead2, 'label_base': base, 'role_base': role_base,
                   'key_y': ky_}
    reg['transform'] = mid['target']['arrow']
    reg['edit_label'] = mid['target']['label']
    reg['edit_arrow_names'] = ['edit (main to later)', 'control (main to later)']
    reg['edit_arrow_colours'] = {'edit (main to later)': col_t, 'control (main to later)': col_c}
    reg['arrow_colours'] = L['arrow_colours']
    reg['control_arrow'] = {'arrow': mid['control']['arrow'], 'y': mid['control']['y'],
                            'label': mid['control']['label'], 'bot_line': bot_line}
    ctrl = reg['control_arrow']
    reg['required_heads'] = BASE_HEADS | {HEADS[elw]}
    mid['boat_c'], mid['bus_c'], mid['bus_span'] = boat_c, bus_c, (mb.y0, mb.y1)
    mid['boat_span'] = (cbb.y0, cbb.y1)
    mid['column'] = (cx, col_w, gx0, gx1)
    mid['gap_centre'] = gc
    mid['key_text_dx'] = key_text_dx
    mid['lab_sz'] = lab_sz
    reg['mid'] = mid
    reg['defs'] = []
    reg['defs_txt'] = defs_txt
    reg['panel_a_box'] = (lx, 3.0, None, lb + cap(7.5))
    geo.update({
        'photos': {'main_x': [main_x, main_x + P], 'later_x': [later_x, later_x + P], 'y': [round(py0, 2), round(py1, 2)],
                   'side_pt': P, 'gap_pt': round(gap, 2), 'pt_per_px': round(P / img_a.shape[1], 4)},
        'entry_label_baseline': lb,
        'band': [lx, band_y0, round(band_x1, 2), band_y1], 'row': [row_y0, row_y0 + row_h, row_c],
        'slot': [in_x0, in_x1], 'blocks_x': [[b.get_x(), b.get_x() + bw] for b in blocks],
        'nodes_x': [n.center[0] for n in nodes], 'slot_to_layer1_arrow_pt': round(b0 - in_x1, 2),
        'branch': {'y': bus_y, 'x': [n1x, later_c], 'stroke_y': [bus_y - LW_EMPH / 2, bus_y + LW_EMPH / 2]},
        'answer_box': [round(ans_x0, 2), round(ans_x1, 2)],
        'bus_y': bus_y, 'trunk_x': later_c, 'main_arrow_x': main_tok_c,
        'question_baselines': {nm: round(mid[nm]['base'], 2) for nm in ('target', 'control')},
        'middle': {'x': [round(gx0, 2), round(gx1, 2)], 'column_x': round(cx, 2), 'column_w': round(col_w, 2),
                   'centre_x': round(gc, 2), 'side_clearance_pt': round(cx - gx0, 2),
                   'transform_arrow_y': round(bus_c, 3), 'edited_key_y': round(mid['target']['key_y'], 3),
                   'control_key_y': round(mid['control']['key_y'], 3),
                   'boat_centre_y': round(boat_c, 3), 'boat_span_y': [round(cbb.y0, 2), round(cbb.y1, 2)],
                   'bus_span_y': [round(mb.y0, 2), round(mb.y1, 2)],
                   'arrow_colours': L['arrow_colours'],
                   'edit_arrow': {'stroke_pt': elw, 'head_pt': list(HEADS[elw]), 'colour': col_t},
                   'control_arrow': {'stroke_pt': elw, 'head_pt': list(HEADS[elw]), 'colour': col_c},
                   'control_arrow_y': round(ctrl['y'], 3),
                   'lines': {f'{nm}: {t_.get_text()}': round(t_.get_position()[1], 3)
                             for nm in ('target', 'control') for t_ in [mid[nm]['label']] + mid[nm]['texts']},
                   'definition_baselines': []},
    })
    close_zone('a')

    # ---- panel b: the headline result (unchanged content; placed after the setup)
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    a_right = max(outer_bbox(a, rr)[2] for a in zones['a'])
    xc = L['chart_x0']                      # M7b: 293.5 (M7 296, M6G 306); the 2-D setup-to-chart rule is check 9
    heading(ax, xc, head_base, 'b', 'Which entry does the answer follow?')
    reg['letter_b'] = ax.texts[-2]
    name_r = xc + 41.5
    ent_l = name_r + 3.5
    bx0 = ent_l + 19.5
    full = L['chart_full']
    bar_h, in_gap, grp_gap = 8.0, 2.5, 10.0
    route_header = spec.get('route_header_merged', spec['route_header'])
    t_b = text(ax, ent_l, top - cap(6.5), route_header, size=6.5, color=MUTED)
    reg['top_lines']['panel b header (text top)'] = t_b
    y_first = 126.5
    bars = []
    for gi, model in enumerate(COUNTS):
        ym = y_first - gi * (2 * bar_h + in_gap + grp_gap)
        yl = ym - (bar_h + in_gap)
        text(ax, name_r, (ym + yl) / 2 - cap(7.5) / 2, model, weight='bold', ha='right')
        for entry, yy, col, ink in (('main', ym, MAIN, MAIN_INK), ('later', yl, LATER, LATER_INK)):
            c, n = COUNTS[model][entry]
            p = pct(c, n)
            text(ax, ent_l, yy - 2.3, entry, color=ink)
            wbar = max(full * c / n, .9)
            bar = Rectangle((bx0, yy - bar_h / 2), wbar, bar_h, facecolor=col, edgecolor='none', lw=0, zorder=3)
            ax.add_patch(bar)
            bars.append(bar)
            text(ax, bx0 + wbar + 2.2, yy - cap(7.5) / 2, f'{p}%')
    bar_top = y_first + bar_h / 2
    axis_y = bars[-1].get_y() - 3.5
    line(ax, [bx0 - LW_HAIR / 2, bx0 + full + LW_HAIR / 2], [axis_y, axis_y], EDGE, LW_HAIR)
    line(ax, [bx0, bx0], [axis_y, bar_top], EDGE, LW_HAIR, zorder=2)
    for v in (0, 50, 100):
        xx = bx0 + full * v / 100
        line(ax, [xx, xx], [axis_y, axis_y - 2.0], EDGE, LW_HAIR)
        text(ax, xx, axis_y - 3.5 - cap(6.5), str(v), size=6.5, color=MUTED, ha='center')
        if v:
            line(ax, [xx, xx], [axis_y, bar_top], GRID, LW_HAIR, zorder=1)
    xl_base = axis_y - 3.5 - cap(6.5) - 4.7
    reg['xlabel'] = text(ax, bx0 + full / 2, xl_base - cap(6.5), spec['xlabel'], size=6.5, color=MUTED, ha='center')
    reg['obstacles'] += [('bar', b) for b in bars]
    chart_ink = [a for a in ax.get_children() if a not in seen and a.get_visible()]   # M7: chart only (no credit)
    lines_ = spec['credit']
    cb0, cb1 = L['credit_base']                # M9B: fixed baselines (11.1, 4.1), as in M9base
    cr = [text(ax, W - 2.5, (cb0, cb1)[k], ln_, size=6.0, color=MUTED, ha='right')
          for k, ln_ in enumerate(lines_)]
    reg['credit'] = cr[0]
    reg['credit_lines'] = list(lines_)
    geo['chart'] = {'x0': xc, 'name_right': name_r, 'entry_left': ent_l, 'bar_x0': bx0, 'full_width': full,
                    'bar_h': bar_h, 'in_group_gap': in_gap, 'group_gap': grp_gap,
                    'bar_centres_y': [round(b.get_y() + bar_h / 2, 2) for b in bars],
                    'axis_y': axis_y, 'credit_baselines': [cb0, cb1]}
    close_zone('b')
    reg['chart_ink'] = chart_ink
    reg['required_text'] = (['a', spec['title_merged'], 'b', 'Which entry does the answer follow?',
                             'Main entry', spec['main_label'], 'Later entries', spec['later_sub'], spec['edit_label'],
                             spec['control_label'],
                             *reg['defs_txt'],
                             *[t.get_text() for t in reg['lm_texts']], *[f'Layer {i + 1}' for i in range(n_lay)], 'red or blue?', 'question', 'Target question', 'Control question',
                             spec['q_target'], spec['q_control'], route_header, 'main', 'later',
                             *COUNTS, *{f'{pct(c, n)}%' for d in COUNTS.values() for c, n in d.values()},
                             '0', '50', '100', spec['xlabel'], *reg['credit_lines']])
    reg['credit_sentence'] = ' '.join(spec['credit'])

    checks = run_checks(fig, ax, reg, zones)
    layout_issues = checks['4_text_layout']['issues']
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{spec['stem']}.pdf"
    png = outdir / f"{spec['stem']}.png"
    meta = {'Title': 'Figure 1', 'Creator': None, 'Producer': None, 'CreationDate': None}
    fig.savefig(pdf, dpi=dpi, metadata=meta)
    fig.savefig(png, dpi=dpi)
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text()]
    plt.close(fig)
    info.update({'pdf': str(pdf), 'png': str(png), 'size_pt': [W, H], 'min_font_pt': min(sizes),
                 'text_layout_issues': layout_issues, 'checks_pass': checks['all_pass'],
                 'checks': checks, 'geometry_pt': geo, 'layout': 'merged'})
    return info


if __name__ == '__main__':
    output_dir()
    setup_rc()
    load_counts()
    build_merged('photo', output_dir(), 300)
