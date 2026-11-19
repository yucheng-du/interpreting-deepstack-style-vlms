"""Shared paper-figure presentation: Figure 1 palette, plain panels on a white canvas.

Colors and decorative surfaces only: callers retain their data, scales and text.
All panel coordinates are PDF points, measured from the lower-left corner.
"""
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

CANVAS = '#FFFFFF'
HEADER = '#E7EFF6'
BORDER = '#D4E1EC'
BADGE = '#5B7891'
INK = '#20374C'
MAIN = '#3188B8'
LATER = '#199A86'
CORAL = '#E77965'
MUTED = '#526677'
# Added for the v19 Figures 2-3; existing names keep their meaning.
LATER_INK = '#0E6B5D'    # dark teal for numeric labels on later-entry marks
CELL_CC = '#2B4156'      # both clean: darkest cell, solid
CELL_DD = '#C9D6E2'      # both degraded: lightest cell, hatched
HATCH_LIGHT = '#FFFFFF'  # hatch lines on dark fills
HATCH_DARK = '#5F7488'   # hatch lines on light fills
DD_LINE = '#7A8FA3'      # readable grey-blue line for the both-degraded series


def apply_paper_style():
    mpl.rcParams.update({
        'font.family': 'Arial', 'text.color': INK,
        'axes.labelcolor': INK, 'axes.edgecolor': '#8DA3B5',
        'xtick.color': INK, 'ytick.color': INK,
        'figure.facecolor': CANVAS, 'axes.facecolor': 'white',
        'savefig.facecolor': 'auto', 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none',
    })


def panel_card(fig, bounds_pt, letter, title, title_size=9.3, header_height=19):
    """Add a plain panel heading: a bold title, preceded by a bold letter only when given.

    Nothing frames the panel. The returned regions keep the earlier card geometry but
    are invisible, so callers' layout checks still test that headings stay in their panel.
    """
    x, y, width, height = bounds_pt
    fw, fh = fig.get_size_inches() * 72
    transform = Affine2D().scale(1/fw, 1/fh) + fig.transFigure
    surface = Rectangle((x, y), width, height, transform=transform, fill=False,
                        edgecolor='none', zorder=-10)
    band = Rectangle((x, y+height-header_height), width, header_height, transform=transform,
                     fill=False, edgecolor='none', zorder=-9)
    fig.add_artist(surface)
    fig.add_artist(band)
    cy = y+height-header_height/2
    title_x = x+5.5
    if letter:
        badge = Rectangle((x+5, cy-6), 12, 12, transform=transform, fill=False,
                          edgecolor='none', zorder=4)
        fig.add_artist(badge)
        letter_text = fig.text((x+5.5)/fw, cy/fh, letter, fontsize=title_size+.6, weight='bold',
                               color=INK, ha='left', va='center', zorder=5)
        title_x = x+18
    title_text = fig.text(title_x/fw, cy/fh, title, fontsize=title_size, weight='bold',
                          color=INK, ha='left', va='center', zorder=5)
    if not letter:
        # Without a letter, the title stands in for it in callers' containment checks.
        badge, letter_text = band, title_text
    return {'surface': surface, 'band': band, 'badge': badge,
            'letter': letter_text, 'title': title_text}
