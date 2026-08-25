"""
cb_style.py
===========
Colorblind-safe plotting defaults.  Call cb_style.apply() at the top of any
plotting code, and use OKABE_ITO for discrete series.

Okabe-Ito is the standard 8-colour palette distinguishable under all common
forms of colour-vision deficiency.  Sequential maps use 'cividis' (perceptually
uniform AND colorblind-safe).  Avoid jet/rainbow/magma.
"""

OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00',
             '#CC79A7', '#56B4E9', '#F0E442', '#000000']
SEQ_CMAP = 'cividis'
LINESTYLES = ['-', '--', ':', '-.']
HATCHES = ['', '///', '...', 'xxx', '\\\\\\', '+++', 'ooo']

def apply():
    import matplotlib as mpl
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=OKABE_ITO)
    mpl.rcParams['image.cmap'] = SEQ_CMAP
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3


def styles(n):
    """Return n line-style dicts that are pairwise visually distinct even when
    n > len(OKABE_ITO): we cycle colour fastest, then dash pattern, so no two
    series ever share BOTH colour and linestyle. Good for up to 8*4 = 32 lines.
    Use as: for st, ...: ax.plot(x, y, **st, label=...)."""
    nc = len(OKABE_ITO)
    return [dict(color=OKABE_ITO[i % nc],
                 linestyle=LINESTYLES[(i // nc) % len(LINESTYLES)])
            for i in range(n)]


def fills(n):
    """Distinct (facecolor, hatch) pairs for stacked-area / bar plots, where
    linestyle is unavailable. Colour cycles fastest, hatch disambiguates the
    wrap-around so adjacent same-colour bands stay separable."""
    nc = len(OKABE_ITO)
    return [dict(facecolor=OKABE_ITO[i % nc],
                 hatch=HATCHES[(i // nc) % len(HATCHES)])
            for i in range(n)]
