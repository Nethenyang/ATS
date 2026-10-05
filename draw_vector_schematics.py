"""Editable merged motivation/framework diagram; no raster inputs or tracing."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

OUT = Path(__file__).resolve().parent / 'artifacts' / 'reproduced' / 'figures'
BLUE, ORANGE, INK = '#0072B2', '#D55E00', '#172B4D'
MUTED, EDGE = '#536579', '#D5DEE8'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'font.size': 12})

def text(ax, x, y, label, size=12, weight='normal', color=INK):
    return ax.text(x, y, label, fontsize=size, fontweight=weight, color=color,
                   ha='center', va='center', linespacing=1.5)

def box(ax, x, y, w, h, color=EDGE, fill='#FAFBFD', radius=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}', linewidth=1.25,
        edgecolor=color, facecolor=fill))

def arrow(ax, points, color=INK):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.3)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
        mutation_scale=11, linewidth=1.3, color=color, shrinkA=0, shrinkB=1))

def document(ax, x, y, w, h, color, fill, label=None):
    fold = w*.23
    points = [(x, y), (x+w-fold, y), (x+w, y+fold), (x+w, y+h), (x, y+h)]
    ax.add_patch(Polygon(points, closed=True, facecolor=fill, edgecolor=color,
                         linewidth=1.2, joinstyle='round'))
    ax.plot([x+w-fold, x+w-fold, x+w], [y, y+fold, y+fold], color=color, linewidth=1.0)
    if label:
        text(ax, x+w/2, y+h*.66, label, 10.5, 'bold', color)

def main():
    fig = plt.figure(figsize=(8, 5.25), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1050); ax.set_ylim(690, 0); ax.axis('off')
    # No overall title or footer slogan; explanation belongs in the caption.
    box(ax, 28, 20, 230, 104)
    text(ax, 143, 43, 'Reference audio', 12.5, 'bold')
    heights = [3, 8, 17, 23, 16, 9, 5, 12, 18, 11, 5, 2]
    for i, height in enumerate(heights):
        x = 54+i*9
        ax.plot([x, x], [88-height, 88+height], color=BLUE, linewidth=1.5,
                solid_capstyle='round')
    text(ax, 206, 87, 'x[n]', 12)
    arrow(ax, [(264, 72), (322, 72)])
    box(ax, 330, 20, 336, 104)
    text(ax, 498, 42, 'Candidate encoding', 12.5, 'bold')
    for x, label, color, fill in [
        (347, 'WAV', MUTED, '#EAF0F5'), (425, 'MP3', BLUE, '#E4F1FA'),
        (503, 'AAC', ORANGE, '#FFF0E3'), (581, 'Opus', '#258268', '#E7F5EF')]:
        document(ax, x, 63, 65, 50, color, fill, label)
    text(ax, 498, 140, 'Measure file bytes B(c)', 10.5, color=MUTED)
    arrow(ax, [(672, 72), (752, 72)])
    box(ax, 760, 20, 262, 104)
    text(ax, 891, 43, 'Decode + align', 12.5, 'bold')
    text(ax, 891, 86, 'Common grid · fixed delay\nFull reference duration', 10.5)
    arrow(ax, [(891, 129), (891, 160), (525, 160), (525, 175)])
    # The two constraints share one evaluation module and Loud/Quiet motivation.
    box(ax, 28, 180, 994, 396)
    text(ax, 525, 207, 'Fidelity evaluation', 13.5, 'bold')
    text(ax, 140, 270, 'Reference\nactive frames', 11.5, color=MUTED)
    text(ax, 522, 235, 'Loud', 11.5, 'bold', BLUE)
    text(ax, 842, 235, 'Quiet', 11.5, 'bold', ORANGE)
    # Nine loud and one quiet equal-duration frames match the analytical example.
    for i in range(10):
        quiet = i == 9
        x = 239+i*64
        color = ORANGE if quiet else BLUE
        box(ax, x, 249, 55, 45, color,
            '#FFF1E5' if quiet else '#EAF4FC', radius=4)
        for j, amplitude in enumerate([6, 13, 18, 12, 6]):
            h = amplitude*.02 if quiet else amplitude
            xx = x+9+j*9
            ax.plot([xx, xx], [271.5-h, 271.5+h], color=color, linewidth=1.2,
                    solid_capstyle='round')
    # Shared branch: both metrics compare reconstruction with the reference.
    ax.plot([525, 525], [299, 313], color=MUTED, linewidth=1.1)
    ax.plot([277, 773], [313, 313], color=MUTED, linewidth=1.1)
    arrow(ax, [(277, 313), (277, 327)], BLUE)
    arrow(ax, [(773, 313), (773, 327)], ORANGE)
    box(ax, 72, 330, 410, 178, BLUE, '#EAF4FC')
    text(ax, 277, 355, 'Global fidelity', 13.5, 'bold', BLUE)
    text(ax, 277, 406, 'Loud frames dominate\nenergy-weighted global error', 11.5)
    text(ax, 277, 468, r'$Q_{\mathrm{G}}\geq g$', 15, 'bold', BLUE)
    box(ax, 568, 330, 410, 178, ORANGE, '#FFF1E5')
    text(ax, 773, 355, 'Temporal tail', 13.5, 'bold', ORANGE)
    text(ax, 773, 406, 'Worst active-frame relative errors\nexpose quiet-frame distortion', 11.5)
    text(ax, 773, 468, r'$Q_{\mathrm{T}}\geq t$', 15, 'bold', ORANGE)
    text(ax, 525, 547, 'Both constraints must pass', 12.5, 'bold')
    arrow(ax, [(525, 581), (525, 600)])
    box(ax, 330, 605, 390, 76, INK, '#F4F6F8')
    document(ax, 351, 622, 35, 43, INK, '#FFFFFF')
    text(ax, 546, 625, 'Smallest feasible file', 12.5, 'bold')
    text(ax, 546, 657, r'$c^*=\arg\min_{c\;\mathrm{feasible}} B(c)$', 11.5)
    OUT.mkdir(parents=True, exist_ok=True)
    name = 'graphical_abstract'
    fig.savefig(OUT/(name+'.svg'), metadata={
        'Description': 'Unified framework and Loud/Quiet motivation; editable vector primitives; no measured traces.'})
    fig.savefig(OUT/(name+'.pdf'))
    fig.savefig(OUT/(name+'_preview.png'), dpi=200)
    plt.close(fig)
    print('Created one merged, untitled SVG/PDF diagram and a preview.')

if __name__ == '__main__':
    main()
