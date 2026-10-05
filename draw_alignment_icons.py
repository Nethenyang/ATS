"""Draw three matching alignment icons as transparent editable SVG line art."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT=Path(__file__).resolve().parent/'artifacts'/'reproduced'/'figures'/'alignment_icons'
COLOR='#536579'
STYLE=dict(color=COLOR,linewidth=1.8,solid_capstyle='round',solid_joinstyle='round')

def line(ax,points,yoff=0):
    ax.plot([p[0] for p in points],[p[1]+yoff for p in points],**STYLE)

def grid(ax,yoff=0):
    # Equal sampling positions on two signals share the same reference grid.
    for x in [10,17,24,31,38]:
        line(ax,[(x,8),(x,40)],yoff)
    line(ax,[(8,17),(40,17)],yoff)
    line(ax,[(8,31),(40,31)],yoff)
    for x in [10,17,24,31,38]:
        for y in [17,31]:
            ax.add_patch(Circle((x,y+yoff),1.75,facecolor=COLOR,edgecolor='none'))

def delay(ax,yoff=0):
    # Two identical pulses and a horizontal arrow denote a single translation.
    line(ax,[(7,15),(12,15),(12,8),(21,8),(21,15),(36,15)],yoff)
    line(ax,[(7,29),(20,29),(20,22),(29,22),(29,29),(41,29)],yoff)
    line(ax,[(12,38),(29,38)],yoff)
    line(ax,[(16,35),(12,38),(16,41)],yoff)
    line(ax,[(25,35),(29,38),(25,41)],yoff)

def duration(ax,yoff=0):
    # Both endpoints are retained; the bounded waveform spans the full interval.
    line(ax,[(12,8),(7,8),(7,40),(12,40)],yoff)
    line(ax,[(36,8),(41,8),(41,40),(36,40)],yoff)
    line(ax,[(11,24),(14,24),(16,17),(19,32),(22,12),(25,35),(28,19),(31,29),(34,24),(37,24)],yoff)

def canvas(height):
    fig=plt.figure(figsize=(48/72,height/72))
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,48); ax.set_ylim(height,0)
    ax.set_aspect('equal'); ax.axis('off')
    return fig,ax

def save(fig,name):
    fig.savefig(OUT/(name+'.svg'),transparent=True,metadata={'Creator':'Deterministic vector drawing; no raster inputs'})
    fig.savefig(OUT/(name+'.png'),transparent=True,dpi=288)
    plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name,draw in [('common_time_grid',grid),('fixed_delay',delay),('full_duration',duration)]:
        fig,ax=canvas(48); draw(ax); save(fig,name)
    fig,ax=canvas(168)
    for offset,draw in [(0,grid),(60,delay),(120,duration)]: draw(ax,offset)
    save(fig,'alignment_icons_vertical')
    # A larger horizontal preview lets the author inspect the icons together.
    fig,axes=plt.subplots(1,3,figsize=(6,2),facecolor='white')
    for ax,draw in zip(axes,[grid,delay,duration]):
        ax.set_xlim(0,48);ax.set_ylim(48,0);ax.set_aspect('equal');ax.axis('off');draw(ax)
    fig.subplots_adjust(left=.015,right=.985,bottom=.03,top=.97,wspace=.20)
    fig.savefig(OUT/'icons_preview.png',dpi=180)
    plt.close(fig)
    print('Saved 3 individual SVGs and 1 vertical combined SVG. Color: '+COLOR)

if __name__=='__main__': main()
