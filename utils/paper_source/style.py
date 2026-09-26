"""Paper-like Matplotlib output without a typesetting dependency."""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from .paths import FIG_DIR

MODEL_COLORS = {"ResNet+":"#1f7a5c","NN+":"#9a3d3f","ResNet":"#1f7a5c","NN":"#9a3d3f"}
_LAST_FIGURE = None

def init_mpl(*,usetex=False):
    mpl.rcParams.update({"text.usetex":False,"font.family":"serif","font.size":10,
        "axes.labelsize":10,"axes.titlesize":10,"legend.fontsize":8,"xtick.labelsize":8,
        "ytick.labelsize":8,"axes.spines.top":False,"axes.spines.right":False,
        "axes.grid":True,"grid.color":"#d9d9d9","grid.linestyle":"--","grid.linewidth":.6,
        "axes.unicode_minus":False})

def savefig_pair(name,fig):
    global _LAST_FIGURE
    _LAST_FIGURE=fig
    FIG_DIR.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG_DIR/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(FIG_DIR/f'{name}.pdf',bbox_inches='tight',metadata={"Creator":"ResAssetPricing","CreationDate":None,"ModDate":None})
