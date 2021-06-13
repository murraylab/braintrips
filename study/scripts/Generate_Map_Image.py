# NOTE: wbplot will not work if you're on a Windows machine. This is a current
#  limitation of the Connectome Workbench software!

from braintrips.data import group_dgbc

import wbplot
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from scipy.stats import zscore

# change to a path on your local machine
image = "/Users/jbb/Desktop/empirical_group_dgbc.png"


# My custom colormap for this paper
def tbkry_cmap(set_bad='w'):
    cdict1 = {'green': ((0.0, 0.0, 1.0),
                        (0.25, 0, 0),
                        (0.75, 0, 0),
                        (1.0, 1.0, 0.0)),
              'blue': ((0.0, 0.0, 1.0),
                       (0.25, 1.0, 1.0),
                       (0.5, 0., 0.0),
                       (1.0, 0.0, 0.0)),
              'red': ((0.0, 0.0, 0.0),
                      (0.5, 0.0, 0.0),
                      (0.75, 1.0, 1.0),
                      (1.0, 1.0, 1.0))
              }
    cmap_ = LinearSegmentedColormap('tbkry', cdict1)
    cmap_.set_bad(set_bad)
    return cmap_


# Register the colormap
plt.register_cmap(name='tbkry', cmap=tbkry_cmap())

# Save image
wbplot.pscalar(
    image, zscore(group_dgbc),
    orientation='landscape',
    hemisphere='left',
    cmap='tbkry',
    vrange=(-2, 2),
    transparent=True
)
