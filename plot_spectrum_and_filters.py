"""
Makes student-friendly plots of the data

Usage:
    plot_spectrum_and_filters.py <spectra_folder>

"""
import os
import glob
from configparser import ConfigParser
from pathlib import Path

from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

args = docopt(__doc__)

folder= args['<spectra_folder>']
spectrum = np.genfromtxt('{:s}/raw_spectrum.csv'.format(folder), delimiter=',', skip_header=1)

transparency_data = np.zeros((1000, 4))
transparency_data[:, 3] = np.linspace(1, 0, 1000)
transparency_cmap = ListedColormap(transparency_data)

fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

ax1.plot(spectrum[:,0], spectrum[:,1])

y_dim = np.array((0, 1))
yy, xx = np.meshgrid(y_dim, spectrum[:,0])
color_spectrum = np.zeros_like(xx)
intensity_spectrum = np.zeros_like(xx)
print(xx.shape)
for i in range(2):
    color_spectrum[:,i] = spectrum[:,0]
    intensity_spectrum[:,i] = spectrum[:,1]

ax2.pcolormesh(xx, yy, color_spectrum, cmap='gist_rainbow_r')
ax2.pcolormesh(xx, yy, intensity_spectrum, cmap=transparency_cmap, vmin=0, vmax=1)

ax3.pcolormesh(xx, yy, intensity_spectrum, cmap='binary_r', vmin=0, vmax=1)

for ax in [ax1, ax2, ax3]:
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlim((xx.min(), xx.max()))

fig.savefig('{:s}/raw_spectrum.png'.format(folder), dpi=300, bbox_inches='tight')


# Make element filters
elements = glob.glob('line_files/*')
for f in elements:
    line_data = np.genfromtxt(f, delimiter=',', skip_header=2)
    these_lines = []
    for i in range(line_data.shape[0]):
        for ax in [ax1, ax2, ax3]:
            these_lines.append(ax.axvline(line_data[i,0], c='r'))
    fig.suptitle('{:s}'.format(f.split('/')[-1].split('.csv')[0]))
    fig.savefig('{:s}/{:s}_lines.png'.format(folder, f.split('/')[-1].split('.csv')[0]), dpi=300, bbox_inches='tight')
    for line in these_lines:
        line.remove()
