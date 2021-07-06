"""
Generates a simple CSV file of an absorption spectrum of a planet

Beer-Lambert Law: I = I0 * exp(-alpha * L * n)

I = intensity
I0 = entering intensity
alpha = absorption cross-section
L = absorption path length
n = density of absorber
L * n = N = column density

Usage:
    make_line.py <planet>

"""
import os
from configparser import ConfigParser
from pathlib import Path

from docopt import docopt
import numpy as np
from scipy.special import voigt_profile
import matplotlib.pyplot as plt

args = docopt(__doc__)
#Read planet file
if args['<planet>'] is not None: 
    planet_file = Path(args['<planet>'])
    planet = ConfigParser()
    planet.read(str(planet_file))
    elements = dict(planet.items('elements'))
    atmosphere = dict(planet.items('atmosphere'))
for k, frac in elements.items():
    elements[k] = float(frac)
for k, frac in atmosphere.items():
    atmosphere[k] = float(frac)

    
c = 3.0e8 #m/s
nm_to_m = 1e-9
min_freq = c / (nm_to_m * 100)

def add_line(wavelength, intensity, column_density, atmo_velocity, line_center, inverse_lifetime, oscillator_strength=1):
    freq = c / (nm_to_m * wavelength)
    center_freq = c / (nm_to_m * line_center)
    return intensity*np.exp(-column_density*oscillator_strength*voigt_profile(freq - center_freq, atmo_velocity*(center_freq/min_freq)**(1/2), inverse_lifetime))

print(atmosphere)
N = atmosphere['n']
atmo_sigma = atmosphere['sigma']

lambdas = np.linspace(1, 1000, 10000)
intensities = np.ones_like(lambdas)

for ele, ele_frac in elements.items():
    ele_data = np.genfromtxt('line_files/{}.csv'.format(ele), delimiter=',', skip_header=2)
    N_ele = ele_frac*N
    for i in range(ele_data.shape[0]):
        line = tuple(ele_data[i,:])
        intensities = add_line(lambdas, intensities, N_ele, atmo_sigma, *line)

if not os.path.exists('outreach_data/'):
    os.makedirs('outreach_data/')
data_dir = 'outreach_data/{:s}/'.format(args['<planet>'].split('/')[-1])
if not os.path.exists('{:s}'.format(data_dir)):
    os.makedirs('{:s}'.format(data_dir))

header = "wavelength (nm), relative flux"
data = np.array([lambdas, intensities]).T
np.savetxt('{:s}/raw_spectrum.csv'.format(data_dir), data, delimiter=',', header=header)

plt.plot(lambdas, intensities, label='N={}'.format(N))
plt.show()
