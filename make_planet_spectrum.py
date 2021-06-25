"""
Beer-Lambert Law: I = I0 * exp(-alpha * L * n)

I = intensity
I0 = entering intensity
alpha = absorption cross-section
L = absorption path length
n = density of absorber
L * n = N = column density

Usage:
    make_planet_spectrum.py <planet>

"""
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
for k, frac in elements.items():
    elements[k] = float(frac)
    


c = 3.0e8 #m/s
nm_to_m = 1e-9

def add_line(wavelength, intensity, column_density, atmo_velocity, line_center, inverse_lifetime, oscillator_strength=1):
    freq = c / (nm_to_m * wavelength)
    center_freq = c / (nm_to_m * line_center)
    return intensity*np.exp(-column_density*oscillator_strength*voigt_profile(freq - center_freq, atmo_velocity, inverse_lifetime))

hydrogen = np.genfromtxt('line_files/hydrogen.csv', delimiter=',', skip_header=2)
helium = np.genfromtxt('line_files/helium.csv', delimiter=',', skip_header=2)

cols = [1e17, 1e12]
atmo_sigma = 1e12

lambdas = np.linspace(1, 1000, 10000)

for N in cols:
    intensities = np.ones_like(lambdas)
    for ele, ele_frac in elements.items():
        ele_data = np.genfromtxt('line_files/{}.csv'.format(ele), delimiter=',', skip_header=2)
        N_ele = ele_frac*N
        for i in range(ele_data.shape[0]):
            line = tuple(hydrogen[i,:])
            intensities = add_line(lambdas, intensities, N, atmo_sigma, *line)

    plt.plot(lambdas, intensities, label='N={}'.format(N))
plt.show()
    
