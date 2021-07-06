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
    make_line.py <planet> <techno>

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

## read techno signature
techno = args['<techno>']

    
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


if techno != 'None':
    if data_dir[-1] == os.sep:
        data_dir=data_dir[:-1]
    technos = ['sawtooth','linear','laser','gap']
    if techno == 'random': techno = technos[np.random.randint(len(technos))]

    if techno == 'sawtooth':
        thetas = np.linspace(0,2*np.pi,intensities.size,endpoint=False)
        sawtooth_mask = np.abs(np.sin(10*thetas)>0.9)
        intensities[sawtooth_mask]*=0.2
    elif techno == 'linear':
        xs = np.arange(intensities.size)
        ys = 1-1/xs[-1]*xs
        ys[int(intensities.size//2):] = 1
        intensities*=ys
    elif techno == 'laser':
        index = np.random.randint(intensities.size)
        intensities[index]+=1
    elif techno == 'gap':
        indices = np.zeros(intensities.size,dtype=bool)
        midpoint = int(intensities.size//2)
        qtile = int(midpoint//2)
        indices[qtile:qtile+midpoint] = 1
        intensities[indices]*=0

    else:
        raise KeyError("Bad techno %s"%techno)

    if not os.path.exists('%s_techno'%data_dir):
        os.makedirs('%s_techno'%data_dir)
    data = np.array([lambdas, intensities]).T
    np.savetxt('%s_techno/raw_spectrum.csv'%data_dir, data, delimiter=',', header=header)

#plt.plot(lambdas, intensities, label='N={}'.format(N))
#plt.show()
