"""
Beer-Lambert Law: I = I0 * exp(-alpha * L * n)

I = intensity
I0 = entering intensity
alpha = absorption cross-section
L = absorption path length
n = density of absorber
L * n = N = column density

"""
import numpy as np
from scipy.special import voigt_profile
import matplotlib.pyplot as plt

c = 3.0e8 #m/s
nm_to_m = 1e-9

def add_line(wavelength, intensity, column_density, atmo_velocity, line_center, inverse_lifetime, oscillator_strength=1):
    freq = c / (nm_to_m * wavelength)
    center_freq = c / (nm_to_m * line_center)
    return intensity*np.exp(-column_density*oscillator_strength*voigt_profile(freq - center_freq, atmo_velocity, inverse_lifetime))

hydrogen = np.genfromtxt('line_files/hydrogen.csv', delimiter=',', skip_header=2)
helium = np.genfromtxt('line_files/helium.csv', delimiter=',', skip_header=2)

cols = [1e17]
atmo_sigma = 1e12

lambdas = np.linspace(1, 1000, 10000)

for N in cols:
    intensities = np.ones_like(lambdas)
    h_intensities = np.ones_like(lambdas)
    for i in range(hydrogen.shape[0]):
        line = tuple(hydrogen[i,:])
        intensities = add_line(lambdas, intensities, N, atmo_sigma, *line)
        h_intensities = add_line(lambdas, h_intensities, N, atmo_sigma, *line)
    for i in range(helium.shape[0]):
        line = tuple(helium[i,:])
        intensities = add_line(lambdas, intensities, N, atmo_sigma, *line)

    plt.plot(lambdas, intensities, label='N={}'.format(N))
    plt.plot(lambdas, h_intensities, label='N={}'.format(N))
plt.show()
    

