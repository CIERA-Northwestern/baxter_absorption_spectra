# baxter_absorption_spectra

Code for making lightcurves and absorption spectra for outreach / educational activities.

## To make absorption spectra

1. Make a raw spectrum csv file using a config file (e.g., for a 'rocky' planet): 

> python3 make_planet_spectrum.py planet_config_files/rocky1

Data will be output to outreach_data/rocky1/


2. Then create plots of the raw spectrum and plots of elemental filters for element identification:

> python3 plot_spectrum_and_filters.py outreach_data/rocky1

Data will be output to outreach_data/rocky1/ and outreach_data/rocky1/filters/
