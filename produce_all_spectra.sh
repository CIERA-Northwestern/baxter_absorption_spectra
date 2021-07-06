for config in planet_config_files/*
do
    python3 make_planet_spectrum.py $config None
    python3 plot_spectrum_and_filters.py outreach_data/`basename $config`

    python3 make_planet_spectrum.py $config random
    python3 plot_spectrum_and_filters.py outreach_data/`basename $config`_techno
done
