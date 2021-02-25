VLA CASA Scripts
================
CASA imaging pipelien for JVLA project 17A-146. The pipeline includes
functionality single-dish image combination with GBT data and automated
deconvolution of all lines.

The pipeline may be run using the ``test_imaging.py`` script in CASA v5.6. The
``iter_feather.py`` script provides an implementation of the "SD+INT"
single-dish+interferometric image combination algorithm described in Rau et al.
((2019)[https://ui.adsabs.harvard.edu/abs/2019AJ....158....3R/abstract]) to
combine the GBT+VLA image cubes using the ``PySynthesisImager`` API.

License
-------
Copyright 2020 Brian Svoboda and released under the MIT License.
