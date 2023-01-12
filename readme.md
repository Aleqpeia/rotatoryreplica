Lipid Bilayer Analysis and Enhanced Sampling Tools
This repository contains a collection of Python scripts for analyzing and manipulating molecular dynamics data, specifically for the purpose of
guiding enhanced sampling simulations and analyzing the behavior of lipid bilayers.

Installation
To use the scripts in this repository, you will need to have Python 3 and the following packages installed:

MDAnalysis
NumPy
PyVista and open3d (for membrane surface reconstruction)

Right now it's bunch of functions, sorted between scripts in toolbox, lpanalysis.py for lipid bilayer analysis geometrical and
specific biophysical properties analysis, onthefly.py for manipulations, procfunc.py for preprocessing and processing, visual.py for plotting.

But, there's plans for development:
Add scripts for analyzing and visualizing the properties of lipid bilayers such as thickness, tilt angle, per lipid area,
Improve the visualization scripts to map biophysical parameters on rendered surface
Work on algorithm of phase separation detection and phase mapping
