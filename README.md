Splitting of Taylor Hood Discretizations and Minimal Extension for Stable Time Integration of Nonviscous Flow
========

For the theory see our [Preprint](http://www3.math.tu-berlin.de/preprints/files/Preprint-11-2013.pdf).

To check invertibility of the local FEM divergence matrices of regular n-gons use the script `detOfRegTayHooCluster.py`.

The simulations are started from the file `ct_main.py`. There all parameters are set. The default values of internal parameters of the first commit are the ones used in the simulations presented in the paper.

To get a hand on, start with `python comp_timeschemes_main.py`. You will need a [Dolfin/FeNiCs](http://fenicsproject.org/) installation. You may want to install [Krypy](https://github.com/andrenarchy/krypy), but you can solve the linear systems with a different solver.

Other features:
* Paraview integration for visualization
* Convergence plots in the ipython notebook
* Storing of simulation data in json files
