#!/bin/bash
# use this script by calling `source get-upd-gh-deps.sh` in the console

# mkdir matlibplots
# cd matlibplots
# rm conv_plot_utils.py
# wget https://raw.githubusercontent.com/highlando/mat-lib-plots/master/conv_plot_utils.py
# touch __init__.py
# cd ..

mkdir dolfin_navier_scipy
cd dolfin_navier_scipy
rm *.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/dolfin_navier_scipy/data_output_utils.py
# wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/get_exp_nsmats.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/dolfin_navier_scipy/problem_setups.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/dolfin_navier_scipy/dolfin_to_sparrays.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/dolfin_navier_scipy/stokes_navier_utils.py
touch __init__.py
cd ..

mkdir krypy
cd krypy
rm linsys.py utils.py
wget https://raw.githubusercontent.com/highlando/krypy/master/krypy/linsys.py
wget https://raw.githubusercontent.com/highlando/krypy/master/krypy/utils.py
touch __init__.py
cd ..

# mkdir sadptprj_riclyap_adi
# cd sadptprj_riclyap_adi
# rm lin_alg_utils.py
# wget https://raw.githubusercontent.com/highlando/sadptprj_riclyap_adi/master/lin_alg_utils.py
# touch __init__.py
# cd ..

# import distr_control_fenics.cont_obs_utils as cou
# mkdir distr_control_fenics
# cd distr_control_fenics
# rm cont_obs_utils.py
# wget https://raw.githubusercontent.com/highlando/distr_control_fenics/master/cont_obs_utils.py
# touch __init__.py
# cd ..
