#! /bin/bash

cd submodules/simple-knn
pip install -e .

cd ../diff-surfel-rasterization
rm dist build diff_surfel_rasterization.egg-info -rfd
pip install -e .

cd ../visibility_estimator
cd src && ./compile_ptx.sh && cd ..
rm dist build visibility_estimator.egg-info -rfd
pip install -e .

cd ../one_bounce_estimator
cd src && ./compile_ptx.sh && cd ..
rm dist build one_bounce_estimator.egg-info -rfd
pip install -e .

cd ../radiosity_solver
cd src && ./compile_ptx.sh && cd ..
rm dist build radiosity_solver_aux.egg-info -rfd
pip install -e .