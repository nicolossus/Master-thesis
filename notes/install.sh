




conda create --name test-env2 -c conda-forge python=3.8 numpy scipy matplotlib numba pandas seaborn jupyter cython pytest nose scikit-learn nest-simulator=*=mpi_openmpi* lfpy pymc3 glmnet

conda install pytorch torchvision torchaudio -c pytorch

pip install tensorflow sbi abcpy


# run: conda env create --file environment.yml
name: test-env
dependencies:
- python>=3.5
- anaconda
- pip
- numpy=1.13.3  # pin version for conda
- pip:
  # works for regular pip packages
  - docx
  - gooey
  - matplotlib==2.0.0  # pin version for pip


  name: master
  channels:
    - conda-forge
    - defaults
  dependencies:
    - python>3.8
    - notebook>5.2
    - matplotlib
    - numpy
    - scikit-learn
    - jupyter_nbextensions_configurator
    - gdal
    - jupyter_contrib_nbextensions
    - markdown
    - pandas
    - tensorflow
