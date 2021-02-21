# Master Thesis

Simulation-based inference of parameteres in mechanistic models.

## Environment

**Install Anaconda**

See [anaconda.com](https://www.anaconda.com/products/individual).

**Create environment**

    $ conda env create --file environment.yml

**Activate environment**

    $ conda activate master

**Deactivate environment**

    $ conda deactivate

**To remove environment**

    $ conda remove --name master --all

**To verify that the environment was removed**

    $ conda info --envs

## Installation

`cd` into root directory

**Development Install**

    $ pip install --editable .

**Install Package**

    $ pip install .

## Tests

`cd` into root directory

**Run tests:**

    $ pytest -v -p no:warnings

## Structure

The [latex folder](https://github.com/nicolossus/Master-thesis/tree/master/latex) contains the LaTeX source for building the thesis, as well as [figures](https://github.com/nicolossus/Master-thesis/tree/master/latex/figures) and [tables](https://github.com/nicolossus/Master-thesis/tree/master/tables) generated in the analyses.

The [notebooks folder](https://github.com/nicolossus/Master-thesis/tree/master/notebooks) contains Jupyter notebooks used in the analyses. For details, see the [notebooks readme](https://github.com/nicolossus/Master-thesis/blob/master/notebooks/README.md).

The [report folder](https://github.com/nicolossus/Master-thesis/tree/master/report) contains the report rendered to PDF from the LaTeX source.

The [resources folder](https://github.com/nicolossus/Master-thesis/tree/master/resources) contains project resources such as raw data to be analysed.

The [src folder](https://github.com/nicolossus/Master-thesis/tree/master/src) contains the source code. For details, see the [src readme](https://github.com/nicolossus/Master-thesis/blob/master/src/README.md).

The [test folder](https://github.com/nicolossus/Master-thesis/tree/master/test) contains procedures for unit testing and [benchmarking](https://github.com/nicolossus/Master-thesis/tree/master/test/benchmark) the source code developed for the project.
