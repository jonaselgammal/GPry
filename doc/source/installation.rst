Installation
============

.. note::

   In the following, commands to be run in the shell are displayed here with a leading
   ``$``. You do not have to type it.

   Commands starting with ``python`` may need to be run with ``python3`` instead, depending on your system.


Pre-requisites
--------------
GPry requires **Python** (version ≥ 3.8, check the output of ``$ python --version``) and an up to date version of the Python package manager **pip** (to install: ``$ python -m ensurepip``; to update: ``$ python -m pip install pip --upgrade``). Most of the requisites will automatically be installed by pip.


Installing GPry
---------------

To install **GPry** or upgrade it to the latest release, simply type in a terminal

.. code:: bash

   $ python -m pip install gpry --upgrade

This should install GPry and all dependencies.

Of course you can also fork/clone the repo or download the source code and install that.
Be aware though, that we are actively developing the code in git, though most of the development occurs in branches.


Configuring MPI
---------------

Though optional, in order to make GPry as fast as possible, you will need a working MPI installation and the necessary Python bindings.

In order to install MPI in a Debian-like system, do:

.. code:: bash

   $ sudo apt install openmpi-bin libopenmpi-dev

To get the MPI bindings, install the ``mpi4py`` package with ``pip``.

If successful, you should be able to run the following command and get ``MPI is working`` as an output:

.. code:: bash

   $ mpirun -n 2 python -c "exec('from mpi4py import MPI\nif MPI.COMM_WORLD.Get_rank() == 1: print(\"MPI is working\")')"


Installing Nested Samplers
--------------------------

In order to use the highly parallelizable and more exploratory NORA acquisition engine [TODO: add reference], you will need to install one of the following nested samplers:

- `PolyChord <https://github.com/PolyChord/PolyChordLite>`_: it is the preferred option, since in combination with MPI it is very fast. To install it, follow the instructions at `https://github.com/PolyChord/PolyChordLite <https://github.com/PolyChord/PolyChordLite>`_. Try the MPI example mentioned there to make sure it works.

- `UltraNest <https://ultranest.readthedocs.io>`_: it is the default of PolyChord is not present. It is slower but easier to install: simply ``pip install ultranest``. It does take advantage of MPI parallelization.

- `nessai <https://nessai.readthedocs.io>`_: [EXPERIMENTAL SUPPORT] this sampler uses ML to increase efficiency in posteriors with non-linear degeneracies, but cannot take advantage of MPI parallelisation. To install it, follow the instructions at `https://nessai.readthedocs.io/en/latest/installation.html <https://nessai.readthedocs.io/en/latest/installation.html>`_.

To check that any of the three have been correctly installed, check that the following command does not throw an error: ``$ python -c "import SAMPLER"``, replacing ``SAMPLER`` with ``pypolychord``, ``ultranest``, ``nessai``.
