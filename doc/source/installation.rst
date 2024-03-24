Installation
============

Pre-requisites
--------------

GPry requires **python** (version ≥ 3.8), the python package manager **pip**
(version ≥ 21.3.1). The required packages (cobaya, scikit-learn, numpy, scipy, dill,
mpi4py) will automatically be installed by pip.

To check if you have Python installed, type ``python --version`` in the shell, and you
should get ``Python 3.[whatever]``. Then, type ``python -m pip --version`` in the shell,
and see if you get a proper version line starting with ``pip 20.0.0 [...]``
or a higher version. If an older version is shown, please update pip with
``python -m pip install pip --upgrade``. If either Python 3 is not installed, or the
``pip`` version check produces a ``no module named pip`` error, use your system's
package manager or contact your local IT service.

.. note::

   In the following, commands to be run in the shell are displayed here with a leading
   ``$``. You do not have to type it.


Installing GPry
---------------

To install **GPry** or upgrade it to the latest release, simply type in a terminal

.. code:: bash

   $ python -m pip install gpry --upgrade

This should install GPry and all dependencies.

Of course you can also fork/clone the repo or download the source code and install that.
Be aware though, that we are actively developing the code in git so unless you want to
contribute to the code and/or are happy digging through half-finished code we advise
using the latest release.


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

In `arXiv:2305.19267 <https://arxiv.org/abs/2305.19267>`_ we propose an approach to active learning that involves sampling from the mean of the GPR surrogate using a nested sampler. The advantages of this approach are that it is highly parallelizable and more exploratory. In order to take advantage of it (see section TODO!!!) you will need to install one of the following samplers:

- `PolyChord <https://github.com/PolyChord/PolyChordLite>`_: it is the preferred option, since in combination with MPI it is very fast. To install it, follow the instructions at `https://github.com/PolyChord/PolyChordLite <https://github.com/PolyChord/PolyChordLite>`_. Try the MPI example mentioned there to make sure it works.

- `UltraNest <https://ultranest.readthedocs.io>`_: it is the default of PolyChord is not present. It is slower but easier to install: simply ``pip install ultranest``. It does take advantage of MPI parallelization.

- `nessai <https://nessai.readthedocs.io>`_: [EXPERIMENTAL SUPPORT] this sampler uses ML to increase efficiency in posteriors with non-linear degeneracies, but cannot take advantage of MPI parallelisation. To install it, follow the instructions at `https://nessai.readthedocs.io/en/latest/installation.html <https://nessai.readthedocs.io/en/latest/installation.html>`_.
