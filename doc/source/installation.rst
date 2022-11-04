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
