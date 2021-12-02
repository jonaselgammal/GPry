"""
Defining some helpers for parallelisation.
"""

from mpi4py import MPI
from numpy.random import SeedSequence, default_rng

mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()
is_main_process = not bool(mpi_rank)


def get_random_state(seed=None):
    """
    Generates seed sequences for processes running in parallel.
    """
    if is_main_process:
        ss = SeedSequence(seed)
        child_seeds = ss.spawn(mpi_size)
    ss = mpi_comm.scatter(child_seeds if is_main_process else None)
    return default_rng(ss)
