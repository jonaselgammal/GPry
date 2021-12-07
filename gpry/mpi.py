"""
Defining some helpers for parallelisation.
"""
import dill
from mpi4py import MPI
import numpy as np
from numpy.random import SeedSequence, default_rng

# Use dill pickler (can seriealize more stuff, e.g. lambdas)
MPI.pickle.__init__(dill.dumps, dill.loads)

# Define some interfaces
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


def split_number_for_parallel_processes(n, n_proc=mpi_size):
    """
    Splits a number of atomic tasks `n` between the parallel processes.

    If `n` is not divisible by the number of processes, processes with lower rank are
    preferred, e.g. 5 tasks for 3 processes are assigned as [2, 2, 1].

    Returns an array with the number of tasks corresponding each process.
    """
    n_rounded_to_nproc = int(np.ceil(n / n_proc)) * n_proc
    slots = np.zeros(n_rounded_to_nproc, dtype=int)
    slots[:n] = 1
    slots = slots.reshape((int(len(slots) / n_proc), n_proc))
    return np.sum(slots, axis=0)
