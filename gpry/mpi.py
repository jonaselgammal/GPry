# Defining some helpers for parallelisation.
import dill
from mpi4py import MPI
import numpy as np
from numpy.random import SeedSequence, default_rng, Generator

# Use dill pickler (can seriealize more stuff, e.g. lambdas)
MPI.pickle.__init__(dill.dumps, dill.loads)

# Define some interfaces
comm = MPI.COMM_WORLD
SIZE = comm.Get_size()
RANK = comm.Get_rank()
is_main_process = not bool(RANK)
multiple_processes = SIZE > 1


def get_random_state(seed=None):
    """
    Generates seed sequences for processes running in parallel.

    Parameters
    ----------

    seed : int or numpy seed, or numpy.random.Generator, optional (default=None)
        A random seed to initialise a Generator, or a Generator to be used directly.
        If none is provided a random one will be drawn.
    """
    if isinstance(seed, Generator):
        return seed
    if is_main_process:
        ss = SeedSequence(seed)
        child_seeds = ss.spawn(SIZE)
    ss = comm.scatter(child_seeds if is_main_process else None)
    return default_rng(ss)


def split_number_for_parallel_processes(n, n_proc=SIZE):
    """
    Splits a number of atomic tasks `n` between the parallel processes.

    If `n` is not divisible by the number of processes, processes with lower rank are
    preferred, e.g. 5 tasks for 3 processes are assigned as [2, 2, 1].

    Parameters
    ----------
    n : int
        The number of atomic tasks
    n_proc : int, optional (default=number of MPI comm's)
        The number of processes to divide the tasks between

    Returns
    -------
    An array with the number of tasks corresponding each process.
    """
    n_rounded_to_nproc = int(np.ceil(n / n_proc)) * n_proc
    slots = np.zeros(n_rounded_to_nproc, dtype=int)
    slots[:n] = 1
    slots = slots.reshape((int(len(slots) / n_proc), n_proc))
    return np.sum(slots, axis=0)


def multi_gather_array(arrs):
    """
    Gathers (possibly a list of) arrays from all processes into the main process.

    NB: mpi-gather guarantees rank order is preserved.

    Parameters
    ----------
    arrs : array-like
        The arrays to gather

    Returns
    -------
    The gathered array(s) from all processes
    """
    if not isinstance(arrs, (list, tuple)):
        arrs = [arrs]
    Nobj = len(arrs)
    if multiple_processes:
        all_arrs = comm.gather(arrs)
        if is_main_process:
            arrs = [np.concatenate([all_arrs[r][i]
                                   for r in range(SIZE)]) for i in range(Nobj)]
            return arrs
        else:
            return [None for i in range(Nobj)]
    else:
        return arrs


def sync_processes():
    """
    Makes all processes halt here until all have reached this point.
    """
    comm.barrier()


def share_attr(instance, attr_name, root=0):
    """Broadcasts ``attr`` of ``instance`` from process of rank ``root``."""
    if not multiple_processes:
        return
    setattr(instance, attr_name,
            comm.bcast(getattr(instance, attr_name, None), root=root))
