# Defining some helpers for parallelisation.
import dill
import numpy as np
from warnings import warn
from numpy.random import SeedSequence, default_rng, Generator

try:
    from mpi4py import MPI

    # Use dill pickler (can seriealize more stuff, e.g. lambdas)
    MPI.pickle.__init__(dill.dumps, dill.loads)
    # Define some interfaces
    comm = MPI.COMM_WORLD
    SIZE = comm.Get_size()
    RANK = comm.Get_rank()
    is_main_process = not bool(RANK)
    multiple_processes = SIZE > 1
except ImportError:
    warn(
        "mpi4py could not be imported. "
        "It is optional but recommended for faster running in parallel."
    )
    # Define dummy interfaces
    comm = None
    SIZE = 1
    RANK = 0
    is_main_process = True
    multiple_processes = False


def get_random_generator(seed=None):
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
    if not multiple_processes:
        return default_rng(child_seeds[0])
    ss = comm.scatter(child_seeds if is_main_process else None)
    return default_rng(ss)


def bcast(args, root=0):
    """
    Wrapper for MPI.comm.bcast, that works if MPI not present.
    """
    if multiple_processes:
        return comm.bcast(args, root=root)
    return args


def gather(args, root=0):
    """
    Wrapper for MPI.comm.gather, that works if MPI not present.
    """
    if multiple_processes:
        return comm.gather(args, root=root)
    return [args]


def allgather(args):
    """
    Wrapper for MPI.allgather, that works if MPI not present.
    """
    if multiple_processes:
        return comm.allgather(args)
    return [args]


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


def step_split(values):
    """
    Broadcasts from rank=0 and splits array between MPI processes, using mpi.size as step.

    If starting from sorted arrays, it preserves "computational scaling" among
    processes, but producing similar-in-content partial arrays.
    """
    if not multiple_processes:
        return values
    values = comm.bcast(values)
    return values[RANK::SIZE]


def merge_step_split(values):
    """
    Gather step-split (with ``::mpi.SIZE``) arrays and returns the merged set for the
    rank=0 process (``None`` for the rest).
    """
    if not multiple_processes:
        return values
    values_step = comm.gather(values)
    if is_main_process:
        values_merged = np.zeros(sum(len(v) for v in values_step))
        for i, v in enumerate(values_step):
            values_merged[i::SIZE] = v
        return values_merged
    return None


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
    if not multiple_processes:
        return arrs
    all_arrs = comm.gather(arrs)
    if is_main_process:
        arrs = [
            np.concatenate([all_arrs[r][i] for r in range(SIZE)]) for i in range(Nobj)
        ]
        return arrs
    else:
        return [None for i in range(Nobj)]


def sync_processes():
    """
    Makes all processes halt here until all have reached this point.
    """
    if not multiple_processes:
        return
    comm.barrier()


def share_attr(instance, attr_name, root=0):
    """Broadcasts ``attr`` of ``instance`` from process of rank ``root``."""
    if not multiple_processes:
        return
    setattr(
        instance, attr_name, comm.bcast(getattr(instance, attr_name, None), root=root)
    )


def compute_y_parallel(gpr, X, y, sigma_y, ensure_sigma_y=False):
    """
    Computes the GPR mean (and std if `do_sigma_y=True`) in parallel.

    Returns the resulting `(y, sigma_y)` arrays (computed or given) for rank 0, and
    ``None`` otherwise.
    """
    if multiple_processes:
        y = comm.bcast(y)
    if y is None:  # assume sigma_y is also None
        this_X = step_split(X)
        if len(this_X) > 0:
            if ensure_sigma_y:
                this_y, this_sigma_y = gpr.predict(
                    this_X, return_std=True, validate=False
                )
            else:
                this_y = gpr.predict(this_X, return_std=False, validate=False)
        else:
            this_y = np.array([], dtype=float)
            this_sigma_y = np.array([], dtype=float) if ensure_sigma_y else None
        return (
            merge_step_split(this_y),
            merge_step_split(this_sigma_y) if ensure_sigma_y else None,
        )
    sigma_y = comm.bcast(sigma_y)
    if sigma_y is None and ensure_sigma_y:
        this_X = step_split(X)
        if len(this_X) > 0:
            this_sigma_y = gpr.predict_std(this_X, validate=False)
        else:
            this_sigma_y = np.array([], dtype=float)
        return (
            y if is_main_process else None,
            merge_step_split(this_sigma_y),
        )
    return (y, sigma_y) if is_main_process else (None, None)


def round_MPI(n, up=True, warn=True, name=None):
    """
    Rounds up (default) or down the value of the first argument to an integer multiple
    of the number of MPI processes.

    Does never round down ``n < mpi.SIZE`` to 0.
    """
    if n < SIZE and up:
        new_n = SIZE
    elif n > SIZE and n % SIZE:
        new_n = n // SIZE * (SIZE + (1 if up else 0))
    else:
        new_n = n
    if new_n != n and warn:
        warn(
            f"{name or 'Variable'} has been rounded {'up' if up else 'down'} from {n} "
            f"to {new_n} to better exploit parallelization."
        )
    return new_n
