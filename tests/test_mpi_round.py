"""
Regression tests for ``gpry.mpi.round_MPI``.

Free of any ``cobaya`` import, so this module collects in a bare environment.
"""

import pytest

import gpry.mpi as mpi


@pytest.fixture
def mpi_size(request, monkeypatch):
    """Patches ``mpi.SIZE``, restoring it afterwards."""
    monkeypatch.setattr(mpi, "SIZE", request.param)
    return request.param


SIZES = [1, 2, 3, 4, 7, 8]
NS = list(range(0, 33))


@pytest.mark.parametrize("mpi_size", SIZES, indirect=True)
@pytest.mark.parametrize("n", NS)
def test_round_MPI_up(mpi_size, n):
    """``up=True`` must give the smallest multiple of SIZE that is >= n (never 0)."""
    size = mpi_size
    result = mpi.round_MPI(n, up=True, warn_rounding=False)
    assert result % size == 0
    assert result >= n
    # Smallest such multiple, except that n < SIZE is lifted to SIZE rather than to 0.
    expected = size if n < size else -(-n // size) * size
    assert result == expected
    assert result > 0


@pytest.mark.parametrize("mpi_size", SIZES, indirect=True)
@pytest.mark.parametrize("n", NS)
def test_round_MPI_down(mpi_size, n):
    """``up=False`` must give the largest multiple of SIZE that is <= n, but never 0."""
    size = mpi_size
    result = mpi.round_MPI(n, up=False, warn_rounding=False)
    assert result <= n
    if n < size:
        # Documented guard: never round n < SIZE down to 0.
        assert result == n
    else:
        assert result % size == 0
        assert result == n // size * size


@pytest.mark.parametrize("mpi_size", [4], indirect=True)
def test_round_MPI_known_values(mpi_size):
    """The concrete case that exposed the bug: SIZE = 4, n = 14."""
    assert mpi.round_MPI(14, up=True, warn_rounding=False) == 16
    assert mpi.round_MPI(14, up=False, warn_rounding=False) == 12
    # This one used to be rounded "up" to 5, i.e. below n and not a multiple of SIZE.
    assert mpi.round_MPI(6, up=True, warn_rounding=False) == 8


@pytest.mark.parametrize("mpi_size", SIZES, indirect=True)
@pytest.mark.parametrize("n", NS)
def test_round_MPI_never_rounds_below_n_when_up(mpi_size, n):
    """Rounding up must never decrease n -- the original bug did, e.g. 6 -> 5."""
    assert mpi.round_MPI(n, up=True, warn_rounding=False) >= n
