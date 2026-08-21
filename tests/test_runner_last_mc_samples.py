"""
Regression test for ``Runner.last_mc_samples(as_pandas=True)``.

Free of any ``cobaya`` import, so this module collects in a bare environment: the Runner
is built with ``__new__`` and only the few attributes this method touches are set.
"""

from types import SimpleNamespace

import numpy as np

from gpry.run import Runner


def make_runner():
    runner = Runner.__new__(Runner)  # bypass the heavy __init__
    runner.truth = SimpleNamespace(params=["a", "b"], labels=["a", "b"])
    runner._last_mc_samples = {
        "w": None,
        "X": np.arange(6.0).reshape(3, 2),
        "logpost": np.array([-1.0, -2.0, -3.0]),
    }
    return runner


def test_as_pandas_does_not_consume_the_stored_samples():
    """
    With ``copy=False`` the method used to hand out the runner's own dict and then pop
    "X" out of it, destroying the stored samples: a second call raised ``KeyError: 'X'``.
    """
    runner = make_runner()
    df = runner.last_mc_samples(copy=False, as_pandas=True)
    assert set(runner._last_mc_samples) == {"w", "X", "logpost"}
    assert runner._last_mc_samples["w"] is None
    assert set(df.columns) == {"w", "a", "b", "logpost"}
    # Idempotent: calling again must give the same thing, not raise.
    df2 = runner.last_mc_samples(copy=False, as_pandas=True)
    assert list(df2.columns) == list(df.columns)
    assert np.array_equal(df2["a"].to_numpy(), df["a"].to_numpy())


def test_as_pandas_columns_match_the_stored_arrays():
    runner = make_runner()
    df = runner.last_mc_samples(copy=False, as_pandas=True)
    X = runner._last_mc_samples["X"]
    assert np.array_equal(df["a"].to_numpy(), X[:, 0])
    assert np.array_equal(df["b"].to_numpy(), X[:, 1])
    assert np.array_equal(df["w"].to_numpy(), np.ones(len(X)))
