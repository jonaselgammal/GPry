__version__ = "3.0.0"

import importlib.util

from gpry.run import Runner as Runner


def check_cobaya_installed():
    """Returns True or False depending on whether Cobaya can be imported as a package."""
    if importlib.util.find_spec("cobaya") is not None:
        return True
    return False


if check_cobaya_installed():
    from gpry.cobaya import CobayaWrapper

    def get_cobaya_class():
        return CobayaWrapper
