__version__ = "3.0.0"

from gpry.run import Runner as Runner


def check_cobaya_installed():
    """Returns True or False depending on whether Cobaya can be imported as a package."""
    import importlib.util
    from packaging.version import Version
    from warnings import warn

    if importlib.util.find_spec("cobaya") is not None:
        from cobaya import __version__
        from gpry.cobaya import __min_cobaya_version__

        if Version(__version__) < Version(__min_cobaya_version__):
            warn(
                f"Needs min Cobaya version {__min_cobaya_version__}, "
                f"but installed one is {__version__}"
            )
            return False
        return True
    return False


if check_cobaya_installed():
    from gpry.cobaya import CobayaWrapper

    def get_cobaya_class():
        return CobayaWrapper
