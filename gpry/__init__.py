__version__ = "4.0rc1"

from gpry.run import Runner as Runner


def check_cobaya_installed():
    """Returns True or False depending on whether Cobaya can be imported as a package."""
    import importlib.util
    from packaging.version import Version

    if importlib.util.find_spec("cobaya") is not None:
        from cobaya import __version__  as __cobaya_version__ # type: ignore
        from gpry.cobaya_interface import __min_cobaya_version__

        if Version(__cobaya_version__) < Version(__min_cobaya_version__):
            raise ValueError(
                f"Needs min Cobaya version {__min_cobaya_version__}, "
                f"but installed one is {__cobaya_version__}"
            )
            return False
        return True
    return False


if check_cobaya_installed():
    from gpry.cobaya_interface import CobayaWrapper

    def get_cobaya_class():
        return CobayaWrapper
