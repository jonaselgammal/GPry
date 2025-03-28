__version__ = "3.0a1"

def check_cobaya_installed():
    """Returns True or False depending on whether Cobaya can be imported as a package."""
    try:
        import cobaya
    except ModuleNotFoundError:
        return False
    return True

from gpry.run import Runner

if check_cobaya_installed():
    from gpry.cobaya import CobayaWrapper

    def get_cobaya_class():
        return CobayaWrapper
