__version__ = "3.0a1"

from gpry.run import Runner

from gpry.cobaya import CobayaWrapper

def get_cobaya_class():
    return CobayaWrapper
