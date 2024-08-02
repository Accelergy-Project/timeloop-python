
__version__ = '0.4'

def assert_version(version: str):
    if str(version) != str(__version__):
        raise ValueError(
            f'Version {str(version)} is not supported. '
            f'Only version {__version__} is supported.'
        )
    return str(version)