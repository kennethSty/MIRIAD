import argparse

def str2bool(value):
    """Convert string to boolean for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1', 'yes'}:
        return True
    elif value.lower() in {'false', '0', 'no'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")