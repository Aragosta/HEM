import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.modules.loss

def str2bool(value):
    """Parse a boolean flag from the command line.

    ``argparse`` with ``type=bool`` is a trap -- ``bool("False")`` is ``True`` --
    and the released config went further and gave boolean options integer
    defaults, so ``--train_curv True`` ran ``int("True")`` and crashed. Every
    example script in ``example/`` passes exactly that, so none of them start.
    """
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("true", "t", "yes", "y", "1"):
        return True
    if lowered in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            elif isinstance(default, bool):
                parser.add_argument(f"--{param}", type=str2bool, nargs="?", const=True,
                                    default=default, help=description)
            else:
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser

