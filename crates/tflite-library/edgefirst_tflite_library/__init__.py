# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

"""Pre-built TensorFlow Lite C API shared library.

Provides a single function :func:`library_path` that returns the absolute path
to the shipped ``libtensorflowlite_c`` shared library for the current platform.

Usage::

    from edgefirst_tflite_library import library_path

    # Pass to edgefirst-tflite Interpreter
    from edgefirst_tflite import Interpreter
    interpreter = Interpreter(model_path="model.tflite", library_path=library_path())
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

__all__ = ["library_path", "__version__"]
__version__ = "2.19.0"

_PACKAGE_DIR = Path(__file__).parent

#: Map of sys.platform values to shared library filenames.
_LIB_NAMES = {
    "win32": "tensorflowlite_c.dll",
    "darwin": "libtensorflowlite_c.dylib",
}
_DEFAULT_LIB = "libtensorflowlite_c.so"


def library_path() -> str:
    """Return the absolute path to the bundled TFLite C API shared library.

    Returns:
        Absolute filesystem path to the shared library.

    Raises:
        FileNotFoundError: If no library is found for the current platform.
            This typically means the wheel was built for a different platform.
            Use ``Interpreter(library_path=...)`` with an explicit path as a
            workaround.
    """
    name = _LIB_NAMES.get(sys.platform, _DEFAULT_LIB)
    path = _PACKAGE_DIR / name

    if not path.exists():
        raise FileNotFoundError(
            f"TFLite library not found at {path}. "
            f"This wheel may not support your platform "
            f"({sys.platform}/{platform.machine()}). "
            f"Use Interpreter(library_path=...) to specify a custom path."
        )

    return str(path)
