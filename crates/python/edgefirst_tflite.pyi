# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

"""Python API for TFLite inference with EdgeFirst extensions.

This package provides an ``Interpreter`` class for running TFLite models
with optional hardware acceleration via delegates, DMA-BUF zero-copy
inference, and NPU-accelerated camera preprocessing.

Tensor indices
--------------
``get_input_tensor``, ``get_output_tensor``, and ``set_tensor`` use indices
that are **relative** within input or output tensors (0-based), not global
model-level tensor indices.  The ``"index"`` field in the dicts returned by
``get_input_details()`` / ``get_output_details()`` matches these relative
indices for consistent round-trip access.

``tensor()`` uses a combined index space: ``[0, input_count)`` addresses
input tensors, ``[input_count, input_count + output_count)`` addresses
output tensors.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

try:
    from typing import deprecated
except ImportError:
    from typing_extensions import deprecated

import numpy as np
import numpy.typing as npt

__version__: str

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TfLiteError(RuntimeError):
    """Base exception for all TFLite errors."""

    ...

class LibraryError(TfLiteError):
    """TFLite shared library could not be loaded or is missing symbols."""

    ...

class DelegateError(TfLiteError):
    """A delegate returned an error status."""

    ...

class InvalidArgumentError(TfLiteError):
    """An invalid argument was passed to the API."""

    ...

# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

class Interpreter:
    """TFLite model interpreter with EdgeFirst extensions.

    Loads a TFLite model and prepares it for inference.  Tensors are
    allocated during construction; call ``allocate_tensors()`` again only
    after ``resize_tensor_input()``.

    Example::

        interp = Interpreter(model_path="model.tflite")
        interp.set_tensor(0, input_data)
        interp.invoke()
        output = interp.get_output_tensor(0)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_content: bytes | None = None,
        num_threads: int | None = None,
        experimental_delegates: list[Delegate] | None = None,
        *,
        library_path: str | Path | None = None,
    ) -> None:
        """Create a new Interpreter.

        Args:
            model_path: Path to a ``.tflite`` model file.
            model_content: Raw model bytes (alternative to ``model_path``).
            num_threads: Number of CPU threads for inference (``None`` = auto).
            experimental_delegates: List of hardware acceleration delegates.
            library_path: Explicit path to ``libtensorflowlite_c.so``
                (EdgeFirst extension; omit to auto-discover).

        Raises:
            InvalidArgumentError: If neither ``model_path`` nor
                ``model_content`` is provided.
            LibraryError: If the TFLite library cannot be loaded.
        """
        ...

    def allocate_tensors(self) -> None:
        """Re-allocate tensors.  Required after ``resize_tensor_input()``.

        Invalidates any zero-copy tensor views obtained via ``tensor()``.
        Tensors are also allocated during ``__init__``, so this only needs
        to be called explicitly after resizing inputs.
        """
        ...

    def resize_tensor_input(self, input_index: int, tensor_size: list[int]) -> None:
        """Resize an input tensor's dimensions.

        You **must** call ``allocate_tensors()`` after resizing and before
        calling ``invoke()``.  Immediately invalidates any existing
        zero-copy tensor views from ``tensor()``.

        Args:
            input_index: 0-based index of the input tensor to resize.
            tensor_size: New shape as a list of dimension sizes.
        """
        ...

    def invoke(self) -> None:
        """Run model inference.

        Input tensors must be populated (via ``set_tensor``) before calling
        this method.  After ``invoke``, read results with
        ``get_output_tensor``.
        """
        ...

    def get_input_details(self) -> list[dict[str, object]]:
        """Return metadata for each input tensor.

        Each dict contains: ``name`` (str), ``index`` (int), ``shape``
        (numpy int32 array), ``dtype`` (numpy dtype), ``quantization``
        (scale, zero_point), and ``quantization_parameters`` (dict with
        ``scales``, ``zero_points``, ``quantized_dimension``).
        """
        ...

    def get_output_details(self) -> list[dict[str, object]]:
        """Return metadata for each output tensor.

        Same format as ``get_input_details()``.
        """
        ...

    def get_input_tensor(self, input_index: int) -> npt.NDArray[np.generic]:
        """Return a copy of an input tensor's data as a numpy array.

        Args:
            input_index: 0-based index within the model's input tensors.
        """
        ...

    def get_output_tensor(self, output_index: int) -> npt.NDArray[np.generic]:
        """Return a copy of an output tensor's data as a numpy array.

        Args:
            output_index: 0-based index within the model's output tensors.
        """
        ...

    def set_tensor(self, input_index: int, value: npt.NDArray[np.generic]) -> None:
        """Copy numpy array data into an input tensor.

        Args:
            input_index: 0-based index within the model's input tensors.
            value: Numpy array whose dtype and shape must match the tensor.
        """
        ...

    def tensor(self, tensor_index: int) -> Callable[[], npt.NDArray[np.generic]]:
        """Return a callable that yields a zero-copy numpy view of tensor data.

        The returned callable produces a numpy array that shares memory with
        the TFLite C-allocated buffer and reflects the latest inference
        results after each ``invoke()`` call.

        The callable is invalidated by ``allocate_tensors()`` or
        ``resize_tensor_input()`` — call ``tensor()`` again to get a fresh
        one.

        Index mapping: ``[0, input_count)`` addresses input tensors,
        ``[input_count, input_count + output_count)`` addresses outputs.

        Args:
            tensor_index: Combined input/output tensor index.

        Returns:
            A callable that returns a zero-copy numpy array view.
        """
        ...

    @property
    def input_count(self) -> int:
        """Number of input tensors."""
        ...

    @property
    def output_count(self) -> int:
        """Number of output tensors."""
        ...

    def delegate(self, index: int = 0) -> DelegateRef | None:
        """Access a delegate owned by this interpreter.

        Args:
            index: Delegate index (0-based, in order they were added).

        Returns:
            A ``DelegateRef`` for querying capabilities and accessing
            extensions, or ``None`` if no delegate exists at that index.
        """
        ...

    def dmabuf(self, delegate_index: int = 0) -> DmaBuf | None:
        """Get a DMA-BUF interface for a delegate's zero-copy extensions.

        Shorthand for ``interp.delegate(i).dmabuf()``.

        Returns:
            A ``DmaBuf`` interface, or ``None`` if the delegate does not
            support DMA-BUF.
        """
        ...

    def camera_adaptor(self, delegate_index: int = 0) -> CameraAdaptor | None:
        """Get a CameraAdaptor interface for NPU preprocessing.

        Shorthand for ``interp.delegate(i).camera_adaptor()``.

        Returns:
            A ``CameraAdaptor`` interface, or ``None`` if the delegate does
            not support CameraAdaptor.
        """
        ...

    def get_metadata(self) -> Metadata | None:
        """Extract model metadata from the TFLite FlatBuffer.

        Returns:
            A ``Metadata`` object, or ``None`` if the model has no metadata.
        """
        ...

# ---------------------------------------------------------------------------
# Delegates
# ---------------------------------------------------------------------------

class Delegate:
    """An external TFLite delegate for hardware acceleration.

    Created via ``load_delegate()``.  Passed to the ``Interpreter``
    constructor via ``experimental_delegates``.  Consumed on use — cannot
    be reused across multiple interpreters.
    """

    @property
    def has_dmabuf(self) -> bool:
        """Whether this delegate supports DMA-BUF zero-copy."""
        ...

    @property
    def has_camera_adaptor(self) -> bool:
        """Whether this delegate supports CameraAdaptor."""
        ...

class DelegateRef:
    """Borrowed reference to a delegate owned by an Interpreter.

    Obtained via ``Interpreter.delegate(index)``.  Provides capability
    queries and access to delegate extensions (DmaBuf, CameraAdaptor).
    """

    @property
    def index(self) -> int:
        """The delegate index within the interpreter."""
        ...

    @property
    def has_dmabuf(self) -> bool:
        """Whether this delegate supports DMA-BUF zero-copy."""
        ...

    @property
    def has_camera_adaptor(self) -> bool:
        """Whether this delegate supports CameraAdaptor."""
        ...

    def dmabuf(self) -> DmaBuf | None:
        """Get a DmaBuf interface for this delegate.

        Returns:
            A ``DmaBuf`` interface, or ``None`` if not supported.
        """
        ...

    def camera_adaptor(self) -> CameraAdaptor | None:
        """Get a CameraAdaptor interface for this delegate.

        Returns:
            A ``CameraAdaptor`` interface, or ``None`` if not supported.
        """
        ...

def load_delegate(
    library: str | Path,
    options: dict[str, str] | None = None,
) -> Delegate:
    """Load an external delegate from a shared library.

    Args:
        library: Path to the delegate ``.so`` (e.g., ``libvx_delegate.so``).
        options: Optional key-value configuration options.

    Returns:
        A ``Delegate`` to pass to ``Interpreter(experimental_delegates=[...])``.

    Raises:
        LibraryError: If the shared library cannot be loaded.
    """
    ...

def xnnpack_delegate(num_threads: int = 0) -> Delegate:
    """Create an XNNPACK delegate for CPU-accelerated inference.

    XNNPACK optimises floating-point and quantised operations on ARM and
    x86 CPUs using SIMD instructions (NEON on ARM, AVX/SSE on x86).

    Args:
        num_threads: XNNPACK threadpool size. Use 1 for single-threaded,
            higher values for parallelism, or 0 to let XNNPACK choose.

    Returns:
        A ``Delegate`` to pass to ``Interpreter(experimental_delegates=[...])``.

    Raises:
        InvalidArgumentError: If the TFLite library was not compiled with
            XNNPACK support (``-DTFLITE_ENABLE_XNNPACK=ON``).
        LibraryError: If no TFLite shared library can be found.

    Example::

        delegate = xnnpack_delegate(num_threads=4)
        interp = Interpreter(
            model_path="model.tflite",
            experimental_delegates=[delegate],
        )
    """
    ...

# ---------------------------------------------------------------------------
# DMA-BUF
# ---------------------------------------------------------------------------

class DmaBuf:
    """DMA-BUF zero-copy interface for TFLite delegates.

    Uses the HAL Delegate DMA-BUF API as the primary backend, with
    legacy VxDelegate methods available as deprecated fallbacks.

    Obtained via ``interp.dmabuf()`` or ``delegate_ref.dmabuf()``.
    """

    # --- Primary API (HAL Delegate DMA-BUF) ---

    def is_supported(self) -> bool:
        """Check if DMA-BUF zero-copy is supported by the hardware."""
        ...

    def tensor_info(self, tensor_index: int) -> dict[str, object]:
        """Get DMA-BUF tensor information for a given tensor index.

        Returns a dict with keys:
            - ``fd`` (int): DMA-BUF file descriptor (borrowed, do NOT close).
            - ``size`` (int): Buffer size in bytes.
            - ``offset`` (int): Byte offset within the DMA-BUF.
            - ``shape`` (list[int]): Tensor dimensions.
            - ``dtype`` (str): Element data type (e.g., ``"u8"``, ``"f32"``).

        Requires the HAL Delegate DMA-BUF API.
        """
        ...

    def sync_for_device(self, tensor_index: int) -> None:
        """Sync tensor buffer for device (NPU) access by tensor index.

        Flushes CPU caches so the device can read the buffer contents.
        """
        ...

    def sync_for_cpu(self, tensor_index: int) -> None:
        """Sync tensor buffer for CPU access by tensor index.

        Invalidates CPU caches so the CPU sees device-written data.
        """
        ...

    # --- Legacy VxDelegate API (deprecated) ---

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def register(self, fd: int, size: int, sync_mode: str = "none") -> int:
        """Register an externally-allocated DMA-BUF (import mode).

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.

        Args:
            fd: DMA-BUF file descriptor.
            size: Buffer size in bytes.
            sync_mode: Cache sync mode (``"none"``, ``"read"``, ``"write"``,
                ``"readwrite"``).

        Returns:
            An opaque buffer handle for use with other DmaBuf methods.
        """
        ...

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def unregister(self, handle: int) -> None:
        """Unregister a previously registered DMA-BUF.

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.
        """
        ...

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def request(
        self,
        tensor_index: int,
        ownership: str = "client",
        size: int = 0,
    ) -> tuple[int, dict[str, object]]:
        """Request the delegate to allocate a DMA-BUF (export mode).

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.

        Args:
            tensor_index: Tensor to allocate a buffer for.
            ownership: ``"client"`` (app owns) or ``"delegate"`` (delegate owns).
            size: Requested buffer size (0 = auto from tensor).

        Returns:
            Tuple of (handle, desc_dict) where desc_dict contains ``fd``,
            ``size``, and ``map_ptr`` keys.
        """
        ...

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def release(self, handle: int) -> None:
        """Release a delegate-allocated DMA-BUF.

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.
        """
        ...

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def bind_to_tensor(self, handle: int, tensor_index: int) -> None:
        """Bind a DMA-BUF to a tensor for zero-copy inference.

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.
        """
        ...

    @deprecated("VxDelegate-specific, use tensor_info() instead; will be removed in a future release")
    def fd(self, handle: int) -> int:
        """Get the file descriptor for a buffer handle.

        .. deprecated::
            VxDelegate-specific, use ``tensor_info()`` instead.
        """
        ...

    @deprecated("VxDelegate-specific, use sync_for_cpu() instead; will be removed in a future release")
    def begin_cpu_access(self, handle: int, mode: str = "read") -> None:
        """Begin CPU access to a DMA-BUF (ensure cache coherency).

        .. deprecated::
            VxDelegate-specific, use ``sync_for_cpu()`` instead.
        """
        ...

    @deprecated("VxDelegate-specific, use sync_for_device() instead; will be removed in a future release")
    def end_cpu_access(self, handle: int, mode: str = "read") -> None:
        """End CPU access to a DMA-BUF (flush caches).

        .. deprecated::
            VxDelegate-specific, use ``sync_for_device()`` instead.
        """
        ...

    @deprecated("VxDelegate-specific, use sync_for_device(tensor_index) instead; will be removed in a future release")
    def sync_for_device_by_handle(self, handle: int) -> None:
        """Sync buffer for device (NPU) access by buffer handle.

        .. deprecated::
            VxDelegate-specific, use ``sync_for_device(tensor_index)`` instead.
        """
        ...

    @deprecated("VxDelegate-specific, use sync_for_cpu(tensor_index) instead; will be removed in a future release")
    def sync_for_cpu_by_handle(self, handle: int) -> None:
        """Sync buffer for CPU access by buffer handle.

        .. deprecated::
            VxDelegate-specific, use ``sync_for_cpu(tensor_index)`` instead.
        """
        ...

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def set_active(self, tensor_index: int, handle: int) -> None:
        """Set the active DMA-BUF for a tensor (buffer pool cycling).

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.
        """
        ...

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def active_buffer(self, tensor_index: int) -> int | None:
        """Get the currently active buffer handle for a tensor.

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.
        """
        ...

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def invalidate_graph(self) -> None:
        """Invalidate the compiled graph (forces recompilation on next invoke).

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.
        """
        ...

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def is_graph_compiled(self) -> bool:
        """Check if the graph has been compiled.

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.
        """
        ...

# ---------------------------------------------------------------------------
# CameraAdaptor
# ---------------------------------------------------------------------------

class CameraAdaptor:
    """NPU-accelerated camera format conversion.

    Uses the HAL Delegate Camera Adaptor API as the primary backend,
    with legacy VxDelegate methods available as deprecated fallbacks.

    Obtained via ``interp.camera_adaptor()`` or ``delegate_ref.camera_adaptor()``.
    """

    # --- Primary API (HAL Delegate Camera Adaptor) ---

    def is_format_supported(self, format: str) -> bool:
        """Check if a format string is supported by this delegate.

        Uses the HAL API when available, falling back to VxDelegate.

        Args:
            format: Format string (e.g., ``"rgba"``, ``"bgra"``).
        """
        ...

    def format_info(self, format: str) -> dict[str, object]:
        """Get format information for a camera format string.

        Returns a dict with keys:
            - ``input_channels`` (int): Number of input channels.
            - ``output_channels`` (int): Number of output channels.
            - ``fourcc`` (str): V4L2 FourCC code string.

        Args:
            format: Format string (e.g., ``"rgba"``).
        """
        ...

    # --- Legacy VxDelegate API (deprecated) ---

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def set_format(self, tensor_index: int, format: str) -> None:
        """Set the camera format for an input tensor.

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.

        Args:
            tensor_index: Input tensor index.
            format: Format string (e.g., ``"rgba"``, ``"bgra"``).
        """
        ...

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def set_format_ex(
        self,
        tensor_index: int,
        format: str,
        width: int,
        height: int,
        letterbox: bool = False,
        letterbox_color: int = 0,
    ) -> None:
        """Set camera format with resize and letterbox options.

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.

        Args:
            tensor_index: Input tensor index.
            format: Format string.
            width: Target width for resize.
            height: Target height for resize.
            letterbox: Whether to apply letterboxing.
            letterbox_color: Fill color for letterbox padding.
        """
        ...

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def set_formats(
        self,
        tensor_index: int,
        camera_format: str,
        model_format: str,
    ) -> None:
        """Set explicit camera and model formats.

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.

        Args:
            tensor_index: Input tensor index.
            camera_format: Input format from camera (e.g., ``"rgba"``).
            model_format: Expected model format (e.g., ``"rgb"``).
        """
        ...

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def set_fourcc(self, tensor_index: int, fourcc: int) -> None:
        """Set camera format using a V4L2 FourCC code.

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.

        Args:
            tensor_index: Input tensor index.
            fourcc: V4L2 FourCC code as integer.
        """
        ...

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def format(self, tensor_index: int) -> str | None:
        """Get the current format for an input tensor.

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.
        """
        ...

    @deprecated("VxDelegate-specific, use is_format_supported() instead; will be removed in a future release")
    def is_supported(self, format: str) -> bool:
        """Check if a format string is supported.

        .. deprecated::
            VxDelegate-specific, use ``is_format_supported()`` instead.
        """
        ...

    @deprecated("VxDelegate-specific, use format_info() instead; will be removed in a future release")
    def input_channels(self, format: str) -> int:
        """Get the number of input channels for a format.

        .. deprecated::
            VxDelegate-specific, use ``format_info()`` instead.
        """
        ...

    @deprecated("VxDelegate-specific, use format_info() instead; will be removed in a future release")
    def output_channels(self, format: str) -> int:
        """Get the number of output channels for a format.

        .. deprecated::
            VxDelegate-specific, use ``format_info()`` instead.
        """
        ...

    @deprecated("VxDelegate-specific, use format_info() instead; will be removed in a future release")
    def fourcc(self, format: str) -> str | None:
        """Get the FourCC code string for a format.

        .. deprecated::
            VxDelegate-specific, use ``format_info()`` instead.
        """
        ...

    @deprecated("VxDelegate-specific, will be removed in a future release")
    def from_fourcc(self, fourcc: str) -> str | None:
        """Convert a FourCC code to a format string.

        .. deprecated::
            VxDelegate-specific, will be removed in a future release.
        """
        ...

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class Metadata:
    """Extracted metadata from a TFLite model file.

    Obtained via ``Interpreter.get_metadata()``.  All fields are ``None``
    if the model does not contain the corresponding metadata.
    """

    @property
    def name(self) -> str | None:
        """Model name."""
        ...

    @property
    def version(self) -> str | None:
        """Model version."""
        ...

    @property
    def description(self) -> str | None:
        """Model description."""
        ...

    @property
    def author(self) -> str | None:
        """Model author."""
        ...

    @property
    def license(self) -> str | None:
        """Model license."""
        ...

    @property
    def min_parser_version(self) -> str | None:
        """Minimum parser version required."""
        ...
