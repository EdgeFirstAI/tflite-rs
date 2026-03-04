/****************************************************************************
 *
 *    Copyright (c) 2025 Au-Zone Technologies
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#ifndef TENSORFLOW_LITE_DELEGATES_VX_DELEGATE_DMABUF_H_
#define TENSORFLOW_LITE_DELEGATES_VX_DELEGATE_DMABUF_H_

#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * DMA-BUF synchronization modes for cache coherency.
 */
typedef enum {
  kVxDmaBufSyncNone = 0,      // No synchronization needed
  kVxDmaBufSyncRead = 1,      // CPU will read from buffer
  kVxDmaBufSyncWrite = 2,     // CPU will write to buffer
  kVxDmaBufSyncReadWrite = 3, // CPU will read and write
} VxDmaBufSyncMode;

/**
 * DMA-BUF ownership model.
 */
typedef enum {
  kVxDmaBufOwnerClient = 0,   // Client owns the buffer (import mode)
  kVxDmaBufOwnerDelegate = 1, // Delegate owns the buffer (export mode)
} VxDmaBufOwnership;

/**
 * DMA-BUF descriptor returned when requesting a delegate-allocated buffer.
 */
typedef struct {
  int fd;           // DMA-BUF file descriptor
  size_t size;      // Buffer size in bytes
  void* map_ptr;    // Optional: mmap'd pointer (NULL if not mapped)
} VxDmaBufDesc;

/**
 * Get the current VxDelegate instance (the inner DerivedDelegateData*).
 *
 * When TFLite's TfLiteExternalDelegateCreate() is used, the returned pointer
 * is a wrapper — not the actual DerivedDelegateData*.  All VxDelegate DMA-BUF
 * API functions require the inner pointer.  Call this function to obtain it.
 *
 * @return Inner delegate pointer (as TfLiteDelegate*), or NULL if none exists.
 */
TfLiteDelegate* VxDelegateGetInstance(void);

/**
 * Register an externally-allocated DMA-BUF with the delegate.
 *
 * Use this when the application already has a dmabuf (e.g., from V4L2,
 * DRM, or dma_heap allocation) and wants to use it for zero-copy inference.
 *
 * @param delegate The VX delegate instance
 * @param fd DMA-BUF file descriptor (caller retains ownership)
 * @param size Size of the buffer in bytes
 * @param sync_mode Default synchronization mode for this buffer
 * @return Buffer handle, or kTfLiteNullBufferHandle on failure
 */
TfLiteBufferHandle VxDelegateRegisterDmaBuf(TfLiteDelegate* delegate,
                                             int fd,
                                             size_t size,
                                             VxDmaBufSyncMode sync_mode);

/**
 * Unregister a previously registered DMA-BUF.
 *
 * @param delegate The VX delegate instance
 * @param handle Buffer handle returned by VxDelegateRegisterDmaBuf
 * @return kTfLiteOk on success
 */
TfLiteStatus VxDelegateUnregisterDmaBuf(TfLiteDelegate* delegate,
                                         TfLiteBufferHandle handle);

/**
 * Request the delegate to allocate a DMA-BUF for a tensor.
 *
 * Use this when you want the delegate to allocate the buffer and provide
 * the fd to the application (export mode). Useful for output tensors that
 * will be consumed by downstream hardware (display, encoder, etc.).
 *
 * @param delegate The VX delegate instance
 * @param tensor_index TFLite tensor index to bind to
 * @param ownership Who owns the buffer lifetime
 * @param desc Output: DMA-BUF descriptor with fd and size
 * @return Buffer handle, or kTfLiteNullBufferHandle on failure
 */
TfLiteBufferHandle VxDelegateRequestDmaBuf(TfLiteDelegate* delegate,
                                            int tensor_index,
                                            VxDmaBufOwnership ownership,
                                            VxDmaBufDesc* desc);

/**
 * Release a delegate-allocated DMA-BUF.
 *
 * @param delegate The VX delegate instance
 * @param handle Buffer handle returned by VxDelegateRequestDmaBuf
 * @return kTfLiteOk on success
 */
TfLiteStatus VxDelegateReleaseDmaBuf(TfLiteDelegate* delegate,
                                      TfLiteBufferHandle handle);

/**
 * Begin CPU access to a DMA-BUF.
 *
 * Call this before the CPU reads from or writes to a dmabuf that may have
 * been accessed by the NPU. This ensures cache coherency.
 *
 * @param delegate The VX delegate instance
 * @param handle Buffer handle
 * @param mode Access mode (read, write, or both)
 * @return kTfLiteOk on success
 */
TfLiteStatus VxDelegateBeginCpuAccess(TfLiteDelegate* delegate,
                                       TfLiteBufferHandle handle,
                                       VxDmaBufSyncMode mode);

/**
 * End CPU access to a DMA-BUF.
 *
 * Call this after the CPU finishes reading from or writing to a dmabuf,
 * before the NPU accesses it. This flushes CPU caches if necessary.
 *
 * @param delegate The VX delegate instance
 * @param handle Buffer handle
 * @param mode Access mode that was used
 * @return kTfLiteOk on success
 */
TfLiteStatus VxDelegateEndCpuAccess(TfLiteDelegate* delegate,
                                     TfLiteBufferHandle handle,
                                     VxDmaBufSyncMode mode);

/**
 * Bind a DMA-BUF to a TFLite tensor for zero-copy inference.
 *
 * After binding, the delegate will use this dmabuf directly instead of
 * copying data to/from the TFLite tensor buffer.
 *
 * @param delegate The VX delegate instance
 * @param handle Buffer handle
 * @param tensor_index TFLite tensor index to bind to
 * @return kTfLiteOk on success
 */
TfLiteStatus VxDelegateBindDmaBufToTensor(TfLiteDelegate* delegate,
                                           TfLiteBufferHandle handle,
                                           int tensor_index);

/**
 * Check if DMA-BUF zero-copy is supported on this platform.
 *
 * @param delegate The VX delegate instance
 * @return true if supported, false otherwise
 */
bool VxDelegateIsDmaBufSupported(TfLiteDelegate* delegate);

/**
 * Get the DMA-BUF file descriptor for a buffer handle.
 *
 * @param delegate The VX delegate instance
 * @param handle Buffer handle
 * @return fd on success, -1 on failure
 */
int VxDelegateGetDmaBufFd(TfLiteDelegate* delegate, TfLiteBufferHandle handle);

/**
 * Synchronize a DMA-BUF for device (NPU) access.
 *
 * Call this after CPU writes to flush caches before NPU reads.
 * In a true zero-copy pipeline (camera→NPU→display), this is not needed
 * per-inference since the buffers are managed by hardware.
 *
 * @param delegate The VX delegate instance
 * @param handle Buffer handle
 * @return kTfLiteOk on success
 */
TfLiteStatus VxDelegateSyncForDevice(TfLiteDelegate* delegate,
                                      TfLiteBufferHandle handle);

/**
 * Synchronize a DMA-BUF for CPU access.
 *
 * Call this before CPU reads to invalidate caches after NPU writes.
 * In a true zero-copy pipeline (camera→NPU→display), this is not needed
 * per-inference since the buffers are managed by hardware.
 *
 * @param delegate The VX delegate instance
 * @param handle Buffer handle
 * @return kTfLiteOk on success
 */
TfLiteStatus VxDelegateSyncForCpu(TfLiteDelegate* delegate,
                                   TfLiteBufferHandle handle);

/* ============================================================================
 * Buffer Cycling APIs
 * ============================================================================
 *
 * These APIs enable cycling through multiple DMA-BUF buffers (e.g., a pool
 * of 4 camera buffers) without graph recompilation. The delegate uses
 * TIM-VX's SwapHandle() mechanism to change the underlying buffer.
 */

/**
 * Set the active DMA-BUF for a tensor (buffer cycling).
 *
 * Use this to switch which buffer from a pool is used for the next inference.
 * The buffer must have been previously registered with VxDelegateRegisterDmaBuf.
 * This is the key API for buffer cycling - call it before each Invoke() to
 * select which buffer the NPU should read from or write to.
 *
 * @param delegate The VX delegate instance
 * @param tensor_index TFLite tensor index
 * @param handle Buffer handle to make active
 * @return kTfLiteOk on success, kTfLiteError if handle not registered or
 *         tensor not bound to dmabuf
 *
 * Example - V4L2 camera buffer cycling:
 * @code
 * // Register 4 camera buffers at startup
 * TfLiteBufferHandle handles[4];
 * for (int i = 0; i < 4; i++) {
 *     handles[i] = VxDelegateRegisterDmaBuf(delegate, camera_fds[i], size, kVxDmaBufSyncNone);
 *     VxDelegateBindDmaBufToTensor(delegate, handles[i], input_tensor_idx);
 * }
 *
 * // First inference compiles the graph
 * interpreter->Invoke();
 *
 * // Subsequent inferences: cycle through buffers
 * while (running) {
 *     int buf_idx = dequeue_camera_buffer();
 *     VxDelegateSetActiveDmaBuf(delegate, input_tensor_idx, handles[buf_idx]);
 *     interpreter->Invoke();
 *     enqueue_camera_buffer(buf_idx);
 * }
 * @endcode
 */
TfLiteStatus VxDelegateSetActiveDmaBuf(TfLiteDelegate* delegate,
                                        int tensor_index,
                                        TfLiteBufferHandle handle);

/**
 * Invalidate the compiled graph, forcing recompilation on next Invoke().
 *
 * Use this when tensor dimensions or format change (e.g., after GStreamer
 * caps renegotiation). The next Invoke() will rebuild the TIM-VX graph
 * with new tensor specifications and dmabuf bindings.
 *
 * @param delegate The VX delegate instance
 * @return kTfLiteOk on success
 *
 * Note: Graph recompilation is expensive (~10-100ms). Only call this when
 * the tensor format actually changes, not for buffer cycling.
 */
TfLiteStatus VxDelegateInvalidateGraph(TfLiteDelegate* delegate);

/**
 * Check if the graph has been compiled.
 *
 * Useful for determining if the first inference has occurred (which
 * triggers graph compilation) or if InvalidateGraph() was called.
 *
 * @param delegate The VX delegate instance
 * @return true if graph is compiled, false otherwise
 */
bool VxDelegateIsGraphCompiled(TfLiteDelegate* delegate);

/**
 * Get the currently active buffer handle for a tensor.
 *
 * @param delegate The VX delegate instance
 * @param tensor_index TFLite tensor index
 * @return Active buffer handle, or kTfLiteNullBufferHandle if none set
 */
TfLiteBufferHandle VxDelegateGetActiveBuffer(TfLiteDelegate* delegate,
                                              int tensor_index);

/* ============================================================================
 * Camera Adaptor APIs
 * ============================================================================
 *
 * These APIs configure runtime preprocessing to convert camera formats to
 * model-expected formats (e.g., RGBA -> RGB). The conversion operations are
 * injected into the TIM-VX graph and run on the NPU.
 *
 * This is the runtime counterpart to the EdgeFirst Python CameraAdaptor library.
 */

/**
 * Configure camera adaptor for an input tensor.
 *
 * Call this before the first Invoke() to inject preprocessing operations.
 * The format can be a lowercase name (e.g., "rgba", "yuyv") or a FourCC
 * code (e.g., "RGBA", "YUYV", "NV12").
 *
 * @param delegate The VX delegate instance
 * @param input_tensor_index TFLite input tensor index
 * @param adaptor Format string (e.g., "rgba", "yuyv", "nv12")
 * @return kTfLiteOk on success
 */
TfLiteStatus VxCameraAdaptorSetFormat(TfLiteDelegate* delegate,
                                       int input_tensor_index,
                                       const char* adaptor);

/**
 * Configure camera adaptor with resize options.
 *
 * Extended version that also configures optional resize/letterbox.
 *
 * @param delegate The VX delegate instance
 * @param input_tensor_index TFLite input tensor index
 * @param adaptor Format string
 * @param resize_width Target width (0 = no resize)
 * @param resize_height Target height (0 = no resize)
 * @param letterbox If true, preserve aspect ratio with padding
 * @param letterbox_color RGB packed padding color (e.g., 0x808080 for grey)
 * @return kTfLiteOk on success
 */
TfLiteStatus VxCameraAdaptorSetFormatEx(TfLiteDelegate* delegate,
                                         int input_tensor_index,
                                         const char* adaptor,
                                         uint32_t resize_width,
                                         uint32_t resize_height,
                                         bool letterbox,
                                         uint32_t letterbox_color);

/**
 * Configure camera adaptor with explicit model format.
 *
 * Use this when the model expects a different format than RGB (e.g., BGR).
 * Supported combinations:
 * - RGBA → RGB, RGBA → BGR
 * - BGRA → RGB, BGRA → BGR
 * - Similar for RGBX, BGRX, ARGB, ABGR, XRGB, XBGR
 *
 * @param delegate The VX delegate instance
 * @param input_tensor_index TFLite input tensor index
 * @param adaptor Camera format string (e.g., "rgba", "bgra")
 * @param model_format Model's expected format (e.g., "rgb", "bgr")
 * @return kTfLiteOk on success
 */
TfLiteStatus VxCameraAdaptorSetFormats(TfLiteDelegate* delegate,
                                        int input_tensor_index,
                                        const char* adaptor,
                                        const char* model_format);

/**
 * Configure camera adaptor using V4L2-style FourCC code.
 *
 * @param delegate The VX delegate instance
 * @param input_tensor_index TFLite input tensor index
 * @param fourcc V4L2 FourCC code (e.g., V4L2_PIX_FMT_RGBA32)
 * @return kTfLiteOk on success
 */
TfLiteStatus VxCameraAdaptorSetFourCC(TfLiteDelegate* delegate,
                                       int input_tensor_index,
                                       uint32_t fourcc);

/**
 * Query current adaptor format for a tensor.
 *
 * @param delegate The VX delegate instance
 * @param input_tensor_index TFLite input tensor index
 * @return Format string, or NULL if not configured
 */
const char* VxCameraAdaptorGetFormat(TfLiteDelegate* delegate,
                                      int input_tensor_index);

/**
 * Check if an adaptor format is supported.
 *
 * @param adaptor Format string to check
 * @return true if supported
 */
bool VxCameraAdaptorIsSupported(const char* adaptor);

/**
 * Get the number of input channels for a format.
 *
 * @param adaptor Format string
 * @return Input channel count (e.g., 4 for RGBA)
 */
int VxCameraAdaptorGetInputChannels(const char* adaptor);

/**
 * Get the number of output channels for a format.
 *
 * @param adaptor Format string
 * @return Output channel count (e.g., 3 for RGBA->RGB)
 */
int VxCameraAdaptorGetOutputChannels(const char* adaptor);

/**
 * Get the FourCC code for an adaptor format.
 *
 * @param adaptor Format string
 * @return FourCC string, or NULL if unknown
 */
const char* VxCameraAdaptorGetFourCC(const char* adaptor);

/**
 * Convert a FourCC code to adaptor format string.
 *
 * @param fourcc FourCC string
 * @return Format string
 */
const char* VxCameraAdaptorFromFourCC(const char* fourcc);

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_DELEGATES_VX_DELEGATE_DMABUF_H_
