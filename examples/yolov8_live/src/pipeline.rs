// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Threaded inference pipeline.
//!
//! The stages of a frame use different hardware — GPU letterbox, NPU
//! inference, CPU decode, then GPU drawing — so running them on their
//! own threads lets consecutive frames overlap:
//!
//! ```text
//!  main (camera, display)      pre (GPU)           npu (NPU)          decode (CPU)
//!  frame N+3 in      ──────▶   letterbox N+2  ──▶  invoke N+1   ──▶   decode N   ──▶  main: draw N
//!                              into a free slot     slot, copy out
//! ```
//!
//! The Neutron NPU runs one inference at a time whatever the number of
//! interpreter contexts, so extra contexts are not used for parallel
//! inference. Two *slots* — each an interpreter with its own delegate and
//! input buffer, as in the `async_pipeline` example — let the GPU letterbox
//! the next frame into one slot while the NPU runs the other.
//!
//! Work moves between threads by ownership: a slot's input buffer travels
//! with its frame and returns once the NPU has read it, and output buffers
//! come from a small pool the decode thread refills. Every channel is
//! bounded, so a full pipeline pushes back to the camera, which then drops
//! frames rather than queue them.

use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::time::{Duration, Instant};

use edgefirst_decoder::{Decoder, DetectBox, ProtoData, Segmentation};
use edgefirst_image::{ConvertOptions, ImageProcessor, ImageProcessorTrait as _};
use edgefirst_tensor::TensorDyn;
use edgefirst_tflite::Interpreter;

use crate::error::Result;
use crate::{ModelInput, OutputBuffers};

/// A camera frame on its way through the pipeline.
#[derive(Debug)]
pub struct Frame {
    /// libcamera buffer index (the request cookie).
    pub index: usize,
    /// The buffer imported as a HAL image.
    pub image: TensorDyn,
    /// When the frame entered the pipeline.
    pub entered: Instant,
}

/// Time spent on a frame in each pipeline thread.
#[derive(Debug, Default, Clone, Copy)]
pub struct StageTimes {
    pub preprocess: Duration,
    pub infer: Duration,
    pub copy: Duration,
    pub decode: Duration,
}

/// An interpreter slot's input buffer, passed between the pre and npu
/// threads as the right to use that slot.
pub struct Slot {
    id: usize,
    input: ModelInput,
}

impl Slot {
    pub fn new(id: usize, input: ModelInput) -> Self {
        Self { id, input }
    }
}

/// A letterboxed frame waiting for the NPU.
pub struct Prepared {
    slot: Slot,
    frame: Frame,
    times: StageTimes,
}

/// A frame whose outputs have been copied out of the interpreter.
pub struct Inferred {
    frame: Frame,
    outputs: OutputBuffers,
    times: StageTimes,
}

/// A decoded frame, ready to draw.
#[derive(Debug)]
pub struct Decoded {
    pub frame: Frame,
    pub detections: Vec<DetectBox>,
    pub masks: Vec<Segmentation>,
    pub protos: Option<ProtoData>,
    pub times: StageTimes,
}

/// What the main thread waits on: camera frames and decoded results.
#[derive(Debug)]
pub enum Event {
    Frame(libcamera::request::Request),
    Decoded(Box<Decoded>),
}

/// Pre thread: letterbox each frame into a free slot's input buffer.
pub fn preprocess(
    mut processor: ImageProcessor,
    letterbox: ConvertOptions,
    frames: &Receiver<Frame>,
    free: &Receiver<Slot>,
    prepared: &SyncSender<Prepared>,
) -> Result<()> {
    while let Ok(frame) = frames.recv() {
        let Ok(mut slot) = free.recv() else {
            break;
        };
        let start = Instant::now();
        processor.convert(&frame.image, slot.input.image(), Some(letterbox))?;
        let times = StageTimes {
            preprocess: start.elapsed(),
            ..StageTimes::default()
        };
        if prepared.send(Prepared { slot, frame, times }).is_err() {
            break;
        }
    }
    Ok(())
}

/// Npu thread: run each prepared slot and copy its outputs into a pooled
/// buffer, then hand the slot back to the pre thread.
pub fn infer(
    mut interpreters: Vec<Interpreter<'_>>,
    prepared: &Receiver<Prepared>,
    free: &Sender<Slot>,
    pool: &Receiver<OutputBuffers>,
    inferred: &SyncSender<Inferred>,
) -> Result<()> {
    while let Ok(Prepared {
        slot,
        frame,
        mut times,
    }) = prepared.recv()
    {
        let interpreter = &mut interpreters[slot.id];
        let start = Instant::now();
        slot.input.submit(interpreter)?;
        interpreter.invoke()?;
        let invoked = Instant::now();
        let Ok(mut outputs) = pool.recv() else {
            break;
        };
        outputs.sync_from(interpreter)?;
        times.infer = invoked - start;
        times.copy = invoked.elapsed();
        // The pre thread may already be blocked waiting for this slot.
        if free.send(slot).is_err() {
            break;
        }
        if inferred
            .send(Inferred {
                frame,
                outputs,
                times,
            })
            .is_err()
        {
            break;
        }
    }
    Ok(())
}

/// Decode thread: turn model outputs into detections for the main thread.
///
/// `decode_proto` returns `None` both for a detection model, whose boxes it
/// has already decoded, and for a segmentation model it cannot give
/// prototypes for; only the second needs `decode`.
pub fn decode(
    decoder: &Decoder,
    segmentation: bool,
    inferred: &Receiver<Inferred>,
    pool: &Sender<OutputBuffers>,
    events: &Sender<Event>,
) -> Result<()> {
    while let Ok(Inferred {
        frame,
        outputs,
        mut times,
    }) = inferred.recv()
    {
        let start = Instant::now();
        let mut detections = Vec::new();
        let mut masks = Vec::new();
        let protos = decoder.decode_proto(&outputs.refs(), &mut detections)?;
        if protos.is_none() && segmentation {
            decoder.decode(&outputs.refs(), &mut detections, &mut masks)?;
        }
        times.decode = start.elapsed();
        // Decoding is done with the buffers; the npu thread may be waiting.
        let _ = pool.send(outputs);
        let result = Decoded {
            frame,
            detections,
            masks,
            protos,
            times,
        };
        if events.send(Event::Decoded(Box::new(result))).is_err() {
            break;
        }
    }
    Ok(())
}
