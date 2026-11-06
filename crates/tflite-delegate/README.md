# edgefirst-tflite-delegate

A TensorFlow Lite delegate that runs the operators NPU delegates leave on the CPU with the NEON kernels of [`edgefirst-kernels`](https://crates.io/crates/edgefirst-kernels), in place of TFLite's builtin CPU kernels.

It claims connected regions of:

- **Compute operators**: `SOFTMAX` (int8, int16), `LOGISTIC`, `ADD`, `SUB`, `MUL` (int8), `BATCH_MATMUL` (int8) and 1×1 `CONV_2D` (int8). Each reproduces TFLite's result: bit for bit on the scalar kernel tier, and on NEON too except int8 `SOFTMAX`, which is within ±1 LSB.
- **Layout operators**: `TRANSPOSE`, `RESHAPE`, `SLICE` and `CONCATENATION`. The delegate never runs these; they become strided views and part lists that the compute kernels read and write directly.
- **Attention blocks** (`[MUL] → BATCH_MATMUL → SOFTMAX → BATCH_MATMUL`, as in YOLO11 and YOLO26). Each block is fused into one kernel that reads Q, K and V where they are and keeps every one of TFLite's int8 roundings.
- **Distribution-focal-loss heads** (`SOFTMAX → [RESHAPE…] → CONV_2D` with one 1×1 output channel, as in YOLOv8 and YOLO11). Each head is fused into one kernel that skips the 8-bit probabilities between the two operators. Its results sit closer to the float model than TFLite's, within a few LSB of them.

The delegate claims a node only when it can reproduce it (the fused heads aside), and leaves every other node to TFLite or to the delegates added after it. Kernels read and write tensors in the memory TFLite gives them; the delegate never sets buffer handles.

The delegate is built on the TFLite opaque delegate API, so the runtime must export it (TFLite 2.17 and later, as the NXP BSP runtimes do). The crate needs Rust 1.94, as `edgefirst-kernels` does.

## Usage

```toml
[dependencies]
edgefirst-tflite = "0.10"
edgefirst-tflite-delegate = "0.10"
```

```rust,no_run
use edgefirst_tflite::{Delegate, Interpreter, Library, Model};
use edgefirst_tflite_delegate::EdgeFirstDelegate;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lib = Library::new()?;
    let model = Model::from_file(&lib, "model.imx95.tflite")?;

    let edgefirst = EdgeFirstDelegate::new()?;
    let stats = edgefirst.stats();
    let mut interpreter = Interpreter::builder(&lib)?
        .num_threads(4)
        .delegate(Delegate::load("/usr/lib/libneutron_delegate.so")?)
        .delegate(edgefirst.into_delegate(&lib)?)
        .build(&model)?;
    println!("EdgeFirst delegate claimed {} nodes", stats.claimed_nodes());

    interpreter.invoke()?;
    Ok(())
}
```

Delegates are applied in the order they are added:

- **After the NPU delegate** (the default, `Scope::Everything`), the EdgeFirst delegate takes over the NPU delegate's CPU fallback. Use this with Neutron on i.MX 95.
- **Before the NPU delegate**, with `Options::default().scope(Scope::Attention)`, it keeps only the attention blocks off the NPU. Use this with VX on i.MX 8M Plus.

Use one instance per interpreter.

## Threads

Kernels use as many threads as the interpreter was given (`InterpreterBuilder::num_threads`, `TfLiteInterpreterOptionsSetNumThreads` or `Interpreter::SetNumThreads`), the calling thread included, up to the CPUs the process may use. With no thread count set, they use only the calling thread, as TFLite's own CPU kernels do. The count is read on every invoke. The delegate keeps its worker threads across invokes: they poll between the kernels of an invoke and sleep as soon as it ends, leaving the CPUs to the operators between partitions.

## Kernels and accuracy

`Options::kernels` names the TFLite build whose int8 rounding the delegate reproduces:

- `NeonRuy`, the default on aarch64: TFLite's Arm builds.
- `NeonGemmlowp`: x86 builds with SSE4.1, such as the LiteRT wheels.
- `Ruy` and `Reference`: other builds.

`EDGEFIRST_KERNELS_FORCE_TIER` (`scalar`, `neon`, `neon_fp16`, `neon_dotprod`) forces the CPU tier. `EdgeFirstDelegate::with_cpu_features` sets the tier without reading the environment.

Each kernel run emits a `kernels.<op>` `tracing` span. TFLite's profiler labels the delegate's partitions `DELEGATE`.

## License

Apache-2.0. See [LICENSE](https://github.com/EdgeFirstAI/tflite-rs/blob/main/LICENSE).
