# edgefirst-tflite

Ergonomic Rust API for [TensorFlow Lite](https://www.tensorflow.org/lite)
inference with DMA-BUF zero-copy and NPU-accelerated preprocessing.

## Usage

```toml
[dependencies]
edgefirst-tflite = "0.9"
```

```rust,no_run
use edgefirst_tflite::{Library, Model, Interpreter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lib = Library::new()?;
    let model = Model::from_file(&lib, "model.tflite")?;

    let mut interpreter = Interpreter::builder(&lib)?
        .num_threads(4)
        .build(&model)?;

    interpreter.invoke()?;

    for (i, tensor) in interpreter.outputs()?.iter().enumerate() {
        println!("output[{i}]: {tensor}");
    }
    Ok(())
}
```

When the loaded shared library also exports LiteRT Next (`LiteRt*`) symbols, the `litert` module offers explicit accelerator selection and runtime-placed buffers:

```rust,no_run
use edgefirst_tflite::{Library, litert};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lib = Library::new()?;
    if !lib.has_litert() {
        // Names the symbol that failed to resolve, which distinguishes a
        // classic TFLite build from a partial LiteRT one.
        eprintln!("LiteRT unavailable: {:?}", lib.litert_missing_symbol());
        return Ok(());
    }

    let env = litert::Environment::new(&lib)?;
    let model = litert::Model::from_file(&env, "model.tflite")?;
    let opts = litert::Options::new(&lib)?
        .hardware_accelerators(litert::HwAccelerators::NPU | litert::HwAccelerators::CPU)?;
    let mut compiled = litert::CompiledModel::create(&env, &model, &opts)?;
    println!("fully accelerated: {}", compiled.is_fully_accelerated()?);

    let mut inputs = vec![compiled.create_input_buffer(0, 0)?];
    let mut outputs = vec![compiled.create_output_buffer(0, 0)?];
    let input_size = inputs[0].size();
    inputs[0].write_bytes(&vec![0u8; input_size])?;

    compiled.run_default(&mut inputs, &mut outputs)?;
    println!("read {} output bytes", outputs[0].read_bytes()?.len());
    Ok(())
}
```

Handles form an ownership chain the borrow checker enforces: a `CompiledModel`
borrows both its `Environment` and its `Model` (the runtime reads the model's
flatbuffer on every inference), and `BufferRequirements` borrows the
`CompiledModel` that owns it. Dropping a parent too early is a compile error
rather than a crash at runtime.

## API Tour

The main entry points are:

- **`Library`** -- Load the TFLite shared library (auto-discovery or explicit path); probes LiteRT via `has_litert()` / `litert()` / `litert_missing_symbol()`
- **`Model`** -- Load a model from a file or byte buffer (classic TFLite)
- **`Interpreter`** -- Run inference via a builder pattern
- **`litert`** -- LiteRT Next `CompiledModel` path (`Environment`, `Options`, `TensorBuffer`, accelerators)
- **`Tensor` / `TensorMut`** -- Type-safe tensor access with shape and quantization info
- **`Delegate`** -- Hardware acceleration via external delegates
- **`DmaBuf`** -- Zero-copy DMA-BUF operations (feature: `dmabuf`)
- **`CameraAdaptor`** -- NPU preprocessing configuration (feature: `camera_adaptor`)
- **`Metadata`** -- Model metadata extraction (feature: `metadata`)

## Feature Flags

| Feature | Description |
|---------|-------------|
| `dmabuf` | DMA-BUF zero-copy inference via `VxDelegate` |
| `camera_adaptor` | NPU-accelerated format conversion |
| `metadata` | TFLite model metadata extraction |
| `full` | Enables all optional features |

## License

Apache-2.0
