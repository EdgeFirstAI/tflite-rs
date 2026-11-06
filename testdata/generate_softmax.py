# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

"""Generate the quantized SOFTMAX test models for the EdgeFirst delegate.

Both models take x[1,2,8,40] and compute:

    probs  = softmax(x)                 graph output, and an ADD input
    scaled = softmax(x * 3)             a MUL feeds the second SOFTMAX
    sum    = probs + scaled             graph output

so the delegate sees two SOFTMAX nodes in separate partitions: one reads a
graph input and writes a graph output that an unclaimed ADD also reads, the
other sits between two unclaimed nodes.

    softmax_int8.tflite    int8 activations (output scale 1/256, zero point -128)
    softmax_int16.tflite   int16 activations, int8 weights (16x8)

Usage (TensorFlow 2.x in a virtual environment):
    python testdata/generate_softmax.py
"""

import numpy as np
import tensorflow as tf

SHAPE = (2, 8, 40)


def build_model():
    x = tf.keras.Input(shape=SHAPE, batch_size=1, name="x", dtype=tf.float32)
    probs = tf.keras.layers.Softmax(axis=-1, name="probs")(x)
    scaled = tf.keras.layers.Lambda(lambda t: tf.nn.softmax(t * 3.0, axis=-1), name="scaled")(x)
    total = tf.keras.layers.Add(name="sum")([probs, scaled])
    return tf.keras.Model(inputs=x, outputs=[probs, total])


def representative_dataset():
    rng = np.random.default_rng(7)
    for _ in range(64):
        yield [rng.normal(0.0, 3.0, size=(1, *SHAPE)).astype(np.float32)]


def convert(model, int16):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    if int16:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]
        converter.inference_input_type = tf.int16
        converter.inference_output_type = tf.int16
    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter.convert()


def describe(path, content):
    interp = tf.lite.Interpreter(model_content=content)
    tensors = {t["index"]: t for t in interp.get_tensor_details()}
    print(path)
    for op in interp._get_ops_details():
        ins = [tensors[i] for i in op["inputs"] if i >= 0]
        outs = [tensors[i] for i in op["outputs"]]
        print(
            f"  {op['op_name']:10s}",
            [(t["dtype"].__name__, t["quantization"]) for t in ins],
            "->",
            [(t["dtype"].__name__, t["quantization"]) for t in outs],
        )


def main():
    model = build_model()
    for name, int16 in (("softmax_int8", False), ("softmax_int16", True)):
        content = convert(model, int16)
        path = f"testdata/{name}.tflite"
        with open(path, "wb") as f:
            f.write(content)
        describe(path, content)


if __name__ == "__main__":
    main()
