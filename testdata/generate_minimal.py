# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

"""Generate a minimal TFLite model for testing.

Produces a model with a single Add op: float32[1,4] input + constant -> float32[1,4] output.
The model is small and exercises the core TFLite interpreter pipeline.

Usage:
    python testdata/generate_minimal.py
"""

import numpy as np
import tensorflow as tf

# Single float32[1,4] input -> Add constant -> float32[1,4] output
input_tensor = tf.keras.Input(shape=(4,), batch_size=1, name="input", dtype=tf.float32)
# Add a constant (ones) using a Lambda layer
output = tf.keras.layers.Lambda(lambda x: x + tf.constant([1.0, 1.0, 1.0, 1.0]))(input_tensor)

model = tf.keras.Model(inputs=input_tensor, outputs=output)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

out_path = "testdata/minimal.tflite"
with open(out_path, "wb") as f:
    f.write(tflite_model)

print(f"Wrote {len(tflite_model)} bytes to {out_path}")

# Verify: load and run inference
interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()

input_details = interp.get_input_details()
output_details = interp.get_output_details()

print(f"Inputs:  {len(input_details)}")
for d in input_details:
    print(f"  {d['name']}: shape={d['shape']}, dtype={d['dtype']}")

print(f"Outputs: {len(output_details)}")
for d in output_details:
    print(f"  {d['name']}: shape={d['shape']}, dtype={d['dtype']}")

# Test inference: [1,2,3,4] + [1,1,1,1] = [2,3,4,5]
input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
interp.set_tensor(input_details[0]["index"], input_data)
interp.invoke()
output_data = interp.get_tensor(output_details[0]["index"])
print(f"Input:  {input_data}")
print(f"Output: {output_data}")
