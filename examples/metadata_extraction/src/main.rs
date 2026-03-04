// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! TFLite model metadata extraction example.
//!
//! Loads a model file and extracts embedded metadata fields such as name,
//! version, author, description, and license.
//!
//! ```sh
//! cargo run -p metadata-extraction -- model.tflite
//! ```

use edgefirst_tflite::metadata::Metadata;
use edgefirst_tflite::{Library, Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "model.tflite".to_string());

    // Discover and load the TFLite shared library.
    let lib = Library::new()?;

    // Load the model from a file.
    let model = Model::from_file(&lib, &path)?;

    println!("Model: {path}");
    println!("Size:  {} bytes", model.data().len());

    // Extract metadata from the raw model bytes.
    let metadata = Metadata::from_model_bytes(model.data());

    // Display metadata using the Display impl.
    let display = metadata.to_string();
    if display.is_empty() {
        println!("No metadata found in model.");
    } else {
        println!("\n--- Metadata (Display) ---");
        print!("{metadata}");
    }

    // Individual field access for programmatic use.
    println!("\n--- Individual Fields ---");
    println!(
        "  Name:        {}",
        metadata.name.as_deref().unwrap_or("<none>")
    );
    println!(
        "  Version:     {}",
        metadata.version.as_deref().unwrap_or("<none>")
    );
    println!(
        "  Author:      {}",
        metadata.author.as_deref().unwrap_or("<none>")
    );
    println!(
        "  Description: {}",
        metadata.description.as_deref().unwrap_or("<none>")
    );
    println!(
        "  License:     {}",
        metadata.license.as_deref().unwrap_or("<none>")
    );

    Ok(())
}
