// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Vendored `TFLite` library download and installation.
//!
//! Downloads a pre-built `libtensorflowlite_c` shared library from the
//! `tflite-rs` GitHub Releases and places it in `OUT_DIR`. The library
//! is discovered at runtime via
//! [`discovery::try_vendored`](../src/discovery.rs).

use std::path::{Path, PathBuf};
use std::{env, fs, io};

/// Default `TFLite` version to download. Override with `TFLITE_VERSION` env
/// var.
const DEFAULT_VERSION: &str = "2.19.0";

/// GitHub repo that hosts the pre-built libraries.
const GITHUB_REPO: &str = "EdgeFirstAI/tflite-rs";

/// Download and install the vendored `TFLite` library into `OUT_DIR`.
pub fn download_and_install() {
    println!("cargo:rerun-if-env-changed=TFLITE_VERSION");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo:rerun-if-changed=vendored.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let version = env::var("TFLITE_VERSION").unwrap_or_else(|_| DEFAULT_VERSION.to_string());

    let (artifact_name, lib_name) = platform_artifact();
    let stamp = out_dir.join(format!("tflite-{version}.stamp"));

    // Skip download if the correct version is already cached.
    if stamp.exists() && out_dir.join(lib_name).exists() {
        emit_env(&out_dir);
        println!(
            "cargo:warning=Using cached vendored TFLite v{version}: {}",
            out_dir.join(lib_name).display()
        );
        return;
    }

    // Remove stale libraries from a different version.
    if out_dir.join(lib_name).exists() {
        fs::remove_file(out_dir.join(lib_name)).ok();
    }
    // Remove any old stamp files.
    for entry in fs::read_dir(&out_dir).into_iter().flatten().flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("tflite-")
            && name.to_string_lossy().ends_with(".stamp")
        {
            fs::remove_file(entry.path()).ok();
        }
    }

    let url = format!(
        "https://github.com/{GITHUB_REPO}/releases/download/tflite-v{version}/{artifact_name}"
    );
    println!("cargo:warning=Downloading TFLite v{version} from {url}");

    // Download to a .tmp file and rename on success to avoid partial artifacts.
    let archive_tmp = out_dir.join(format!("{artifact_name}.tmp"));
    download(&url, &archive_tmp);
    extract(&archive_tmp, &out_dir);
    fs::remove_file(&archive_tmp).ok();

    let lib_path = out_dir.join(lib_name);
    assert!(
        lib_path.exists(),
        "Library not found after extraction: {}.\n\
         Expected the archive to contain '{lib_name}' at the root.",
        lib_path.display()
    );

    // Write version stamp so we can detect version changes on next build.
    fs::write(&stamp, &version).unwrap_or_else(|e| {
        panic!("Failed to write version stamp {}: {e}", stamp.display());
    });

    emit_env(&out_dir);
    println!(
        "cargo:warning=Vendored TFLite v{version} installed to {}",
        lib_path.display()
    );
}

fn emit_env(out_dir: &Path) {
    println!(
        "cargo:rustc-env=EDGEFIRST_TFLITE_VENDORED_DIR={}",
        out_dir.display()
    );
}

fn platform_artifact() -> (&'static str, &'static str) {
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    match (os.as_str(), arch.as_str()) {
        ("linux", "x86_64") => (
            "libtensorflowlite_c-x86_64-linux.tar.gz",
            "libtensorflowlite_c.so",
        ),
        ("linux", "aarch64") => (
            "libtensorflowlite_c-aarch64-linux.tar.gz",
            "libtensorflowlite_c.so",
        ),
        ("macos", _) => (
            "libtensorflowlite_c-macos-universal.tar.gz",
            "libtensorflowlite_c.dylib",
        ),
        ("windows", "x86_64") => (
            "libtensorflowlite_c-windows-x86_64.zip",
            "tensorflowlite_c.dll",
        ),
        _ => panic!(
            "Unsupported vendored target: {os}-{arch}.\n\
             Supported: linux-x86_64, linux-aarch64, macos-*, windows-x86_64.\n\
             Use TFLITE_LIBRARY_PATH to point to a manually-installed library instead."
        ),
    }
}

fn download(url: &str, dest: &Path) {
    let resp = ureq::get(url).call().unwrap_or_else(|e| {
        let hint = if e.to_string().contains("404") {
            "The release artifact was not found. Have you run the tflite.yml \
             workflow to create the tflite-vX.Y.Z release?"
        } else {
            "Check your network connection and try again."
        };
        panic!("Failed to download {url}: {e}\n{hint}");
    });

    // Validate Content-Length if available to detect truncated downloads.
    let content_length: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());

    let mut reader = resp.into_reader();
    let mut file = fs::File::create(dest).unwrap_or_else(|e| {
        panic!("Failed to create {}: {e}", dest.display());
    });

    let written = io::copy(&mut reader, &mut file).unwrap_or_else(|e| {
        panic!("Failed to write {}: {e}", dest.display());
    });

    if let Some(expected) = content_length {
        assert_eq!(
            written,
            expected,
            "Incomplete download of {url}: got {written} bytes, expected {expected}.\n\
             The file may be truncated. Delete {} and retry.",
            dest.display()
        );
    }
}

fn extract(archive: &Path, dest: &Path) {
    let name = archive
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .trim_end_matches(".tmp");

    if name.ends_with(".tar.gz") {
        let file = fs::File::open(archive).unwrap_or_else(|e| {
            panic!("Failed to open {}: {e}", archive.display());
        });
        let gz = flate2::read::GzDecoder::new(file);
        let mut tar = tar::Archive::new(gz);
        tar.unpack(dest).unwrap_or_else(|e| {
            panic!(
                "Failed to extract {}: {e}.\n\
                 The archive may be corrupt. Delete it and retry.",
                archive.display()
            );
        });
    } else if Path::new(name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("zip"))
    {
        let file = fs::File::open(archive).unwrap_or_else(|e| {
            panic!("Failed to open {}: {e}", archive.display());
        });
        let mut zip = zip::ZipArchive::new(file).unwrap_or_else(|e| {
            panic!(
                "Failed to read zip {}: {e}.\n\
                 The archive may be corrupt. Delete it and retry.",
                archive.display()
            );
        });
        zip.extract(dest).unwrap_or_else(|e| {
            panic!(
                "Failed to extract {}: {e}.\n\
                 The archive may be corrupt. Delete it and retry.",
                archive.display()
            );
        });
    } else {
        panic!("Unknown archive format: {name}");
    }
}
