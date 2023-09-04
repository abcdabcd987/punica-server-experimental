use std::process::{Command, Stdio};

use crate::comm;

pub fn device_query() -> anyhow::Result<Vec<comm::CudaDeviceProp>> {
    let child = Command::new("python")
        .args(["-m", "punica_runner.device_query"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()?;

    rmp_serde::from_slice(&child.stdout).map_err(|e| e.into())
}
