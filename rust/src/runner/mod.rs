use pyo3::prelude::*;
use uuid::Uuid;

use crate::comm;

pub async fn runner_main() -> anyhow::Result<()> {
    let ret = Python::with_gil(|py| {
        let py_runner = PyModule::import(py, "punica_runner")?;
        let get_all_gpu_info_fn = py_runner.getattr("get_all_gpu_info")?;
        get_all_gpu_info_fn
            .call0()?
            .extract::<Vec<(String, String, u64, i8, i8)>>()
    })?;

    let mut devices = Vec::new();
    for (uuid, name, total_memory, sm_major, sm_minor) in ret {
        devices.push(comm::CudaDeviceProp {
            uuid: Uuid::parse_str(&uuid).unwrap(),
            name,
            total_memory,
            sm_major,
            sm_minor,
        });
    }
    println!("{:?}", devices);
    Ok(())
}
