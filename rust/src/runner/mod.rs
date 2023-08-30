use pyo3::prelude::{Py, PyModule, PyObject, PyResult, Python};
use uuid::Uuid;

use crate::comm;

pub async fn runner_main() -> anyhow::Result<()> {
    let py_runner = Python::with_gil(|py| -> PyResult<Py<PyModule>> {
        Ok(PyModule::import(py, "punica_runner")?.into())
    })?;
    let ret = Python::with_gil(|py| {
        let py_runner = py_runner.as_ref(py);
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

    let py_runner = Python::with_gil(|py| -> PyResult<PyObject> {
        let module = PyModule::import(py, "punica_runner")?;
        let cls = module.getattr("PunicaRunner")?;
        Ok(cls.call0()?.into())
    })?;
    Python::with_gil(|py| -> PyResult<()> {
        py_runner.as_ref(py).call_method0("add")?;
        Ok(())
    })?;
    let counter: i32 = Python::with_gil(|py| -> PyResult<i32> {
        py_runner.as_ref(py).call_method0("get_counter")?.extract()
    })?;
    println!("counter: {}", counter);
    Python::with_gil(|py| -> PyResult<()> {
        py_runner.as_ref(py).call_method0("add")?;
        Ok(())
    })?;
    let counter: i32 = Python::with_gil(|py| -> PyResult<i32> {
        py_runner.as_ref(py).call_method0("get_counter")?.extract()
    })?;
    println!("counter: {}", counter);
    Ok(())
}
