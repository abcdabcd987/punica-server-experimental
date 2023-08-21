use pyo3::prelude::*;

pub fn hello_main() -> PyResult<()> {
    Python::with_gil(|py| {
        let py_runner = PyModule::import(py, "punica_runner")?;
        let hello_fn = py_runner.getattr("hello")?;
        hello_fn.call0()?;
        Ok::<(), PyErr>(())
    })?;
    println!("done");
    Ok(())
}
