use std::path::{Path, PathBuf};

use anyhow::anyhow;
use pyo3::prelude::{Py, PyModule, PyObject, PyResult, Python};
use pyo3::types::{PyDict, PyList};
use tokenizers::Tokenizer;
use uuid::Uuid;

use crate::comm;

#[derive(Debug, clap::Args)]
pub struct RunnerArgs {
    #[arg(long)]
    pub model_path: PathBuf,
}

fn build_tokenizer(model_path: &Path) -> anyhow::Result<(Tokenizer, u32)> {
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|e| {
            anyhow::anyhow!(
                "failed to load tokenizer from {}: {}",
                model_path.display(),
                e
            )
        })?;

    let file =
        std::fs::read_to_string(model_path.join("tokenizer_config.json"))
            .map_err(|e| {
                anyhow!(
                    "Failed to read tokenizer_config.json from {}. Error: {}",
                    model_path.display(),
                    e
                )
            })?;
    let json: serde_json::Value = serde_json::from_str(&file)?;
    let token = json["eos_token"]["content"]
        .as_str()
        .ok_or_else(|| anyhow!("json"))?
        .to_string();
    let eos_id =
        tokenizer.encode(token, false).map_err(|e| anyhow!("{}", e))?.get_ids()
            [0];
    Ok((tokenizer, eos_id))
}

pub async fn runner_main(args: RunnerArgs) -> anyhow::Result<()> {
    let (tokenizer, eos_id) = build_tokenizer(args.model_path.as_path())?;

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
        let kwargs = PyDict::new(py);
        kwargs.set_item("model_path", args.model_path.to_string_lossy())?;
        kwargs.set_item("dtype", "float16")?;
        kwargs.set_item("max_batch_size", 4)?;
        Ok(cls.call((), Some(kwargs))?.into())
    })?;

    let prompt_template = "[INST] <<SYS>> You are a helpful, respectful and honest assistant. <</SYS>>\n{prompt} [/INST]\n";
    let questions = [
        "Give me a 3 day travel plan for Seattle.",
        "Tell me something about University of Washington.",
        "How to dial in an espresso shot?",
    ];

    // Prefill
    let question = questions[0];
    let prompt = prompt_template.replace("{prompt}", question);
    let input_ids = tokenizer.encode(prompt, false).map_err(|e| {
        anyhow::anyhow!("Failed to tokenizer.encode(). Error: {}", e)
    })?;
    let mut out = Vec::from(input_ids.get_ids());
    let (genidx, next_id, _stop) =
        Python::with_gil(|py| -> PyResult<(usize, u32, i8)> {
            let args = (PyList::new(py, &out),);
            let kwargs = PyDict::new(py);
            kwargs.set_item("temperature", 0.7)?;
            kwargs.set_item("repetition_penalty", 1.1)?;
            kwargs.set_item("top_p", 0.9)?;
            kwargs.set_item("stop_token_ids", vec![eos_id])?;
            kwargs.set_item("max_new_tokens", 600)?;
            py_runner
                .as_ref(py)
                .call_method("prefill", args, Some(kwargs))?
                .extract()
        })?;
    out.push(next_id);
    let text = tokenizer.decode(&out, true).map_err(|e| {
        anyhow::anyhow!("Failed to tokenizer.decode(). Error: {}", e)
    })?;
    println!("{}\n---\n", text);

    loop {
        let ret = Python::with_gil(|py| -> PyResult<Vec<(u32, i8)>> {
            let args = (PyList::new(py, [genidx]),);
            py_runner
                .as_ref(py)
                .call_method("batch_decode", args, None)?
                .extract()
        })?;
        let (next_id, stop) = ret[0];
        out.push(next_id);
        let text = tokenizer.decode(&out, true).map_err(|e| {
            anyhow::anyhow!("Failed to tokenizer.decode(). Error: {}", e)
        })?;
        println!("{}\n---\n", text);

        match stop {
            0 => (),
            1 => println!("\n\nStop Reason: Length"),
            2 => println!("\n\nStop Reason: EOS"),
            _ => return Err(anyhow!("Unknown stop reason: {}", stop)),
        }
        if stop != 0 {
            break;
        }
    }

    Ok(())
}
