use std::path::PathBuf;

use uuid::Uuid;

use super::device_query::device_query;
use super::executor::GpuExecutor;
use super::tokenizer::Tokenizer;
use crate::comm;

#[derive(Debug, clap::Args)]
pub struct DebugExecutorArgs {
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub gpu_index: usize,
}

pub async fn debug_executor_main(
    args: DebugExecutorArgs,
) -> anyhow::Result<()> {
    let template = "[INST] <<SYS>> You are a helpful, respectful and honest assistant. <</SYS>>\n{prompt} [/INST]\n";
    let questions = [
        "Give me a 3 day travel plan for Seattle.",
        "Tell me something about University of Washington.",
        "How to dial in an espresso shot?",
    ];

    let devprops = device_query()?;
    let gpu_uuid = devprops[args.gpu_index].uuid;
    info!(gpu=?devprops[args.gpu_index]);
    let tokenizer = Tokenizer::new(&args.model_path)?;
    info!("Tokenizer loaded.");

    let (mut child, mut executor) = GpuExecutor::spawn(gpu_uuid)?;
    let wait_executor = tokio::spawn(async move {
        let ec = child.wait().await.unwrap();
        match ec.success() {
            true => info!("Executor exited normally"),
            false => error!(exitcode = %ec, "Executor exited."),
        }
        ec
    });
    info!("Executor spawned.");

    executor
        .init(
            &args.model_path.display().to_string(),
            "float16",
            16,
            (1024 * questions.len() / 16) as u32,
        )
        .await?;
    info!("Executor initialized.");

    let mut reqids = Vec::new();
    let mut output_ids = Vec::new();
    for question in &questions {
        let prompt = template.replace("{prompt}", question);
        let input_ids = tokenizer.encode(&prompt)?;
        output_ids.push(input_ids);
        let reqid = Uuid::now_v7();
        reqids.push(reqid);
        executor
            .add_request(
                reqid,
                output_ids.last().unwrap(),
                &comm::GenerationConfig {
                    min_tokens: 0,
                    max_tokens: 500,
                    max_new_tokens: 1024,
                    stop_token_id: tokenizer.eos_id(),
                    temperature: 0.7,
                    repetition_penalty: 1.1,
                    top_p: 0.9,
                },
            )
            .await?;
        info!(reqid=%reqid, "Added request.");
    }

    let ret = executor.batch_prefill(&reqids).await?;
    for (o, r) in output_ids.iter_mut().zip(ret.token_ids) {
        o.push(r);
    }
    info!("batch_prefill done.");

    let mut workset: Vec<_> = (0..questions.len()).collect();
    while !workset.is_empty() {
        let ret = executor
            .batch_decode(
                &workset.iter().map(|i| reqids[*i]).collect::<Vec<_>>(),
            )
            .await?;

        let mut new_workset = Vec::new();
        for i in workset {
            output_ids[i].push(ret.token_ids[i]);
            if ret.finish_reasons[i] == comm::FinishReason::NotFinished {
                new_workset.push(i);
            } else {
                info!(idx=%i, reason=?ret.finish_reasons[i], "Request finished.");
                let text = tokenizer.decode(&output_ids[i])?;
                println!("{}", text);
            }
        }
        workset = new_workset;
    }
    info!("All requests finished.");

    executor.shutdown().await?;
    info!("Sent shutdown to executor.");

    let ec = wait_executor.await?;
    match ec.success() {
        true => Ok(()),
        false => Err(anyhow::anyhow!("Executor exited with {}", ec)),
    }
}
