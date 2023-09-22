use std::path::PathBuf;

use uuid::Uuid;

use super::device_query::device_query;
use super::executor::ExecutorSubprocess;
use super::tokenizer::Tokenizer;
use crate::comm;

#[derive(Debug, clap::Args)]
pub struct DebugExecutorArgs {
    #[arg(long)]
    pub fake: bool,
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
        "How to grow tomatoes in a greenhouse?",
        "Tell me something about University of Washington.",
        "How to dial in an espresso shot? Please don't use emoji.",
    ];

    let gpu_uuid = if args.fake {
        Uuid::now_v7()
    } else {
        let devprops = device_query()?;
        info!(gpu=?devprops[args.gpu_index]);
        devprops[args.gpu_index].uuid
    };

    let tokenizer = Tokenizer::new(&args.model_path)?;
    info!("Tokenizer loaded.");
    let (mut child, mut executor) = ExecutorSubprocess::spawn(gpu_uuid)?;
    let wait_executor = tokio::spawn(async move {
        let ec = child.wait().await.unwrap();
        match ec.success() {
            true => info!("Executor exited normally"),
            false => error!(exitcode = %ec, "Executor exited."),
        }
        ec
    });
    info!("Executor spawned.");

    if args.fake {
        executor.init_fake().await?;
        info!("FakeGpuExecutor initialized.");
    } else {
        executor
            .init(
                &args.model_path.display().to_string(),
                "float16",
                16,
                (1024 * questions.len() / 16) as u32,
            )
            .await?;
        info!("GpuExecutor initialized.");
    }

    let mut reqids = Vec::new();
    let mut output_ids = Vec::new();
    for (idx, question) in questions.iter().enumerate() {
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
                    max_tokens: 2048,
                    max_new_tokens: 2048,
                    stop_token_id: tokenizer.eos_id(),
                    temperature: 0.7,
                    repetition_penalty: 1.1,
                    top_p: 0.9,
                },
            )
            .await?;
        info!(idx, %reqid, "Added request.");
    }

    let ret = executor.batch_prefill(&reqids).await?;
    for (o, r) in output_ids.iter_mut().zip(ret.token_ids) {
        o.push(r);
    }
    info!("batch_prefill done.");

    let mut workset: Vec<_> = (0..questions.len()).collect();
    let mut steps = 0;
    while !workset.is_empty() {
        let ret = executor
            .batch_decode(
                &workset.iter().map(|i| reqids[*i]).collect::<Vec<_>>(),
            )
            .await?;
        steps += 1;

        let mut new_workset = Vec::new();
        for ((new_token, finish), idx) in ret
            .token_ids
            .iter()
            .zip(ret.finish_reasons.iter())
            .zip(workset.iter())
        {
            output_ids[*idx].push(*new_token);
            let cancel = *idx == 1 && steps == 50;
            if cancel {
                executor.cancel_request(reqids[*idx]).await?;
                info!(idx, reqid = %reqids[*idx], "Request cancelled.");
            }
            let should_print = if *finish == comm::FinishReason::NotFinished {
                if !cancel {
                    new_workset.push(*idx);
                }
                false
            } else {
                info!(
                    idx,
                    reqid = %reqids[*idx],
                    finish_reason = ?finish,
                    "Request finished."
                );
                true
            };
            if should_print || cancel {
                let text = tokenizer.decode(&output_ids[*idx])?;
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
