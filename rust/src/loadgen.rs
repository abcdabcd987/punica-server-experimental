use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context};
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use url::Url;
use uuid::Uuid;

use crate::comm;
use crate::scheduler_client::{SchedulerClient, SchedulerConnection};

#[derive(Debug, clap::Args)]
pub struct LoadGenArgs {
    #[arg(long, help = "wss://example.com/rpc")]
    pub scheduler_url: Url,
    #[arg(long, help = "Path to trace file. Use stdin if not specified.")]
    pub trace: Option<PathBuf>,
}

pub async fn loadgen_main(args: LoadGenArgs) -> anyhow::Result<()> {
    let trace = read_trace(args.trace.as_deref())?;

    let ct = CancellationToken::new();
    let (shutdown_complete_tx, mut shutdown_complete_rx) = mpsc::channel(1);
    let mut url = args.scheduler_url.clone();
    url.path_segments_mut().unwrap().push("v1").push("frontend");
    let (ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .with_context(|| format!("Failed to connect to scheduler: {}", url))?;

    let conn =
        SchedulerConnection::new(url, ws, ct.clone(), shutdown_complete_tx);
    let scheduler_client = conn.get_client();
    let conn = tokio::spawn(async move { conn.serve().await });

    let ec = tokio::select! {
        _ = conn => {
            Err(anyhow::anyhow!("Scheduler connection closed unexpectedly."))
        }
        _ = loadgen(trace, scheduler_client) => {
            info!("LoadGen finished. Shutting down...");
            Ok(())
        }
    };

    ct.cancel();
    let _ = shutdown_complete_rx.recv().await;

    ec
}

#[derive(Clone)]
struct RequestSpec {
    gap_f32: f32,
    prompt_len: u32,
    output_len: u32,
}

type TraceSpec = Vec<RequestSpec>;

fn parse_trace<R: BufRead>(reader: R) -> anyhow::Result<TraceSpec> {
    let mut ret = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let mut parts = line.split_whitespace();
        let gap_f32 = parts.next().ok_or_else(|| anyhow!("Missing gap_f32"))?;
        let gap_f32 = gap_f32.parse::<f32>().context("gap_f32")?;
        let prompt_len =
            parts.next().ok_or_else(|| anyhow!("Missing prompt_len"))?;
        let prompt_len = prompt_len.parse::<u32>().context("prompt_len")?;
        let output_len =
            parts.next().ok_or_else(|| anyhow!("Missing output_len"))?;
        let output_len = output_len.parse::<u32>().context("output_len")?;
        ret.push(RequestSpec { gap_f32, prompt_len, output_len });
    }

    Ok(ret)
}

fn read_trace(path: Option<&Path>) -> anyhow::Result<TraceSpec> {
    let reader: Box<dyn BufRead> = match path {
        Some(path) => Box::new(std::io::BufReader::new(
            std::fs::File::open(path).with_context(|| {
                format!("Failed to open trace file: {}", path.display())
            })?,
        )),
        None => Box::new(std::io::BufReader::new(std::io::stdin())),
    };
    parse_trace(reader)
}

async fn loadgen(trace: TraceSpec, scheduler: SchedulerClient) {
    println!("start");
    let start_at = Instant::now();
    let start_at_nanos = chrono::Utc::now().timestamp_nanos();
    let mut joinset = tokio::task::JoinSet::new();
    let mut offset_f32 = 0f32;
    for (reqidx, reqspec) in trace.iter().enumerate() {
        offset_f32 += reqspec.gap_f32;
        let send_at = start_at + Duration::from_secs_f32(offset_f32);
        tokio::time::sleep_until(send_at).await;
        joinset.spawn(loadgen_request(
            scheduler.clone(),
            reqidx,
            reqspec.clone(),
            start_at,
            start_at_nanos,
        ));
    }
    while let Some(ret) = joinset.join_next().await {
        ret.unwrap();
    }
}

async fn loadgen_request(
    scheduler: SchedulerClient,
    reqidx: usize,
    reqspec: RequestSpec,
    start_at: Instant,
    start_at_nanos: i64,
) {
    let reqid = Uuid::from_u64_pair(start_at_nanos as u64, reqidx as u64);
    let input_ids = vec![1000u32; reqspec.prompt_len as usize];
    let total_len = reqspec.prompt_len + reqspec.output_len;
    let gencfg = comm::GenerationConfig {
        min_tokens: total_len,
        max_tokens: total_len,
        max_new_tokens: total_len,
        stop_token_id: 32000,
        temperature: 0.0,
        repetition_penalty: 1.0,
        top_p: 1.0,
    };
    let (tx, mut rx) = mpsc::unbounded_channel();

    scheduler.add_textgen(reqid, input_ids, gencfg, tx);

    let mut cnt = 0;
    while let Some(_chunk) = rx.recv().await {
        let elapsed = start_at.elapsed().as_secs_f32();
        println!("{:.9} {}", elapsed, reqidx);
        cnt += 1;
    }
    if cnt != reqspec.output_len {
        panic!(
            "Expected {} chunks, got {}. reqidx: {}",
            reqspec.output_len, cnt, reqidx
        );
    }
}
