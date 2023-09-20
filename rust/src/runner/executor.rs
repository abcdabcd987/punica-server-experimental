use std::process::Stdio;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use uuid::Uuid;

use crate::comm;

#[derive(Serialize, Debug)]
#[serde(tag = "t", content = "c")]
enum Request<'a> {
    Init(Init<'a>),
    Shutdown(Shutdown),
    AddRequest(AddRequest<'a>),
    CancelRequest(CancelRequest),
    BatchPrefill(BatchPrefill<'a>),
    BatchDecode(BatchDecode<'a>),
}

#[derive(Serialize, Debug)]
struct Init<'a> {
    model_path: &'a str,
    dtype_str: &'a str,
    block_len: u32,
    kvpool_capacity: u32,
}

#[derive(Serialize, Debug)]
struct Shutdown {}

#[derive(Serialize, Debug)]
struct AddRequest<'a> {
    reqid: Uuid,
    input_ids: &'a [u32],
    gencfg: &'a comm::GenerationConfig,
}

#[derive(Serialize, Debug)]
struct CancelRequest {
    reqid: Uuid,
}

#[derive(Serialize, Debug)]
struct BatchPrefill<'a> {
    reqids: &'a [Uuid],
}

#[derive(Serialize, Debug)]
struct BatchDecode<'a> {
    reqids: &'a [Uuid],
}

#[derive(Deserialize, Debug)]
pub struct TextGenerationChunkResponse {
    pub token_ids: Vec<u32>,
    pub finish_reasons: Vec<comm::FinishReason>,
}

pub struct GpuExecutor {
    stdin: ChildStdin,
    stdout: ChildStdout,
}

impl GpuExecutor {
    pub fn spawn(gpu_uuid: Uuid) -> anyhow::Result<(Child, Self)> {
        let mut child = Command::new("python")
            .args(["-m", "punica_runner.gpu_executor"])
            .env("CUDA_VISIBLE_DEVICES", format!("GPU-{}", gpu_uuid))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        Ok((child, Self { stdin, stdout }))
    }

    async fn write_msg(&mut self, msg: &Request<'_>) -> anyhow::Result<()> {
        let bin = rmp_serde::to_vec_named(msg).unwrap();
        let len = bin.len() as u32;
        self.stdin.write_u32_le(len).await?;
        self.stdin.write_all(&bin).await?;
        Ok(())
    }

    async fn read_msg<T: DeserializeOwned>(&mut self) -> anyhow::Result<T> {
        let len = self.stdout.read_u32_le().await?;
        let mut buf = vec![0; len as usize];
        self.stdout.read_exact(&mut buf).await?;
        let res: Result<T, String> = rmp_serde::from_slice(&buf)?;
        match res {
            Ok(res) => Ok(res),
            Err(err) => Err(anyhow::anyhow!(err)),
        }
    }

    pub async fn init(
        &mut self,
        model_path: &str,
        dtype_str: &str,
        block_len: u32,
        kvpool_capacity: u32,
    ) -> anyhow::Result<()> {
        let init = Request::Init(Init {
            model_path,
            dtype_str,
            block_len,
            kvpool_capacity,
        });
        self.write_msg(&init).await?;
        self.read_msg::<i32>().await?;
        Ok(())
    }

    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        let shutdown = Request::Shutdown(Shutdown {});
        self.write_msg(&shutdown).await
    }

    pub async fn add_request(
        &mut self,
        reqid: Uuid,
        input_ids: &[u32],
        gencfg: &comm::GenerationConfig,
    ) -> anyhow::Result<()> {
        let add_request =
            Request::AddRequest(AddRequest { reqid, input_ids, gencfg });
        self.write_msg(&add_request).await?;
        self.read_msg::<i32>().await?;
        Ok(())
    }

    pub async fn cancel_request(&mut self, reqid: Uuid) -> anyhow::Result<()> {
        let cancel_request = Request::CancelRequest(CancelRequest { reqid });
        self.write_msg(&cancel_request).await?;
        self.read_msg::<i32>().await?;
        Ok(())
    }

    pub async fn batch_prefill(
        &mut self,
        reqids: &[Uuid],
    ) -> anyhow::Result<TextGenerationChunkResponse> {
        let batch_prefill = Request::BatchPrefill(BatchPrefill { reqids });
        self.write_msg(&batch_prefill).await?;
        self.read_msg().await
    }

    pub async fn batch_decode(
        &mut self,
        reqids: &[Uuid],
    ) -> anyhow::Result<TextGenerationChunkResponse> {
        let batch_decode = Request::BatchDecode(BatchDecode { reqids });
        self.write_msg(&batch_decode).await?;
        self.read_msg().await
    }
}
