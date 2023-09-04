use std::net::SocketAddr;

use anyhow::Context;
use futures::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};
use url::Url;

use crate::comm;
use crate::utils::get_ws_peer_addr;

pub struct SchedulerConnection {
    url: Url,
    addr: SocketAddr,
    ws: WebSocketStream<MaybeTlsStream<TcpStream>>,
}

pub enum SchedulerMessage {
    End,
    Alive,
    Message(comm::SchedulerToRunnerMessage),
}

impl SchedulerConnection {
    pub async fn connect_to_scheduler(url: Url) -> anyhow::Result<Self> {
        info!("Connecting to scheduler: {}", url);
        let (mut ws, _) =
            tokio_tungstenite::connect_async(&url).await.with_context(
                || format!("Failed to connect to scheduler: {}", url),
            )?;
        let addr = get_ws_peer_addr(&ws);

        ws.send(Message::Binary(
            postcard::to_stdvec(&comm::HelloScheduler {
                node_type: comm::NodeType::Runner,
            })
            .unwrap(),
        ))
        .await
        .with_context(|| {
            format!("Failed to send HelloScheduler. scheduler: {}", url)
        })?;

        info!("Connected to scheduler. url: {}, addr: {}", url, addr);
        Ok(Self { url, addr, ws })
    }

    pub async fn send_message(
        &mut self,
        msg: comm::RunnerToSchedulerMessage,
    ) -> anyhow::Result<()> {
        let msg = postcard::to_allocvec(&msg).with_context(|| {
            format!(
                "Failed to serialize message to scheduler. url: {}, addr: {}",
                self.url, self.addr
            )
        })?;
        self.ws.send(Message::Binary(msg)).await?;
        Ok(())
    }

    pub async fn recv_message(&mut self) -> anyhow::Result<SchedulerMessage> {
        let msg = self.ws.next().await;
        match msg {
            Some(Ok(Message::Ping(_))) => Ok(SchedulerMessage::Alive),
            Some(Ok(Message::Pong(_))) => Ok(SchedulerMessage::Alive),
            Some(Ok(Message::Binary(msg))) => {
                let msg = postcard::from_bytes(&msg).with_context(
                    || format!("Failed to deserialize message from scheduler. url: {}, addr: {}", self.url, self.addr),
                )?;
                Ok(SchedulerMessage::Message(msg))
            }
            Some(Ok(Message::Close(_))) => {
                info!(
                    "Scheduler closed connection. url: {}, addr: {}",
                    self.url, self.addr
                );
                Ok(SchedulerMessage::End)
            }
            Some(Ok(Message::Text(_))) => Err(anyhow::anyhow!(
                "Scheduler sent unexpected text message. url: {}, addr: {}",
                self.url,
                self.addr
            )),
            Some(Ok(Message::Frame(_))) => Err(anyhow::anyhow!(
                "Scheduler sent unexpected frame. url: {}, addr: {}",
                self.url,
                self.addr
            )),
            Some(Err(e)) => Err(e.into()),
            None => Ok(SchedulerMessage::End),
        }
    }

    pub async fn close(&mut self) -> anyhow::Result<()> {
        self.ws.close(None).await.map_err(|e| e.into())
    }
}
