use std::net::SocketAddr;
use std::ops::ControlFlow;
use std::sync::Arc;

use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::{Error as WsError, Message};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};
use tokio_util::sync::CancellationToken;
use url::Url;

use super::runner::Runner;
use crate::utils::get_ws_peer_addr;

pub type WebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

pub struct SchedulerConnection {
    url: Url,
    addr: SocketAddr,
    ws_send: SplitSink<WebSocket, Message>,
    ws_recv: SplitStream<WebSocket>,
    ch_recv: mpsc::Receiver<Message>,
    shutdown: CancellationToken,
    _shutdown_complete: mpsc::Sender<()>,
    runner: Arc<Runner>,
}

impl SchedulerConnection {
    pub fn new(
        url: Url,
        ws: WebSocket,
        ch_recv: mpsc::Receiver<Message>,
        shutdown: CancellationToken,
        shutdown_complete: mpsc::Sender<()>,
        runner: Arc<Runner>,
    ) -> Self {
        let addr = get_ws_peer_addr(&ws);
        let (ws_send, ws_recv) = ws.split();
        Self {
            url,
            addr,
            ws_send,
            ws_recv,
            ch_recv,
            shutdown,
            _shutdown_complete: shutdown_complete,
            runner,
        }
    }

    pub async fn serve(mut self) {
        match self.run().await {
            Ok(()) => {
                info!(url=%self.url, addr=%self.addr, "Connection closed.")
            }
            Err(e) => {
                error!(url=%self.url, addr=%self.addr, cause=%e, "Connection error. Connection closed.");
            }
        }
        let mut ws = self.ws_send.reunite(self.ws_recv).unwrap();
        let _ = ws.close(None).await;
    }

    async fn run(&mut self) -> anyhow::Result<()> {
        while !self.shutdown.is_cancelled() {
            let ret = tokio::select! {
                msg = self.ws_recv.next() => {
                    self.handle_message(msg).await?
                },
                recv = self.ch_recv.recv() => {
                    self.forward_message(recv).await?
                },
                _ = self.shutdown.cancelled() => ControlFlow::Break(()),
            };
            match ret {
                ControlFlow::Continue(()) => (),
                ControlFlow::Break(()) => break,
            }
        }
        Ok(())
    }

    async fn handle_message(
        &mut self,
        msg: Option<Result<Message, WsError>>,
    ) -> anyhow::Result<ControlFlow<()>> {
        match msg {
            Some(Ok(Message::Text(_))) => {
                Err(anyhow::anyhow!("Unexpected text message"))
            }
            Some(Ok(Message::Ping(_))) => {
                // Nothing to do. Handled by websocket library.
                Ok(ControlFlow::Continue(()))
            }
            Some(Ok(Message::Pong(_))) => {
                // Nothing to do.
                Ok(ControlFlow::Continue(()))
            }
            Some(Ok(Message::Frame(_))) => Ok(ControlFlow::Break(())),
            Some(Ok(Message::Close(_))) => Ok(ControlFlow::Break(())),
            Some(Err(e)) => Err(e.into()),
            None => Ok(ControlFlow::Continue(())),
            Some(Ok(Message::Binary(m))) => {
                self.runner
                    .handle_scheduler_message(postcard::from_bytes(&m)?)
                    .await?;
                Ok(ControlFlow::Continue(()))
            }
        }
    }

    async fn forward_message(
        &mut self,
        recv: Option<Message>,
    ) -> anyhow::Result<ControlFlow<()>> {
        match recv {
            Some(msg) => {
                self.ws_send.send(msg).await?;
                Ok(ControlFlow::Continue(()))
            }
            None => Ok(ControlFlow::Break(())),
        }
    }
}
