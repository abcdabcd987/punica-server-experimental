use std::net::SocketAddr;
use std::ops::ControlFlow;
use std::sync::Arc;

use dashmap::DashMap;
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::{Error as WsError, Message};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};
use tokio_util::sync::CancellationToken;
use url::Url;
use uuid::Uuid;

use crate::utils::get_ws_peer_addr;

pub type WebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;
use crate::comm;

#[derive(Clone)]
pub struct SchedulerClient {
    requests: Arc<DashMap<Uuid, RequestContext>>,
    tx: mpsc::UnboundedSender<Message>,
}

impl SchedulerClient {
    pub fn add_textgen(
        &self,
        request_id: Uuid,
        input_ids: Vec<u32>,
        gencfg: comm::GenerationConfig,
        chunk_tx: mpsc::UnboundedSender<comm::TextGenChunk>,
    ) {
        self.requests.insert(request_id, RequestContext { chunk_tx });
        self.send(&comm::FrontendToSchedulerMessage::TextGenRequest(
            comm::TextGenRequest { request_id, input_ids, gencfg },
        ));
    }

    pub fn cancel_textgen(&self, request_id: Uuid) {
        self.requests.remove(&request_id);
        self.send(&comm::FrontendToSchedulerMessage::CancelTextGen(
            comm::CancelTextGen { request_id },
        ));
    }

    fn send(&self, msg: &comm::FrontendToSchedulerMessage) {
        let binary =
            postcard::to_stdvec(msg).expect("Failed to serialize message");
        self.tx.send(Message::Binary(binary)).unwrap();
    }
}

struct RequestContext {
    chunk_tx: mpsc::UnboundedSender<comm::TextGenChunk>,
}

pub struct SchedulerConnection {
    url: Url,
    addr: SocketAddr,
    ws_send: SplitSink<WebSocket, Message>,
    ws_recv: SplitStream<WebSocket>,
    ch_send: mpsc::UnboundedSender<Message>,
    ch_recv: mpsc::UnboundedReceiver<Message>,
    requests: Arc<DashMap<Uuid, RequestContext>>,
    shutdown: CancellationToken,
    _shutdown_complete: mpsc::Sender<()>,
}

impl SchedulerConnection {
    pub fn new(
        url: Url,
        ws: WebSocket,
        shutdown: CancellationToken,
        shutdown_complete: mpsc::Sender<()>,
    ) -> Self {
        let addr = get_ws_peer_addr(&ws);
        info!(url=%url, addr=%addr, "Connected to scheduler.");
        let (ws_send, ws_recv) = ws.split();
        let (ch_send, ch_recv) = mpsc::unbounded_channel();
        let requests = Arc::new(DashMap::new());
        Self {
            url,
            addr,
            ws_send,
            ws_recv,
            ch_send,
            ch_recv,
            requests,
            shutdown,
            _shutdown_complete: shutdown_complete,
        }
    }

    pub fn get_client(&self) -> SchedulerClient {
        SchedulerClient {
            requests: self.requests.clone(),
            tx: self.ch_send.clone(),
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
                self.handle_scheduler_message(postcard::from_bytes(&m)?);
                Ok(ControlFlow::Continue(()))
            }
        }
    }

    fn handle_scheduler_message(
        &mut self,
        msg: comm::SchedulerToFrontendMessage,
    ) {
        use comm::SchedulerToFrontendMessage::*;
        match msg {
            TextGenChunk(msg) => {
                let ctx = match self.requests.get(&msg.request_id) {
                    Some(ctx) => ctx,
                    None => {
                        error!(request_id=%msg.request_id, "Got TextGenChunk for unknown request_id.");
                        return;
                    }
                };
                ctx.chunk_tx.send(msg.clone()).unwrap();
                drop(ctx);
                if msg.finish_reason != comm::FinishReason::NotFinished {
                    self.requests.remove(&msg.request_id);
                }
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
