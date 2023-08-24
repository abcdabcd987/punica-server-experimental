use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use futures::stream::SplitStream;
use futures::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::WebSocketStream;
use tokio_util::sync::CancellationToken;
use tungstenite::Message;
use uuid::Uuid;

use crate::comm;

struct Scheduler {
    runners: Arc<Mutex<HashMap<Uuid, Runner>>>,
    gpus: HashMap<Uuid, Gpu>,
}

struct Runner {
    gpus: HashSet<Uuid>,
}

struct Gpu {}

impl Scheduler {
    fn handle_add_runner_request(&mut self, request: comm::AddRunnerRequest) {}
}

#[derive(Debug, clap::Args)]
pub struct SchedulerArgs {
    #[arg(long, default_value = "localhost:23081")]
    pub ws_bind: String,
}

pub async fn scheduler_main(
    args: SchedulerArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let listener =
        TcpListener::bind(&args.ws_bind).await.expect("Failed to bind");
    info!(
        "Scheduler websocket listening on {}",
        listener.local_addr().unwrap()
    );

    while let Ok((stream, _)) = listener.accept().await {
        tokio::spawn(accept_connection(stream));
    }

    Ok(())
}

async fn wait_hello_message(
    ws_stream: &mut WebSocketStream<TcpStream>,
) -> Option<comm::HelloScheduler> {
    let msg = loop {
        let msg = ws_stream.next().await?.ok()?;
        match msg {
            Message::Binary(msg) => break msg,
            _ => continue,
        }
    };
    rmp_serde::from_slice(&msg).ok()
}

struct Connection {
    shutdown: CancellationToken,
}

async fn accept_connection(stream: TcpStream) -> Result<(), ()> {
    let addr = stream
        .peer_addr()
        .expect("connected streams should have a peer address");
    log::info!("Peer address: {}", addr);
    let mut ws_stream = tokio_tungstenite::accept_async(stream)
        .await
        .expect("Error during the websocket handshake occurred");
    log::info!("New WebSocket connection: {}", addr);

    // handshake message
    let hello = wait_hello_message(&mut ws_stream).await.ok_or_else(|| {
        log::error!(
            "Closing connection because failed to receive hello message. Peer: {}",
            addr);
    })?;

    let (mut outgoing, incoming) = ws_stream.split();

    // Outgoing message channel
    let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<Message>(1);
    tokio::spawn(async move {
        while let Some(msg) = out_rx.recv().await {
            if let Err(e) = outgoing.send(msg).await {
                log::error!(
                    "Failed to send message to peer: {}. Error: {}",
                    addr,
                    e
                );
                out_rx.close();
                break;
            }
        }
    });

    // Ping
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            if let Err(e) = out_tx.send(Message::Ping(vec![])).await {
                break;
            }
        }
    });

    match hello.role {
        comm::ConnectorRole::Runner => handle_runner_message(incoming).await,
        comm::ConnectorRole::ApiServer => {
            todo!();
            Err(())
        }
        comm::ConnectorRole::Scheduler => {
            todo!();
            Err(())
        }
    }
}

async fn handle_runner_message(
    incoming: SplitStream<WebSocketStream<TcpStream>>,
) -> Result<(), ()> {
    while let Some(msg) = incoming.next().await {
        match msg.map_err(|e| log::error!("{}", e))? {
            Message::Binary(msg) => {
                let msg: comm::RunnerToSchedulerMessage =
                    rmp_serde::from_slice(&msg).unwrap();
                log::info!("Received: {:?}", msg);
            }
            Message::Ping(_) => {
                outgoing
                    .send(Message::Pong(vec![]))
                    .map_err(|e| log::error!("{}", e))
                    .await?
            }
            _ => continue,
        }
    }

    Ok(())
}
