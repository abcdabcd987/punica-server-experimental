use std::net::SocketAddr;

use tokio::net::TcpStream;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

pub fn get_ws_peer_addr(
    ws: &WebSocketStream<MaybeTlsStream<TcpStream>>,
) -> SocketAddr {
    match ws.get_ref() {
        MaybeTlsStream::Plain(s) => s,
        MaybeTlsStream::Rustls(s) => s.get_ref().0,
        _ => unreachable!(),
    }
    .peer_addr()
    .unwrap()
}
