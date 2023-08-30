use std::collections::VecDeque;
use std::io::Write;
use std::time::{Duration, SystemTime};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};

struct OnlineStats {
    cnt: u32,
    sum: f32,
    sqr: f32,
}

impl OnlineStats {
    fn new() -> OnlineStats {
        OnlineStats { cnt: 0, sum: 0f32, sqr: 0f32 }
    }

    fn add(&mut self, x: f32) {
        self.cnt += 1;
        self.sum += x;
        self.sqr += x * x;
    }

    fn del(&mut self, x: f32) {
        self.cnt -= 1;
        self.sum -= x;
        self.sqr -= x * x;
    }

    fn mean(&self) -> f32 {
        self.sum / self.cnt as f32
    }

    fn std(&self) -> f32 {
        (self.sqr / self.cnt as f32 - (self.sum / self.cnt as f32).powi(2))
            .sqrt()
    }
}

async fn handle_uds_conn(mut stream: UnixStream) -> anyhow::Result<()> {
    let basetime_ns = 1693420991u64 * 1_000_000_000;
    let basetime = SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_nanos(basetime_ns))
        .unwrap();
    let mut send_buf = vec![0u8; 1000];
    let mut recv_buf = vec![0u8; 4096];
    let mut latencies = VecDeque::<(f32, f32)>::with_capacity(1000);
    let mut s2c_stats = OnlineStats::new();
    let mut c2s_stats = OnlineStats::new();
    let mut last_print = 0u64;

    println!("New connection");
    loop {
        let sendtime_ns =
            SystemTime::now().duration_since(basetime).unwrap().as_nanos()
                as u64;
        send_buf[0..8].copy_from_slice(&sendtime_ns.to_le_bytes());
        stream.write_all(&(send_buf.len() as u32).to_le_bytes()).await?;
        stream.write_all(&send_buf).await?;

        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await?;
        let recv_len = u32::from_le_bytes(len_buf);
        stream.read_exact(&mut recv_buf[0..recv_len as usize]).await?;
        let recvtime_ns =
            SystemTime::now().duration_since(basetime).unwrap().as_nanos()
                as u64;
        let client_recv_time_ns =
            u64::from_le_bytes(recv_buf[0..8].try_into().unwrap());
        let client_send_time_ns =
            u64::from_le_bytes(recv_buf[8..16].try_into().unwrap());

        let s2c_us = (client_recv_time_ns - sendtime_ns) as f32 / 1e3;
        let c2s_us = (recvtime_ns - client_send_time_ns) as f32 / 1e3;
        if latencies.len() == latencies.capacity() {
            let (s2c_us, c2s_us) = latencies.pop_front().unwrap();
            s2c_stats.del(s2c_us);
            c2s_stats.del(c2s_us);
        }
        latencies.push_back((s2c_us, c2s_us));
        s2c_stats.add(s2c_us);
        c2s_stats.add(c2s_us);

        if recvtime_ns - last_print > 50_000_000 {
            last_print = recvtime_ns;
            print!(
                "\x1b[2K\rs2c: {:.3} +/- {:.3} us; c2s: {:.3} +/- {:.3} us; rtt: {:.3} +/- {:.3} us",
                s2c_stats.mean(),
                s2c_stats.std(),
                c2s_stats.mean(),
                c2s_stats.std(),
                s2c_stats.mean() + c2s_stats.mean(),
                (s2c_stats.std().powi(2) + c2s_stats.std().powi(2)).sqrt()
            );
            std::io::stdout().flush().unwrap();
        }
    }
}

async fn uds_server() -> anyhow::Result<()> {
    let stream = UnixListener::bind("\0uds-ipc-benchmark")?;
    loop {
        let (socket, _) = stream.accept().await?;
        tokio::spawn(handle_uds_conn(socket));
    }
}

fn main() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(uds_server())
}
