use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use tokio::sync::mpsc;
use tower::ServiceBuilder;
use tower_http::trace::{DefaultMakeSpan, TraceLayer};
use tower_http::ServiceBuilderExt;
use uuid::Uuid;

use super::api;
use crate::comm;
use crate::scheduler_client::SchedulerClient;
use crate::tokenizer::Tokenizer;

struct HttpServerContext {
    scheduler: SchedulerClient,
    tokenizer: Tokenizer,
}

pub async fn run_http(
    bind: SocketAddr,
    http_handle: axum_server::Handle,
    _shutdown_complete: mpsc::Sender<()>,
    scheduler: SchedulerClient,
    tokenizer: Tokenizer,
) {
    let service = ServiceBuilder::new().catch_panic().layer(
        TraceLayer::new_for_http()
            .make_span_with(DefaultMakeSpan::default().include_headers(true)),
    );
    let ctx = Arc::new(HttpServerContext { scheduler, tokenizer });
    let app = Router::new()
        .route("/textgen", post(textgen_handler))
        .layer(service)
        .with_state(ctx);

    let server = axum_server::bind(bind);
    info!("Started HTTP server on {}", bind);
    let ret = server
        .handle(http_handle)
        .serve(app.into_make_service_with_connect_info::<SocketAddr>())
        .await;
    if let Err(e) = ret {
        error!("HTTP server error: {}", e);
    }
    info!("HTTP server stopped");
}

async fn textgen_handler(
    State(ctx): State<Arc<HttpServerContext>>,
    Json(req): Json<api::TextGenRequest>,
) -> impl IntoResponse {
    let reqid = Uuid::now_v7();
    let gencfg = comm::GenerationConfig {
        min_tokens: req.min_tokens.unwrap_or(0),
        max_tokens: req.max_tokens.unwrap_or(4096),
        max_new_tokens: req.max_new_tokens.unwrap_or(4096),
        stop_token_id: ctx.tokenizer.eos_id(),
        temperature: req.temperature.unwrap_or(0.7),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.1),
        top_p: req.top_p.unwrap_or(0.9),
    };
    const TEMPLATE: &str = "[INST] <<SYS>> You are a helpful, respectful and honest assistant. <</SYS>>\n{prompt} [/INST]\n";
    let template = req.template.as_deref().unwrap_or(TEMPLATE);
    let prompt = template.replace("{prompt}", &req.prompt);

    let input_ids = match ctx.tokenizer.encode(&prompt) {
        Ok(v) => v,
        Err(e) => {
            error!(cause=%e, "Failed to encode prompt");
            return (
                StatusCode::BAD_REQUEST,
                Json(api::Error {
                    message: format!("Failed to encode prompt: {}", e),
                }),
            )
                .into_response();
        }
    };
    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut output_ids = input_ids.clone();
    let mut prefix_offset = 0;
    let mut read_offset = input_ids.len();
    ctx.scheduler.add_textgen(reqid, input_ids, gencfg, tx);

    struct DisconnectGuard {
        scheduler: SchedulerClient,
        reqid: Uuid,
        finished: bool,
    }

    impl Drop for DisconnectGuard {
        fn drop(&mut self) {
            if !self.finished {
                self.scheduler.cancel_textgen(self.reqid);
            }
        }
    }

    let stream = async_stream::stream! {
    let mut guard = DisconnectGuard {
        scheduler: ctx.scheduler.clone(),
        reqid,
        finished: false,
    };

    while let Some(chunk) = rx.recv().await {
        let finish_reason = match chunk.finish_reason {
            comm::FinishReason::NotFinished => None,
            comm::FinishReason::Stop => Some(api::FinishReason::Stop),
            comm::FinishReason::Length => Some(api::FinishReason::Length),
            comm::FinishReason::Error => Some(api::FinishReason::Error),
        };

        output_ids.push(chunk.token_id);
        let prefix_text =
            ctx.tokenizer.decode(&output_ids[prefix_offset..read_offset]);
        let new_text = ctx.tokenizer.decode(&output_ids[prefix_offset..]);
        let mut ok = false;
        if let (Ok(prefix_text), Ok(new_text)) = (prefix_text, new_text) {
            if new_text.len() > prefix_text.len() {
                ok = true;
                let text = new_text[prefix_text.len()..].to_owned();
                prefix_offset = read_offset;
                read_offset = output_ids.len();
                yield Event::default()
                    .json_data(api::TextGenChunk { text, finish_reason });
            }
        }
        if !ok && finish_reason.is_some() {
            yield Event::default().json_data(api::TextGenChunk {
                text: "".to_owned(),
                finish_reason,
            });
        }

        if finish_reason.is_some() {
            break;
        }
    }
    guard.finished = true;
    };

    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}
