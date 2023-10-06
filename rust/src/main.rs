#[macro_use(debug, info, warn, error)]
extern crate tracing;

mod comm;
mod frontend;
mod loadgen;
mod model_config;
mod runner;
mod scheduler;
mod scheduler_client;
mod tokenizer;
mod utils;

use clap::Parser;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Debug, Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, clap::Subcommand)]
enum Commands {
    Scheduler(scheduler::SchedulerArgs),
    Runner(runner::RunnerArgs),
    Frontend(frontend::FrontendArgs),
    Loadgen(loadgen::LoadGenArgs),
    DebugExecutor(runner::DebugExecutorArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "punica=debug".into()),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .compact()
                .with_file(true)
                .with_line_number(true)
                .with_target(false)
                .with_writer(std::io::stderr),
        )
        .init();

    let amain = async {
        match cli.command {
            Commands::Scheduler(args) => scheduler::scheduler_main(args).await,
            Commands::Runner(args) => runner::runner_main(args).await,
            Commands::Frontend(args) => frontend::frontend_main(args).await,
            Commands::Loadgen(args) => loadgen::loadgen_main(args).await,
            Commands::DebugExecutor(args) => {
                runner::debug_executor_main(args).await
            }
        }
    };

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(amain)
}
