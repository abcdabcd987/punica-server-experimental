#[macro_use(info, error)]
extern crate tracing;

mod comm;
mod runner;
mod scheduler;

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
    DevHello {},
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "punica=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    match cli.command {
        Commands::Scheduler(args) => {
            rt.block_on(scheduler::scheduler_main(args))?
        }
        Commands::DevHello {} => rt.block_on(async { runner::hello_main() })?,
    }

    Ok(())
}
