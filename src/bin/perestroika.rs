use log::{debug, info, trace, warn};
use perestroika::{run_perestroika, setup_logging};

fn main() {
    setup_logging(1).expect("failed to initialize logging.");
    info!("Starting Perestroika...");
    run_perestroika();
}
