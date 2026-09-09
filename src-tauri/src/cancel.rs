use anyhow::anyhow;
use std::sync::atomic::{AtomicBool, Ordering};

static CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);

pub fn reset() {
    CANCEL_REQUESTED.store(false, Ordering::SeqCst);
}

pub fn request() {
    CANCEL_REQUESTED.store(true, Ordering::SeqCst);
}

pub fn is_requested() -> bool {
    CANCEL_REQUESTED.load(Ordering::SeqCst)
}

pub fn bail_if_requested() -> anyhow::Result<()> {
    if is_requested() {
        Err(anyhow!("operation cancelled"))
    } else {
        Ok(())
    }
}
