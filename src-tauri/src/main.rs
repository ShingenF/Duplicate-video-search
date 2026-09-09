#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    if let Some(exit_code) = duplicate_video_search_lib::ram_disk::handle_elevated_arguments() {
        std::process::exit(exit_code);
    }
    duplicate_video_search_lib::run();
}
