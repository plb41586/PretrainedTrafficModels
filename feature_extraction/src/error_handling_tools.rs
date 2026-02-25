use std::error::Error;

pub fn print_error_chain(e: &dyn Error) {
    eprintln!("Error: {}", e);
    let mut source = e.source();
    while let Some(s) = source {
        eprintln!("Caused by: {}", s);
        source = s.source();
    }
}