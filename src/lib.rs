use pyo3::Python;
use pyo3::prelude::*;

pub mod word_count;
pub mod utils;
pub mod tokenizer;
pub mod damerau;
pub mod basic_levenshtein;


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
