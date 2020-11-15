pub mod word_count;
pub mod utils;
pub mod tokenizer;
pub mod damerau;
pub mod basic_levenshtein;
pub mod string_matcher;


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
