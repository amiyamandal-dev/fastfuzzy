use std::collections::HashMap;

use criterion::{black_box, Criterion, criterion_group, criterion_main};
use rayon::prelude::*;
use strsim::normalized_levenshtein;



fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}


use regex::Regex;

pub struct StringProcessing {
    re: Regex,
}

impl StringProcessing {
    pub(crate) fn new() -> Self {
        let k = Regex::new(r"(?ui)\W").unwrap();
        StringProcessing {
            re: k
        }
    }

    pub(crate) fn replace_non_letters_non_numbers_with_whitespace(&self, input_string: String) -> String {
        let rez = self.re.replace_all(&*input_string, " ");
        rez.to_string()
    }
}

pub fn match_string_percentage_list_fn(source: &str, target: Vec<String>) -> HashMap<String, f64> {
    let s = source.to_string();
    let obj = StringProcessing::new();
    let s_p = obj.replace_non_letters_non_numbers_with_whitespace(s);

    let mut temp_vec: Vec<String> = target.par_iter().map(|i| {
        obj.replace_non_letters_non_numbers_with_whitespace(i.to_string())
    }).collect();
    let mut rez_vec: HashMap<String, f64> = HashMap::new();

    for i in temp_vec.iter() {
        let z = normalized_levenshtein(s_p.trim(), i.as_str().trim());
        rez_vec.insert(i.clone(), z);
    }
    rez_vec
}


fn criterion_benchmark(c: &mut Criterion) {
    let s = "fuzzy wuzzy was a bear";
    let mut temp_vec: Vec<String> = vec![];
    for i in 0..100000 {
        temp_vec.push(s.to_string());
    }
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
