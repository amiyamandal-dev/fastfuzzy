extern crate strsim;

use std::borrow::BorrowMut;
use std::collections::HashMap;

use ngrams::Ngram;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use strsim::{generic_levenshtein, levenshtein, normalized_levenshtein};

use crate::utils::{StringProcessing, thread_process};
use std::time::{Duration, Instant};

#[pyclass]
pub struct Levenshtein {
    re_obj: StringProcessing
}

#[pymethods]
impl Levenshtein {
    #[new]
    pub fn new() -> Self {
        let obj = StringProcessing::new();
        Levenshtein {
            re_obj: obj,
        }
    }

    pub fn match_string_difference(&mut self, source: &PyString, target: &PyString) -> usize {
        let s = source.to_string_lossy().to_string();
        let t = target.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let t_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(t);
        levenshtein(s_p.trim(), t_p.trim())
    }

    pub fn match_string_percentage(&mut self, source: &PyString, target: &PyString) -> f64 {
        let s = source.to_string_lossy().to_string();
        let t = target.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let t_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(t);
        normalized_levenshtein(s_p.trim(), t_p.trim())
    }

    pub fn generic_damerau_levenshtein_process(&mut self, source: &PyString, target: &PyString) {
        // TODO:- implement this code
        // let s = source.to_string_lossy().to_string().split(" ");
        // let t = target.to_string_lossy().to_string().split(" ");
        // generic_damerau_levenshtein(&s, &t)
    }
    pub fn match_string_difference_list(&mut self, source: &PyString, target: &PyList) -> HashMap<String, usize> {
        let s = source.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let mut temp_vec: Vec<String> = Vec::new();
        for i in target.iter() {
            let k = match i.extract() {
                Ok(val) => {
                    val
                }
                Err(why) => {
                    panic!("{:}", why)
                }
            };
            temp_vec.push(self.re_obj.replace_non_letters_non_numbers_with_whitespace(k));
        }
        let mut rez_vec: HashMap<String, usize> = HashMap::new();
        for i in temp_vec.iter() {
            let z = levenshtein(s_p.trim(), i.as_str().trim());
            rez_vec.insert(i.clone(), z);
        }
        rez_vec
    }

    pub fn match_string_percentage_list(&mut self, source: &PyString, target: Vec<String>) -> HashMap<String, f64> {
        let s = source.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let mut temp_vec: Vec<String> = target.par_iter().map(|i| {
            self.re_obj.replace_non_letters_non_numbers_with_whitespace(i.to_string())
        }).collect();
        let mut rez_vec: HashMap<String, f64> = HashMap::new();

        for i in temp_vec.iter() {
            let z = normalized_levenshtein(s_p.trim(), i.as_str().trim());
            rez_vec.insert(i.clone(), z);
        }
        rez_vec
    }

    pub fn match_string_percentage_list_th(&mut self, source: &PyString, target: &PyList) -> HashMap<String, f64> {
        let s = source.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let mut temp_vec: Vec<String> = Vec::new();
        for i in target.iter() {
            let k = match i.extract() {
                Ok(val) => {
                    val
                }
                Err(why) => {
                    panic!("{:}", why);
                    String::new()
                }
            };
            temp_vec.push(self.re_obj.replace_non_letters_non_numbers_with_whitespace(k));
        }
        thread_process(s_p, temp_vec)
        // rez_vec
    }

    pub fn search_in_blob_text(&mut self, source: &PyString, target: &PyString, matching_percentage: f64) -> Vec<String> {
        /*
        this will search in blob of text
        */
        let s = source.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let t = target.to_string_lossy().to_string();
        let t_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(t);
        let len_val: Vec<_> = s_p.split(" ").collect();
        let grams: Vec<_> = t_p.split(' ').ngrams(len_val.len()).collect();
        let mut temp_str: Vec<_> = vec![];

        for i in grams.iter() {
            let mut t = "".to_string();
            for j in i.to_vec() {
                t.push_str(j);
                t.push_str(" ");
            }
            t = t.trim().to_string();
            temp_str.push(t);
        }
        let mut matched_val: Vec<_> = vec![];
        for i in temp_str.iter() {
            let z = normalized_levenshtein(s_p.trim(), i.as_str().trim());
            if z >= matching_percentage {
                matched_val.push(i.clone());
            }
        }
        matched_val
    }
}

#[pyfunction]
pub fn match_string_percentage_list_fn(py: Python, source: &PyString, target: Vec<String>) -> PyResult<HashMap<String, f64>> {
    let s = source.extract().unwrap();
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
    Ok(rez_vec)
}

#[pyfunction]
pub fn match_string_percentage_fn(py: Python,source: &PyString, target: &PyString) -> f64 {
    let s = source.to_string_lossy().to_string();
    let t = target.to_string_lossy().to_string();
    let obj = StringProcessing::new();
    // let s_p = obj.replace_non_letters_non_numbers_with_whitespace(s);
    // let t_p = obj.replace_non_letters_non_numbers_with_whitespace(t);
    normalized_levenshtein(s.trim(), t.trim())
}

#[pymodule]
pub fn basic_levenshtein(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Levenshtein>()?;
    m.add_function(wrap_pyfunction!(match_string_percentage_list_fn, m)?)?;
    m.add_function(wrap_pyfunction!(match_string_percentage_fn, m)?)?;
    Ok(())
}
