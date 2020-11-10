extern crate strsim;

use std::borrow::BorrowMut;

use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use pyo3::wrap_pyfunction;
use strsim::{damerau_levenshtein, generic_damerau_levenshtein, normalized_damerau_levenshtein};

use crate::utils::StringProcessing;

#[pyclass]
pub struct Damerau {
    re_obj: StringProcessing
}

#[pymethods]
impl Damerau {
    #[new]
    pub fn new() -> Self {
        let obj = StringProcessing::new();
        Damerau {
            re_obj: obj,
        }
    }

    pub fn match_string_difference(&mut self, source: &PyString, target: &PyString) -> usize {
        let s = source.to_string_lossy().to_string();
        let t = target.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let t_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(t);
        damerau_levenshtein(s_p.trim(), t_p.trim())
    }

    pub fn match_string_percentage(&mut self, source: &PyString, target: &PyString) -> f64 {
        let s = source.to_string_lossy().to_string();
        let t = target.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let t_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(t);
        normalized_damerau_levenshtein(s_p.trim(), t_p.trim())
    }

    pub fn generic_damerau_levenshtein_process(&mut self, source: &PyString, target: &PyString) {
        // TODO:- implement this code
        // let s = source.to_string_lossy().to_string().split(" ");
        // let t = target.to_string_lossy().to_string().split(" ");
        // generic_damerau_levenshtein(&s, &t)
    }
    pub fn match_string_difference_list(&mut self, source: &PyString, target: &PyList) -> Vec<usize> {
        let s = source.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let mut temp_vec: Vec<String> = Vec::new();
        for i in target.iter() {
            let k = match i.extract() {
                Ok(val) => {
                    val
                }
                Err(Why) => {
                    panic!("{:}", Why);
                    String::new()
                }
            };
            temp_vec.push(self.re_obj.replace_non_letters_non_numbers_with_whitespace(k));
        }
        let mut rez_vec: Vec<usize> = Vec::new();
        for i in temp_vec.iter() {
            let z = damerau_levenshtein(s_p.trim(), i.as_str().trim());
            rez_vec.push(z);
        }
        rez_vec
    }
    pub fn match_string_percentage_list(&mut self, source: &PyString, target: &PyList) -> Vec<f64> {
        let s = source.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let mut temp_vec: Vec<String> = Vec::new();
        for i in target.iter() {
            let k = match i.extract() {
                Ok(val) => {
                    val
                }
                Err(Why) => {
                    panic!("{:}", Why);
                    String::new()
                }
            };
            temp_vec.push(self.re_obj.replace_non_letters_non_numbers_with_whitespace(k));
        }
        let mut rez_vec: Vec<f64> = Vec::new();
        for i in temp_vec.iter() {
            let z = normalized_damerau_levenshtein(s_p.trim(), i.as_str().trim());
            rez_vec.push(z);
        }
        rez_vec
    }
}

#[pymodule]
pub fn damerau(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Damerau>()?;
    Ok(())
}
