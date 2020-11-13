extern crate strsim;

use std::borrow::BorrowMut;
use std::collections::HashMap;

use ngrams::Ngram;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use pyo3::wrap_pyfunction;
use strsim::{hamming, jaro, jaro_winkler, osa_distance, sorensen_dice};

use crate::utils::StringProcessing;

#[pyclass]
pub struct StringMatcher {
    re_obj: StringProcessing
}

#[pymethods]
impl StringMatcher {
    #[new]
    pub fn new() -> Self {
        let obj = StringProcessing::new();
        StringMatcher {
            re_obj: obj,
        }
    }

    pub fn jaro_winkler_difference(&mut self, source: &PyString, target: &PyString) -> f64 {
        let s = source.to_string_lossy().to_string();
        let t = target.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let t_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(t);
        jaro_winkler(s_p.trim(), t_p.trim())
    }

    pub fn jaro_difference(&mut self, source: &PyString, target: &PyString) -> f64 {
        let s = source.to_string_lossy().to_string();
        let t = target.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let t_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(t);
        jaro(s_p.trim(), t_p.trim())
    }

    pub fn hamming_difference(&mut self, source: &PyString, target: &PyString) -> usize {
        let s = source.to_string_lossy().to_string();
        let t = target.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let t_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(t);
        let k = match hamming(s_p.trim(), t_p.trim()) {
            Ok(val) => val,
            Err(why) => panic!("{:}", why)
        };
        k
    }

    pub fn osa_distance_difference(&mut self, source: &PyString, target: &PyString) -> usize {
        let s = source.to_string_lossy().to_string();
        let t = target.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let t_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(t);
        osa_distance(s_p.trim(), t_p.trim())
    }

    pub fn sorensen_dice_difference(&mut self, source: &PyString, target: &PyString) -> f64 {
        let s = source.to_string_lossy().to_string();
        let t = target.to_string_lossy().to_string();
        let s_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(s);
        let t_p = self.re_obj.replace_non_letters_non_numbers_with_whitespace(t);
        sorensen_dice(s_p.trim(), t_p.trim())
    }


    pub fn jaro_winkler_difference_list(&mut self, source: &PyString, target: &PyList) -> HashMap<String, f64> {
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
        let mut rez_vec: HashMap<String, f64> = HashMap::new();
        for i in temp_vec.iter() {
            let z = jaro_winkler(s_p.trim(), i.as_str().trim());
            rez_vec.insert(i.clone(), z);
        }
        rez_vec
    }

    pub fn jaro_difference_list(&mut self, source: &PyString, target: &PyList) -> HashMap<String, f64> {
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
        let mut rez_vec: HashMap<String, f64> = HashMap::new();
        for i in temp_vec.iter() {
            let z = jaro(s_p.trim(), i.as_str().trim());
            rez_vec.insert(i.clone(), z);
        }
        rez_vec
    }

    pub fn sorensen_dice_difference_list(&mut self, source: &PyString, target: &PyList) -> HashMap<String, f64> {
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
        let mut rez_vec: HashMap<String, f64> = HashMap::new();
        for i in temp_vec.iter() {
            let z = sorensen_dice(s_p.trim(), i.as_str().trim());
            rez_vec.insert(i.clone(), z);
        }
        rez_vec
    }

    pub fn hamming_difference_list(&mut self, source: &PyString, target: &PyList) -> HashMap<String, usize> {
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
            let z = match hamming(s_p.trim(), i.as_str().trim()) {
                Ok(val) => val,
                Err(why) => panic!("{:}", why)
            };
            rez_vec.insert(i.clone(), z);
        }
        rez_vec
    }

    pub fn osa_distance_difference_list(&mut self, source: &PyString, target: &PyList) -> HashMap<String, usize> {
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
            let z = osa_distance(s_p.trim(), i.as_str().trim());
            rez_vec.insert(i.clone(), z);
        }
        rez_vec
    }
}

#[pymodule]
pub fn string_matcher(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<StringMatcher>()?;
    Ok(())
}


