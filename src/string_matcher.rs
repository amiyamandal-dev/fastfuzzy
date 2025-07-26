// optimized_algorithms.rs - Optimized algorithms for FastFuzz
extern crate strsim;

use std::cmp::{max, min};
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;

use once_cell::sync::Lazy;
use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;

// Cached regex for better performance
static WORD_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?ui)\W").unwrap());

// Pre-allocated buffer pool for string operations
thread_local! {
    static STRING_BUFFER: std::cell::RefCell<String> = std::cell::RefCell::new(String::with_capacity(256));
}

// ============================================================================
// Optimized String Processor with caching
// ============================================================================

pub struct StringProcessor {
    // Cache for processed strings to avoid recomputation
    cache: Option<Arc<dashmap::DashMap<String, String>>>,
}

impl StringProcessor {
    pub fn new() -> Self {
        StringProcessor {
            cache: Some(Arc::new(dashmap::DashMap::with_capacity(1000))),
        }
    }

    pub fn new_without_cache() -> Self {
        StringProcessor { cache: None }
    }

    #[inline]
    pub fn process_string(&self, s: &str) -> String {
        if let Some(ref cache) = self.cache {
            if let Some(cached) = cache.get(s) {
                return cached.clone();
            }
        }

        let cleaned = WORD_REGEX.replace_all(s, " ");
        let result = cleaned.trim().to_lowercase();

        if let Some(ref cache) = self.cache {
            cache.insert(s.to_string(), result.clone());
        }

        result
    }

    #[inline]
    pub fn ascii_only(&self, s: &str) -> String {
        s.chars()
            .filter(|c| c.is_ascii())
            .collect::<String>()
            .trim()
            .to_string()
    }

    #[inline]
    pub fn trim_whitespace(&self, s: &str) -> String {
        STRING_BUFFER.with(|buffer| {
            let mut buf = buffer.borrow_mut();
            buf.clear();

            let mut prev_was_space = true;
            for ch in s.chars() {
                if ch.is_whitespace() {
                    if !prev_was_space {
                        buf.push(' ');
                        prev_was_space = true;
                    }
                } else {
                    buf.push(ch);
                    prev_was_space = false;
                }
            }

            buf.trim().to_string()
        })
    }

    #[inline]
    pub fn sort_tokens(&self, s: &str) -> String {
        let mut tokens: Vec<&str> = s.split_whitespace().collect();
        tokens.sort_unstable();
        tokens.join(" ")
    }
}

// ============================================================================
// Optimized FuzzyRatio with early termination
// ============================================================================

#[pyclass]
pub struct FuzzyRatio {
    processor: Arc<StringProcessor>,
}

#[pymethods]
impl FuzzyRatio {
    #[new]
    pub fn new() -> Self {
        FuzzyRatio {
            processor: Arc::new(StringProcessor::new()),
        }
    }

    pub fn ratio(
        &self,
        s1: &str,
        target: &str,
        score_cutoff: Option<f64>,
    ) -> PyResult<Option<f64>> {
        // Early length-based filtering
        if let Some(cutoff) = score_cutoff {
            let len_diff = (s1.len() as i32 - target.len() as i32).abs() as f64;
            let max_len = max(s1.len(), target.len()) as f64;

            if max_len > 0.0 {
                let max_possible = 100.0 * (1.0 - len_diff / max_len);
                if max_possible < cutoff {
                    return Ok(None);
                }
            }
        }

        let processed1 = self.processor.process_string(s1);
        let processed2 = self.processor.process_string(target);

        let score = self.calculate_ratio_optimized(&processed1, &processed2, score_cutoff);

        if let Some(cutoff) = score_cutoff {
            if score < cutoff {
                return Ok(None);
            }
        }

        Ok(Some(score))
    }

    pub fn partial_ratio(
        &self,
        s1: &str,
        target: &str,
        score_cutoff: Option<f64>,
    ) -> PyResult<Option<f64>> {
        let processed1 = self.processor.process_string(s1);
        let processed2 = self.processor.process_string(target);

        let score = self.calculate_partial_ratio_optimized(&processed1, &processed2, score_cutoff);

        if let Some(cutoff) = score_cutoff {
            if score < cutoff {
                return Ok(None);
            }
        }

        Ok(Some(score))
    }

    pub fn token_sort_ratio(
        &self,
        s1: &str,
        target: &str,
        score_cutoff: Option<f64>,
    ) -> PyResult<Option<f64>> {
        let sorted1 = self.processor.sort_tokens(s1);
        let sorted2 = self.processor.sort_tokens(target);

        let score = self.calculate_ratio_optimized(&sorted1, &sorted2, score_cutoff);

        if let Some(cutoff) = score_cutoff {
            if score < cutoff {
                return Ok(None);
            }
        }

        Ok(Some(score))
    }

    pub fn token_set_ratio(
        &self,
        s1: &str,
        target: &str,
        score_cutoff: Option<f64>,
    ) -> PyResult<Option<f64>> {
        let score = self.calculate_token_set_ratio_optimized(s1, target, score_cutoff);

        if let Some(cutoff) = score_cutoff {
            if score < cutoff {
                return Ok(None);
            }
        }

        Ok(Some(score))
    }

    pub fn wratio(
        &self,
        s1: &str,
        target: &str,
        score_cutoff: Option<f64>,
    ) -> PyResult<Option<f64>> {
        let processed1 = self.processor.process_string(s1);
        let processed2 = self.processor.process_string(target);

        // Quick check for exact match
        if processed1 == processed2 {
            return Ok(Some(100.0));
        }

        // Calculate length ratio
        let len_ratio = if processed1.len() > processed2.len() {
            processed2.len() as f64 / processed1.len() as f64
        } else {
            processed1.len() as f64 / processed2.len() as f64
        };

        // Choose best algorithm based on characteristics
        let score = if len_ratio > 0.9 {
            // Similar lengths - use basic ratio
            self.calculate_ratio_optimized(&processed1, &processed2, score_cutoff)
        } else {
            // Different lengths - try partial ratio
            let partial =
                self.calculate_partial_ratio_optimized(&processed1, &processed2, score_cutoff);

            // Only calculate token ratios if partial isn't good enough
            if partial < 95.0 {
                let token_sort = self.token_sort_ratio(s1, target, None)?.unwrap_or(0.0);
                partial.max(token_sort)
            } else {
                partial
            }
        };

        if let Some(cutoff) = score_cutoff {
            if score < cutoff {
                return Ok(None);
            }
        }

        Ok(Some(score))
    }

    pub fn ratio_list(
        &self,
        query: &str,
        choices: Vec<String>,
        score_cutoff: Option<f64>,
    ) -> PyResult<Vec<Option<f64>>> {
        let processed_query = Arc::new(self.processor.process_string(query));
        let query_len = processed_query.len();

        let results: Vec<Option<f64>> = choices
            .par_iter()
            .map(|choice| {
                // Early length filtering
                if let Some(cutoff) = score_cutoff {
                    let len_diff = (choice.len() as i32 - query_len as i32).abs() as f64;
                    let max_len = max(choice.len(), query_len) as f64;

                    if max_len > 0.0 {
                        let max_possible = 100.0 * (1.0 - len_diff / max_len);
                        if max_possible < cutoff {
                            return None;
                        }
                    }
                }

                let processed_choice = self.processor.process_string(choice);
                let score = self.calculate_ratio_optimized(
                    &processed_query,
                    &processed_choice,
                    score_cutoff,
                );

                if let Some(cutoff) = score_cutoff {
                    if score < cutoff {
                        return None;
                    }
                }
                Some(score)
            })
            .collect();

        Ok(results)
    }
}

impl FuzzyRatio {
    #[inline]
    fn calculate_ratio_optimized(&self, s1: &str, s2: &str, cutoff: Option<f64>) -> f64 {
        if s1.is_empty() && s2.is_empty() {
            return 100.0;
        }

        if s1 == s2 {
            return 100.0;
        }

        let max_len = max(s1.len(), s2.len());
        if max_len == 0 {
            return 100.0;
        }

        // Early termination check
        if let Some(cutoff_val) = cutoff {
            let len_diff = (s1.len() as i32 - s2.len() as i32).abs();
            let max_possible = 100.0 * (1.0 - len_diff as f64 / max_len as f64);

            if max_possible < cutoff_val {
                return 0.0;
            }
        }

        let distance = strsim::levenshtein(s1, s2);
        100.0 * (1.0 - (distance as f64 / max_len as f64))
    }

    #[inline]
    fn calculate_partial_ratio_optimized(&self, s1: &str, s2: &str, cutoff: Option<f64>) -> f64 {
        if s1.len() == s2.len() {
            return self.calculate_ratio_optimized(s1, s2, cutoff);
        }

        let (shorter, longer) = if s1.len() <= s2.len() {
            (s1, s2)
        } else {
            (s2, s1)
        };

        if shorter.is_empty() {
            return 0.0;
        }

        let shorter_len = shorter.len();
        let mut best_ratio: f64 = 0.0;

        // Use a sliding window with early termination
        for i in 0..=(longer.len().saturating_sub(shorter_len)) {
            let substring = &longer[i..min(i + shorter_len, longer.len())];

            // Quick character frequency check for early rejection
            if !self.quick_char_check(shorter, substring) {
                continue;
            }

            let ratio = self.calculate_ratio_optimized(shorter, substring, None);
            best_ratio = best_ratio.max(ratio);

            // Early termination on perfect match
            if best_ratio >= 100.0 {
                break;
            }

            // Early termination if remaining windows can't improve
            if let Some(cutoff_val) = cutoff {
                if best_ratio >= cutoff_val {
                    break;
                }
            }
        }

        best_ratio
    }

    #[inline]
    fn quick_char_check(&self, s1: &str, s2: &str) -> bool {
        // Quick check if s2 could potentially match s1
        let mut chars1 = [0u32; 256];
        let mut chars2 = [0u32; 256];

        for ch in s1.bytes() {
            if (ch as usize) < 256 {
                chars1[ch as usize] += 1;
            }
        }

        for ch in s2.bytes() {
            if (ch as usize) < 256 {
                chars2[ch as usize] += 1;
            }
        }

        // Check if s2 has at least the characters in s1
        for i in 0..256 {
            if chars1[i] > chars2[i] {
                return false;
            }
        }

        true
    }

    #[inline]
    fn calculate_token_set_ratio_optimized(&self, s1: &str, s2: &str, _cutoff: Option<f64>) -> f64 {
        let tokens1: HashSet<&str> = s1.split_whitespace().collect();
        let tokens2: HashSet<&str> = s2.split_whitespace().collect();

        // Quick check for identical sets
        if tokens1 == tokens2 {
            return 100.0;
        }

        let intersection: Vec<&str> = tokens1.intersection(&tokens2).cloned().collect();
        let diff1: Vec<&str> = tokens1.difference(&tokens2).cloned().collect();
        let diff2: Vec<&str> = tokens2.difference(&tokens1).cloned().collect();

        // Build strings more efficiently
        let mut sorted1 = String::with_capacity(s1.len());
        let mut sorted2 = String::with_capacity(s2.len());

        // Add intersection
        for (i, token) in intersection.iter().enumerate() {
            if i > 0 {
                sorted1.push(' ');
                sorted2.push(' ');
            }
            sorted1.push_str(token);
            sorted2.push_str(token);
        }

        // Add differences
        if !diff1.is_empty() && !intersection.is_empty() {
            sorted1.push(' ');
        }
        for (i, token) in diff1.iter().enumerate() {
            if i > 0 {
                sorted1.push(' ');
            }
            sorted1.push_str(token);
        }

        if !diff2.is_empty() && !intersection.is_empty() {
            sorted2.push(' ');
        }
        for (i, token) in diff2.iter().enumerate() {
            if i > 0 {
                sorted2.push(' ');
            }
            sorted2.push_str(token);
        }

        self.calculate_ratio_optimized(&sorted1, &sorted2, None)
    }
}

// ============================================================================
// Optimized Process Module with better caching and parallel processing
// ============================================================================

#[pyclass]
pub struct Process {
    ratio_calculator: Arc<FuzzyRatio>,
    // Cache for scorer results
    scorer_cache: Arc<dashmap::DashMap<(String, String, String), f64>>,
}

#[pymethods]
impl Process {
    #[new]
    pub fn new() -> Self {
        Process {
            ratio_calculator: Arc::new(FuzzyRatio::new()),
            scorer_cache: Arc::new(dashmap::DashMap::with_capacity(10000)),
        }
    }

    pub fn extract(
        &self,
        query: &str,
        choices: Vec<String>,
        limit: Option<usize>,
        score_cutoff: Option<f64>,
        scorer: Option<&str>,
    ) -> PyResult<Vec<(String, f64, usize)>> {
        let scorer_type = scorer.unwrap_or("ratio");
        let cutoff = score_cutoff.unwrap_or(0.0);

        // Pre-filter based on length if using ratio scorer
        let filtered_choices: Vec<(usize, &String)> = if scorer_type == "ratio" && cutoff > 60.0 {
            let query_len = query.len();
            let max_len_diff = ((100.0 - cutoff) / 100.0 * query_len as f64) as usize;

            choices
                .iter()
                .enumerate()
                .filter(|(_, choice)| {
                    let len_diff = (choice.len() as i32 - query_len as i32).abs() as usize;
                    len_diff <= max_len_diff
                })
                .collect()
        } else {
            choices.iter().enumerate().collect()
        };

        let mut results: Vec<(String, f64, usize)> = filtered_choices
            .par_iter()
            .filter_map(|(idx, choice)| {
                let score = self
                    .calculate_score_cached(query, choice, scorer_type)
                    .ok()?;

                if score >= cutoff {
                    Some(((*choice).clone(), score, *idx))
                } else {
                    None
                }
            })
            .collect();

        // Use unstable sort for better performance
        results.par_sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(limit_val) = limit {
            results.truncate(limit_val);
        }

        Ok(results)
    }

    pub fn extract_one(
        &self,
        query: &str,
        choices: Vec<String>,
        score_cutoff: Option<f64>,
        scorer: Option<&str>,
    ) -> PyResult<Option<(String, f64, usize)>> {
        let results = self.extract(query, choices, Some(1), score_cutoff, scorer)?;
        Ok(results.into_iter().next())
    }

    pub fn cdist(
        &self,
        queries: Vec<String>,
        choices: Vec<String>,
        scorer: Option<&str>,
        workers: Option<usize>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let scorer_type = scorer.unwrap_or("ratio");

        // Configure thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers.unwrap_or_else(|| rayon::current_num_threads()))
            .build()
            .unwrap();

        let results = pool.install(|| {
            queries
                .par_iter()
                .map(|query| {
                    choices
                        .par_iter()
                        .map(|choice| {
                            self.calculate_score_cached(query, choice, scorer_type)
                                .unwrap_or(0.0)
                        })
                        .collect()
                })
                .collect()
        });

        Ok(results)
    }
}

impl Process {
    #[inline]
    fn calculate_score_cached(&self, query: &str, choice: &str, scorer: &str) -> PyResult<f64> {
        let cache_key = (query.to_string(), choice.to_string(), scorer.to_string());

        if let Some(cached_score) = self.scorer_cache.get(&cache_key) {
            return Ok(*cached_score);
        }

        let score = self.calculate_score(query, choice, scorer)?;
        self.scorer_cache.insert(cache_key, score);

        Ok(score)
    }

    #[inline]
    fn calculate_score(&self, query: &str, choice: &str, scorer: &str) -> PyResult<f64> {
        match scorer {
            "ratio" => self
                .ratio_calculator
                .ratio(query, choice, None)
                .map(|opt| opt.unwrap_or(0.0)),
            "partial_ratio" => self
                .ratio_calculator
                .partial_ratio(query, choice, None)
                .map(|opt| opt.unwrap_or(0.0)),
            "token_sort_ratio" => self
                .ratio_calculator
                .token_sort_ratio(query, choice, None)
                .map(|opt| opt.unwrap_or(0.0)),
            "token_set_ratio" => self
                .ratio_calculator
                .token_set_ratio(query, choice, None)
                .map(|opt| opt.unwrap_or(0.0)),
            "wratio" => self
                .ratio_calculator
                .wratio(query, choice, None)
                .map(|opt| opt.unwrap_or(0.0)),
            _ => Ok(0.0),
        }
    }
}

// ============================================================================
// Optimized Additional Metrics with memoization
// ============================================================================

#[pyclass]
pub struct AdditionalMetrics {
    processor: Arc<StringProcessor>,
    lcs_cache: Arc<dashmap::DashMap<(String, String), usize>>,
}

#[pymethods]
impl AdditionalMetrics {
    #[new]
    pub fn new() -> Self {
        AdditionalMetrics {
            processor: Arc::new(StringProcessor::new()),
            lcs_cache: Arc::new(dashmap::DashMap::with_capacity(1000)),
        }
    }

    pub fn longest_common_subsequence(&self, s1: &str, s2: &str) -> PyResult<usize> {
        let processed1 = self.processor.process_string(s1);
        let processed2 = self.processor.process_string(s2);

        let cache_key = (processed1.clone(), processed2.clone());

        if let Some(cached) = self.lcs_cache.get(&cache_key) {
            return Ok(*cached);
        }

        let result = self.lcs_length_optimized(&processed1, &processed2);
        self.lcs_cache.insert(cache_key, result);

        Ok(result)
    }

    pub fn longest_common_substring(&self, s1: &str, s2: &str) -> PyResult<usize> {
        let processed1 = self.processor.process_string(s1);
        let processed2 = self.processor.process_string(s2);

        Ok(self.lcs_substring_length_optimized(&processed1, &processed2))
    }

    pub fn prefix_distance(&self, s1: &str, s2: &str) -> PyResult<usize> {
        let processed1 = self.processor.process_string(s1);
        let processed2 = self.processor.process_string(s2);

        Ok(self.calculate_prefix_distance_optimized(&processed1, &processed2))
    }

    pub fn postfix_distance(&self, s1: &str, s2: &str) -> PyResult<usize> {
        let processed1 = self.processor.process_string(s1);
        let processed2 = self.processor.process_string(s2);

        Ok(self.calculate_postfix_distance_optimized(&processed1, &processed2))
    }

    pub fn indel_distance(&self, s1: &str, s2: &str) -> PyResult<usize> {
        let processed1 = self.processor.process_string(s1);
        let processed2 = self.processor.process_string(s2);

        Ok(self.calculate_indel_distance(&processed1, &processed2))
    }

    pub fn lcs_seq_distance(&self, s1: &str, s2: &str) -> PyResult<usize> {
        let processed1 = self.processor.process_string(s1);
        let processed2 = self.processor.process_string(s2);

        let lcs_len = self.lcs_length_optimized(&processed1, &processed2);
        let total_len = processed1.len() + processed2.len();

        Ok(total_len - 2 * lcs_len)
    }
}

impl AdditionalMetrics {
    #[inline]
    fn lcs_length_optimized(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let m = chars1.len();
        let n = chars2.len();

        if m == 0 || n == 0 {
            return 0;
        }

        // Use rolling array optimization to save memory
        let mut prev = vec![0; n + 1];
        let mut curr = vec![0; n + 1];

        for i in 1..=m {
            for j in 1..=n {
                if chars1[i - 1] == chars2[j - 1] {
                    curr[j] = prev[j - 1] + 1;
                } else {
                    curr[j] = max(prev[j], curr[j - 1]);
                }
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        prev[n]
    }

    #[inline]
    fn lcs_substring_length_optimized(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let m = chars1.len();
        let n = chars2.len();

        if m == 0 || n == 0 {
            return 0;
        }

        // Use rolling array optimization
        let mut prev = vec![0; n + 1];
        let mut curr = vec![0; n + 1];
        let mut max_length = 0;

        for i in 1..=m {
            for j in 1..=n {
                if chars1[i - 1] == chars2[j - 1] {
                    curr[j] = prev[j - 1] + 1;
                    max_length = max(max_length, curr[j]);
                } else {
                    curr[j] = 0;
                }
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        max_length
    }

    #[inline]
    fn calculate_prefix_distance_optimized(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();

        let common_prefix = chars1
            .iter()
            .zip(chars2.iter())
            .take_while(|(a, b)| a == b)
            .count();

        max(chars1.len(), chars2.len()) - common_prefix
    }

    #[inline]
    fn calculate_postfix_distance_optimized(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();

        let common_suffix = chars1
            .iter()
            .rev()
            .zip(chars2.iter().rev())
            .take_while(|(a, b)| a == b)
            .count();

        max(chars1.len(), chars2.len()) - common_suffix
    }

    #[inline]
    fn calculate_indel_distance(&self, s1: &str, s2: &str) -> usize {
        let lcs_len = self.lcs_length_optimized(s1, s2);
        s1.len() + s2.len() - 2 * lcs_len
    }
}

// ============================================================================
// Optimized Phonetic Algorithms
// ============================================================================

#[pyclass]
pub struct PhoneticAlgorithms {
    soundex_cache: Arc<dashmap::DashMap<String, String>>,
}

#[pymethods]
impl PhoneticAlgorithms {
    #[new]
    pub fn new() -> Self {
        PhoneticAlgorithms {
            soundex_cache: Arc::new(dashmap::DashMap::with_capacity(1000)),
        }
    }

    pub fn soundex(&self, s: &str) -> PyResult<String> {
        if let Some(cached) = self.soundex_cache.get(s) {
            return Ok(cached.clone());
        }

        let result = self.calculate_soundex_optimized(s);
        self.soundex_cache.insert(s.to_string(), result.clone());

        Ok(result)
    }

    pub fn metaphone(&self, s: &str) -> PyResult<String> {
        Ok(self.calculate_metaphone(s))
    }

    pub fn double_metaphone(&self, s: &str) -> PyResult<(String, String)> {
        let (primary, secondary) = self.calculate_double_metaphone(s);
        Ok((primary, secondary))
    }

    pub fn soundex_list(&self, string_list: Vec<String>) -> PyResult<Vec<String>> {
        let results: Vec<String> = string_list
            .par_iter()
            .map(|s| {
                if let Some(cached) = self.soundex_cache.get(s) {
                    cached.clone()
                } else {
                    let result = self.calculate_soundex_optimized(s);
                    self.soundex_cache.insert(s.clone(), result.clone());
                    result
                }
            })
            .collect();

        Ok(results)
    }
}

impl PhoneticAlgorithms {
    #[inline]
    fn calculate_soundex_optimized(&self, s: &str) -> String {
        if s.is_empty() {
            return String::new();
        }

        let s = s.to_uppercase();
        let chars: Vec<char> = s.chars().filter(|c| c.is_alphabetic()).collect();

        if chars.is_empty() {
            return String::new();
        }

        let mut result = String::with_capacity(4);
        result.push(chars[0]);

        let mut prev_code = self.soundex_code_optimized(chars[0]);

        for &ch in &chars[1..] {
            let code = self.soundex_code_optimized(ch);
            if code != '0' && code != prev_code {
                result.push(code);
                if result.len() == 4 {
                    return result;
                }
            }
            prev_code = code;
        }

        // Pad with zeros if needed
        while result.len() < 4 {
            result.push('0');
        }

        result
    }

    #[inline]
    fn soundex_code_optimized(&self, c: char) -> char {
        match c {
            'B' | 'F' | 'P' | 'V' => '1',
            'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => '2',
            'D' | 'T' => '3',
            'L' => '4',
            'M' | 'N' => '5',
            'R' => '6',
            _ => '0',
        }
    }

    fn calculate_metaphone(&self, s: &str) -> String {
        let s = s.to_uppercase();
        let mut result = String::with_capacity(4);
        let chars: Vec<char> = s.chars().filter(|c| c.is_alphabetic()).collect();

        if chars.is_empty() {
            return result;
        }

        let mut i = 0;
        while i < chars.len() && result.len() < 4 {
            let ch = chars[i];

            match ch {
                'A' | 'E' | 'I' | 'O' | 'U' => {
                    if i == 0 {
                        result.push(ch);
                    }
                }
                'B' => result.push('B'),
                'C' => {
                    if i + 1 < chars.len() && chars[i + 1] == 'H' {
                        result.push('X');
                        i += 1;
                    } else {
                        result.push('K');
                    }
                }
                'D' => result.push('T'),
                'F' => result.push('F'),
                'G' => result.push('K'),
                'H' => {
                    if i == 0 || is_vowel(chars.get(i.saturating_sub(1)).copied().unwrap_or(' ')) {
                        result.push('H');
                    }
                }
                'J' => result.push('J'),
                'K' => result.push('K'),
                'L' => result.push('L'),
                'M' => result.push('M'),
                'N' => result.push('N'),
                'P' => {
                    if i + 1 < chars.len() && chars[i + 1] == 'H' {
                        result.push('F');
                        i += 1;
                    } else {
                        result.push('P');
                    }
                }
                'Q' => result.push('K'),
                'R' => result.push('R'),
                'S' => result.push('S'),
                'T' => {
                    if i + 1 < chars.len() && chars[i + 1] == 'H' {
                        result.push('0');
                        i += 1;
                    } else {
                        result.push('T');
                    }
                }
                'V' => result.push('F'),
                'W' | 'Y' => {
                    if i + 1 < chars.len() && is_vowel(chars[i + 1]) {
                        result.push(ch);
                    }
                }
                'X' => result.push_str("KS"),
                'Z' => result.push('S'),
                _ => {}
            }

            i += 1;
        }

        result
    }

    fn calculate_double_metaphone(&self, s: &str) -> (String, String) {
        let primary = self.calculate_metaphone(s);
        let secondary = primary.clone();
        (primary, secondary)
    }
}

// ============================================================================
// Optimized Utils Module
// ============================================================================

#[pyclass]
pub struct Utils {
    processor: Arc<StringProcessor>,
}

#[pymethods]
impl Utils {
    #[new]
    pub fn new() -> Self {
        Utils {
            processor: Arc::new(StringProcessor::new()),
        }
    }

    pub fn default_process(&self, s: &str) -> PyResult<String> {
        Ok(self.processor.process_string(s))
    }

    pub fn ascii_only(&self, s: &str) -> PyResult<String> {
        Ok(self.processor.ascii_only(s))
    }

    pub fn trim_whitespace(&self, s: &str) -> PyResult<String> {
        Ok(self.processor.trim_whitespace(s))
    }

    pub fn batch_process(&self, string_list: Vec<String>) -> PyResult<Vec<String>> {
        let results: Vec<String> = string_list
            .par_iter()
            .map(|s| self.processor.process_string(s))
            .collect();

        Ok(results)
    }

    pub fn remove_punctuation(&self, s: &str) -> PyResult<String> {
        STRING_BUFFER.with(|buffer| {
            let mut buf = buffer.borrow_mut();
            buf.clear();

            for ch in s.chars() {
                if ch.is_alphanumeric() || ch.is_whitespace() {
                    buf.push(ch);
                }
            }

            Ok(buf.trim().to_string())
        })
    }

    pub fn normalize_unicode(&self, s: &str) -> PyResult<String> {
        Ok(s.trim().to_string())
    }
}

// ============================================================================
// High-Performance Optimized Scorers
// ============================================================================

#[pyclass]
pub struct OptimizedScorers {
    processor: Arc<StringProcessor>,
    ratio_cache: Arc<dashmap::DashMap<(String, String), f64>>,
}

#[pymethods]
impl OptimizedScorers {
    #[new]
    pub fn new() -> Self {
        OptimizedScorers {
            processor: Arc::new(StringProcessor::new()),
            ratio_cache: Arc::new(dashmap::DashMap::with_capacity(5000)),
        }
    }

    pub fn ratio_cutoff(&self, str1: &str, str2: &str, score_cutoff: f64) -> PyResult<Option<f64>> {
        // Quick length check for early termination
        let len_diff = (str1.len() as i32 - str2.len() as i32).abs() as f64;
        let max_len = str1.len().max(str2.len()) as f64;

        if max_len > 0.0 {
            let max_possible_score = 100.0 * (1.0 - len_diff / max_len);
            if max_possible_score < score_cutoff {
                return Ok(None);
            }
        }

        let processed1 = self.processor.process_string(str1);
        let processed2 = self.processor.process_string(str2);

        // Check cache
        let cache_key = (processed1.clone(), processed2.clone());
        if let Some(cached_score) = self.ratio_cache.get(&cache_key) {
            if *cached_score >= score_cutoff {
                return Ok(Some(*cached_score));
            } else {
                return Ok(None);
            }
        }

        let score = self.calculate_ratio_optimized(&processed1, &processed2, score_cutoff);
        self.ratio_cache.insert(cache_key, score);

        if score < score_cutoff {
            Ok(None)
        } else {
            Ok(Some(score))
        }
    }

    pub fn ratio_batch_cutoff(
        &self,
        query: &str,
        choices: Vec<String>,
        score_cutoff: f64,
    ) -> PyResult<Vec<Option<f64>>> {
        let processed_query = Arc::new(self.processor.process_string(query));
        let query_len = processed_query.len();

        // Calculate maximum allowed length difference
        let max_len_diff = ((100.0 - score_cutoff) / 100.0 * query_len as f64) as usize;

        let results: Vec<Option<f64>> = choices
            .par_iter()
            .map(|choice| {
                // Quick length check
                let len_diff = (choice.len() as i32 - query_len as i32).abs() as usize;
                if len_diff > max_len_diff {
                    return None;
                }

                let processed_choice = self.processor.process_string(choice);

                // Check cache
                let cache_key = (processed_query.as_ref().clone(), processed_choice.clone());
                if let Some(cached_score) = self.ratio_cache.get(&cache_key) {
                    if *cached_score >= score_cutoff {
                        return Some(*cached_score);
                    } else {
                        return None;
                    }
                }

                let score = self.calculate_ratio_optimized(
                    &processed_query,
                    &processed_choice,
                    score_cutoff,
                );

                self.ratio_cache.insert(cache_key, score);

                if score >= score_cutoff {
                    Some(score)
                } else {
                    None
                }
            })
            .collect();

        Ok(results)
    }

    pub fn jaro_winkler_cutoff(
        &self,
        s1: &str,
        s2: &str,
        score_cutoff: f64,
    ) -> PyResult<Option<f64>> {
        let processed1 = self.processor.process_string(s1);
        let processed2 = self.processor.process_string(s2);

        let score = strsim::jaro_winkler(&processed1, &processed2) * 100.0;

        if score >= score_cutoff {
            Ok(Some(score))
        } else {
            Ok(None)
        }
    }

    pub fn auto_score(&self, s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
        let processed1 = self.processor.process_string(s1);
        let processed2 = self.processor.process_string(s2);

        // Quick exact match check
        if processed1 == processed2 {
            return Ok(100.0);
        }

        // Choose algorithm based on string characteristics
        let len_ratio = if processed1.len() > processed2.len() {
            processed2.len() as f64 / processed1.len() as f64
        } else {
            processed1.len() as f64 / processed2.len() as f64
        };

        let score = if len_ratio < 0.5 {
            // Very different lengths - use partial ratio
            self.calculate_partial_ratio_optimized(&processed1, &processed2)
        } else if processed1.split_whitespace().count() > 3
            || processed2.split_whitespace().count() > 3
        {
            // Multi-token strings - use token set ratio
            self.calculate_token_set_ratio(&processed1, &processed2)
        } else {
            // Similar lengths - use standard ratio
            self.calculate_ratio_optimized(&processed1, &processed2, score_cutoff.unwrap_or(0.0))
        };

        Ok(score)
    }
}

impl OptimizedScorers {
    #[inline]
    fn calculate_ratio_optimized(&self, s1: &str, s2: &str, cutoff: f64) -> f64 {
        if s1.is_empty() && s2.is_empty() {
            return 100.0;
        }

        if s1 == s2 {
            return 100.0;
        }

        let max_len = s1.len().max(s2.len());
        if max_len == 0 {
            return 100.0;
        }

        // Early termination based on length difference
        let len_diff = (s1.len() as i32 - s2.len() as i32).abs();
        let max_possible_score = 100.0 * (1.0 - len_diff as f64 / max_len as f64);

        if max_possible_score < cutoff {
            return 0.0;
        }

        let distance = strsim::levenshtein(s1, s2);
        100.0 * (1.0 - distance as f64 / max_len as f64)
    }

    #[inline]
    fn calculate_partial_ratio_optimized(&self, s1: &str, s2: &str) -> f64 {
        if s1.len() == s2.len() {
            return self.calculate_ratio_optimized(s1, s2, 0.0);
        }

        let (shorter, longer) = if s1.len() <= s2.len() {
            (s1, s2)
        } else {
            (s2, s1)
        };

        if shorter.is_empty() {
            return 0.0;
        }

        let shorter_len = shorter.len();
        let mut best_ratio: f64 = 0.0;

        // Use byte comparison for ASCII strings
        if shorter.is_ascii() && longer.is_ascii() {
            let shorter_bytes = shorter.as_bytes();
            let longer_bytes = longer.as_bytes();

            for i in 0..=(longer_bytes.len().saturating_sub(shorter_bytes.len())) {
                let substring = &longer_bytes[i..i + shorter_bytes.len()];

                // Quick byte-level comparison
                let mut matches = 0;
                for j in 0..shorter_bytes.len() {
                    if shorter_bytes[j] == substring[j] {
                        matches += 1;
                    }
                }

                // Only do full comparison if promising
                if matches as f64 / shorter_bytes.len() as f64 > 0.6 {
                    let substring_str = std::str::from_utf8(substring).unwrap();
                    let ratio = self.calculate_ratio_optimized(shorter, substring_str, 0.0);
                    best_ratio = best_ratio.max(ratio);

                    if best_ratio >= 100.0 {
                        break;
                    }
                }
            }
        } else {
            // Fallback to char-based comparison for Unicode
            for i in 0..=(longer.len().saturating_sub(shorter_len)) {
                let end_idx = (i + shorter_len).min(longer.len());
                let substring = &longer[i..end_idx];
                let ratio = self.calculate_ratio_optimized(shorter, substring, 0.0);
                best_ratio = best_ratio.max(ratio);

                if best_ratio >= 100.0 {
                    break;
                }
            }
        }

        best_ratio
    }

    #[inline]
    fn calculate_token_set_ratio(&self, s1: &str, s2: &str) -> f64 {
        let tokens1: HashSet<&str> = s1.split_whitespace().collect();
        let tokens2: HashSet<&str> = s2.split_whitespace().collect();

        if tokens1 == tokens2 {
            return 100.0;
        }

        let intersection: Vec<&str> = tokens1.intersection(&tokens2).cloned().collect();
        let diff1: Vec<&str> = tokens1.difference(&tokens2).cloned().collect();
        let diff2: Vec<&str> = tokens2.difference(&tokens1).cloned().collect();

        let mut sorted1 = String::with_capacity(s1.len());
        let mut sorted2 = String::with_capacity(s2.len());

        for (i, token) in intersection.iter().enumerate() {
            if i > 0 {
                sorted1.push(' ');
                sorted2.push(' ');
            }
            sorted1.push_str(token);
            sorted2.push_str(token);
        }

        if !diff1.is_empty() && !intersection.is_empty() {
            sorted1.push(' ');
        }
        for (i, token) in diff1.iter().enumerate() {
            if i > 0 {
                sorted1.push(' ');
            }
            sorted1.push_str(token);
        }

        if !diff2.is_empty() && !intersection.is_empty() {
            sorted2.push(' ');
        }
        for (i, token) in diff2.iter().enumerate() {
            if i > 0 {
                sorted2.push(' ');
            }
            sorted2.push_str(token);
        }

        self.calculate_ratio_optimized(&sorted1, &sorted2, 0.0)
    }
}

// ============================================================================
// Optimized Streaming Processor
// ============================================================================

#[derive(Clone, Debug)]
struct ScoredResult {
    choice: String,
    score: f64,
    index: usize,
}

impl Eq for ScoredResult {}

impl PartialEq for ScoredResult {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.index == other.index
    }
}

impl PartialOrd for ScoredResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Compare by score first (reversed for max-heap behavior)
        match other.score.partial_cmp(&self.score) {
            Some(std::cmp::Ordering::Equal) => {
                // If scores are equal, compare by index for stability
                Some(self.index.cmp(&other.index))
            }
            other => other,
        }
    }
}

impl Ord for ScoredResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Handle NaN by treating it as less than any other value
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Less)
    }
}

#[pyclass]
pub struct StreamingProcessor {
    processor: Arc<StringProcessor>,
    chunk_size: usize,
}

#[pymethods]
impl StreamingProcessor {
    #[new]
    pub fn new(chunk_size: Option<usize>) -> Self {
        StreamingProcessor {
            processor: Arc::new(StringProcessor::new()),
            chunk_size: chunk_size.unwrap_or(1000),
        }
    }

    pub fn extract_iter(
        &self,
        query_str: &str,
        choices: Vec<String>,
        score_cutoff: f64,
        limit: Option<usize>,
    ) -> PyResult<Vec<(String, f64, usize)>> {
        let processed_query = Arc::new(self.processor.process_string(query_str));
        let query_len = processed_query.len();

        // Pre-calculate maximum allowed length difference
        let max_len_diff = ((100.0 - score_cutoff) / 100.0 * query_len as f64) as usize;

        // Use a min-heap to maintain top results efficiently
        let limit_val = limit.unwrap_or(usize::MAX);
        let mut heap = std::collections::BinaryHeap::with_capacity(limit_val);

        for (chunk_start, chunk) in choices.chunks(self.chunk_size).enumerate() {
            let chunk_results: Vec<ScoredResult> = chunk
                .par_iter()
                .enumerate()
                .filter_map(|(local_idx, choice)| {
                    // Quick length filter
                    let len_diff = (choice.len() as i32 - query_len as i32).abs() as usize;
                    if len_diff > max_len_diff {
                        return None;
                    }

                    let processed_choice = self.processor.process_string(choice);
                    let score =
                        self.calculate_similarity_optimized(&processed_query, &processed_choice);

                    if score >= score_cutoff {
                        let global_idx = chunk_start * self.chunk_size + local_idx;
                        Some(ScoredResult {
                            choice: choice.clone(),
                            score,
                            index: global_idx,
                        })
                    } else {
                        None
                    }
                })
                .collect();

            // Merge results into heap
            for result in chunk_results {
                heap.push(result);
                if heap.len() > limit_val {
                    heap.pop();
                }
            }
        }

        // Extract results from heap and convert back to tuples
        let mut results: Vec<(String, f64, usize)> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|item| (item.choice, item.score, item.index))
            .collect();

        // Results are in ascending order by score, so reverse for descending
        results.reverse();
        Ok(results)
    }

    pub fn streaming_cdist(
        &self,
        queries: Vec<String>,
        choices: Vec<String>,
        score_cutoff: Option<f64>,
    ) -> PyResult<Vec<Vec<Option<f64>>>> {
        let cutoff = score_cutoff.unwrap_or(0.0);

        // Pre-process all queries
        let processed_queries: Vec<String> = queries
            .par_iter()
            .map(|q| self.processor.process_string(q))
            .collect();

        // Process in chunks with parallel execution
        let results: Vec<Vec<Option<f64>>> = processed_queries
            .par_chunks(self.chunk_size)
            .flat_map(|query_chunk| {
                query_chunk
                    .par_iter()
                    .map(|processed_query| {
                        choices
                            .par_chunks(self.chunk_size)
                            .flat_map(|choice_chunk| {
                                choice_chunk.par_iter().map(|choice| {
                                    let processed_choice = self.processor.process_string(choice);
                                    let score = self.calculate_similarity_optimized(
                                        processed_query,
                                        &processed_choice,
                                    );

                                    if score >= cutoff {
                                        Some(score)
                                    } else {
                                        None
                                    }
                                })
                            })
                            .collect()
                    })
                    .collect::<Vec<Vec<Option<f64>>>>()
            })
            .collect();

        Ok(results)
    }

    pub fn approximate_search(
        &self,
        query_str: &str,
        choices: Vec<String>,
        k: usize,
        approximation_factor: f64,
    ) -> PyResult<Vec<(String, f64, usize)>> {
        let processed_query = Arc::new(self.processor.process_string(query_str));
        let query_len = processed_query.len();
        let len_threshold = (query_len as f64 * approximation_factor) as usize;

        // Two-pass approach: filter then score
        let filtered_indices: Vec<usize> = (0..choices.len())
            .into_par_iter()
            .filter(|&idx| {
                let choice_len = choices[idx].len();
                (choice_len as i32 - query_len as i32).abs() <= len_threshold as i32
            })
            .collect();

        // Score filtered choices
        let mut results: Vec<(String, f64, usize)> = filtered_indices
            .par_iter()
            .filter_map(|&idx| {
                let choice = &choices[idx];
                let processed_choice = self.processor.process_string(choice);
                let score =
                    self.calculate_similarity_optimized(&processed_query, &processed_choice);

                if score > 0.0 {
                    Some((choice.clone(), score, idx))
                } else {
                    None
                }
            })
            .collect();

        // Use partial sort for efficiency when k < total results
        if k < results.len() {
            results.select_nth_unstable_by(k, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(k);
        } else {
            results.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Ok(results)
    }
}

impl StreamingProcessor {
    #[inline]
    fn calculate_similarity_optimized(&self, s1: &str, s2: &str) -> f64 {
        if s1.is_empty() && s2.is_empty() {
            return 100.0;
        }

        if s1 == s2 {
            return 100.0;
        }

        let max_len = s1.len().max(s2.len());
        if max_len == 0 {
            return 100.0;
        }

        let distance = strsim::levenshtein(s1, s2);
        100.0 * (1.0 - distance as f64 / max_len as f64)
    }
}

// Helper functions
#[inline]
fn is_vowel(c: char) -> bool {
    matches!(
        c.to_uppercase().next().unwrap_or(' '),
        'A' | 'E' | 'I' | 'O' | 'U'
    )
}
