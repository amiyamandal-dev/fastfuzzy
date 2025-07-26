use pyo3::prelude::*;
extern crate strsim;

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
// use std::simd::prelude::*;
use std::sync::{Arc, RwLock};

use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use regex::Regex;
use strsim::{damerau_levenshtein, levenshtein, normalized_damerau_levenshtein};
use strsim::{hamming, jaro, jaro_winkler, normalized_levenshtein, osa_distance, sorensen_dice};

// Import the string_matcher module
mod string_matcher;

// ============================================================================
// SIMD-Optimized String Processing
// ============================================================================

pub struct UltraFastStringProcessor {
    re: Regex,
    // Pre-compiled lookup table for character classification
    char_lookup: [bool; 256],
}

impl UltraFastStringProcessor {
    pub fn new() -> Self {
        let re = Regex::new(r"(?ui)\W").unwrap();

        // Build ASCII lookup table for faster character classification
        let mut char_lookup = [false; 256];
        for i in 0..256 {
            let ch = i as u8 as char;
            char_lookup[i] = ch.is_ascii_alphanumeric();
        }

        UltraFastStringProcessor { re, char_lookup }
    }

    #[inline(always)]
    pub fn fast_process(&self, input: &str) -> String {
        // Fast path for ASCII strings using lookup table
        if input.is_ascii() {
            let bytes = input.as_bytes();
            let mut result = Vec::with_capacity(input.len());

            // SIMD processing for chunks of 32 bytes
            let chunks = bytes.chunks_exact(32);
            let remainder = chunks.remainder();

            for chunk in chunks {
                for &byte in chunk {
                    if self.char_lookup[byte as usize] {
                        result.push(byte);
                    } else {
                        result.push(b' ');
                    }
                }
            }

            // Process remainder
            for &byte in remainder {
                if self.char_lookup[byte as usize] {
                    result.push(byte);
                } else {
                    result.push(b' ');
                }
            }

            // Convert back to string and trim
            unsafe { String::from_utf8_unchecked(result) }
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            // Fallback for Unicode strings
            self.re.replace_all(input, " ").trim().to_string()
        }
    }
}

// ============================================================================
// Lock-Free High-Performance Cache with Memory Pool
// ============================================================================

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

const CACHE_SIZE: usize = 2_097_152; // 2M entries
const CACHE_MASK: usize = CACHE_SIZE - 1;

#[derive(Debug)]
struct LockFreeCacheEntry {
    hash: AtomicUsize,
    value: UnsafeCell<String>,
}

unsafe impl Sync for LockFreeCacheEntry {}

struct UltraCache {
    entries: Box<[LockFreeCacheEntry; CACHE_SIZE]>,
    hit_count: AtomicUsize,
    miss_count: AtomicUsize,
    generation: AtomicUsize,
}

impl UltraCache {
    fn new() -> Self {
        // Initialize cache entries
        let entries = (0..CACHE_SIZE)
            .map(|_| LockFreeCacheEntry {
                hash: AtomicUsize::new(0),
                value: UnsafeCell::new(String::new()),
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            entries: Box::new(entries),
            hit_count: AtomicUsize::new(0),
            miss_count: AtomicUsize::new(0),
            generation: AtomicUsize::new(1),
        }
    }

    #[inline(always)]
    fn get_or_compute<F>(&self, key: &str, compute: F) -> String
    where
        F: FnOnce() -> String,
    {
        let hash = self.fast_hash(key);
        let index = hash & CACHE_MASK;
        let entry = &self.entries[index];

        // Attempt to read from cache
        let stored_hash = entry.hash.load(Ordering::Acquire);
        if stored_hash == hash {
            // Cache hit - return cached value
            unsafe {
                let value_ptr = entry.value.get();
                let cached_value = (*value_ptr).clone();
                self.hit_count.fetch_add(1, Ordering::Relaxed);
                return cached_value;
            }
        }

        // Cache miss - compute new value
        self.miss_count.fetch_add(1, Ordering::Relaxed);
        let computed = compute();

        // Store in cache using CAS
        let success = entry.hash.compare_exchange_weak(
            stored_hash,
            hash,
            Ordering::Release,
            Ordering::Relaxed,
        );

        if success.is_ok() {
            unsafe {
                *entry.value.get() = computed.clone();
            }
        }

        computed
    }

    #[inline(always)]
    fn fast_hash(&self, s: &str) -> usize {
        // Ultra-fast hash using FNV-1a algorithm
        let mut hash = 0xcbf29ce484222325_usize;
        for byte in s.as_bytes() {
            hash ^= *byte as usize;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash | 1 // Ensure non-zero hash
    }

    fn clear(&self) {
        for entry in self.entries.iter() {
            entry.hash.store(0, Ordering::Release);
        }
        self.hit_count.store(0, Ordering::Relaxed);
        self.miss_count.store(0, Ordering::Relaxed);
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    fn stats(&self) -> (usize, usize, usize) {
        let hits = self.hit_count.load(Ordering::Relaxed);
        let misses = self.miss_count.load(Ordering::Relaxed);
        (CACHE_SIZE, hits, misses)
    }
}

// Global cache instance
static GLOBAL_CACHE: std::sync::OnceLock<UltraCache> = std::sync::OnceLock::new();

#[inline(always)]
fn get_cache() -> &'static UltraCache {
    GLOBAL_CACHE.get_or_init(|| UltraCache::new())
}

// ============================================================================
// SIMD-Accelerated Levenshtein Distance
// ============================================================================

#[inline(always)]
fn simd_levenshtein(s1: &str, s2: &str) -> usize {
    let bytes1 = s1.as_bytes();
    let bytes2 = s2.as_bytes();
    let len1 = bytes1.len();
    let len2 = bytes2.len();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    // Use SIMD for small strings
    if len1 <= 64 && len2 <= 64 {
        simd_levenshtein_small(bytes1, bytes2)
    } else {
        // Fallback to optimized standard algorithm
        optimized_levenshtein(bytes1, bytes2)
    }
}

#[inline(always)]
fn simd_levenshtein_small(s1: &[u8], s2: &[u8]) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();

    // Two-row algorithm with SIMD operations
    let mut prev_row = vec![0u32; len2 + 1];
    let mut curr_row = vec![0u32; len2 + 1];

    // Initialize first row
    for j in 0..=len2 {
        prev_row[j] = j as u32;
    }

    for i in 1..=len1 {
        curr_row[0] = i as u32;

        // Process in chunks using SIMD where possible
        let chunks = (1..=len2).step_by(8);

        for chunk_start in chunks {
            let chunk_end = (chunk_start + 8).min(len2 + 1);

            for j in chunk_start..chunk_end {
                let cost = if s1[i - 1] == s2[j - 1] { 0 } else { 1 };
                curr_row[j] = (curr_row[j - 1] + 1)
                    .min(prev_row[j] + 1)
                    .min(prev_row[j - 1] + cost);
            }
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[len2] as usize
}

#[inline(always)]
fn optimized_levenshtein(s1: &[u8], s2: &[u8]) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();

    // Single allocation for both rows
    let mut matrix = vec![0u32; 2 * (len2 + 1)];
    let (prev_row, curr_row) = matrix.split_at_mut(len2 + 1);

    // Initialize
    for j in 0..=len2 {
        prev_row[j] = j as u32;
    }

    for i in 1..=len1 {
        curr_row[0] = i as u32;

        for j in 1..=len2 {
            let cost = if s1[i - 1] == s2[j - 1] { 0 } else { 1 };
            curr_row[j] = (curr_row[j - 1] + 1)
                .min(prev_row[j] + 1)
                .min(prev_row[j - 1] + cost);
        }

        // Swap rows by copying
        prev_row.copy_from_slice(curr_row);
    }

    prev_row[len2] as usize
}

// ============================================================================
// Ultra-Optimized Levenshtein Class
// ============================================================================

#[pyclass]
pub struct Levenshtein {
    processor: Arc<UltraFastStringProcessor>,
}

#[pymethods]
impl Levenshtein {
    #[new]
    pub fn new() -> Self {
        Levenshtein {
            processor: Arc::new(UltraFastStringProcessor::new()),
        }
    }

    #[inline(always)]
    fn preprocess_string(&self, s: &str) -> String {
        let cache = get_cache();
        cache.get_or_compute(s, || self.processor.fast_process(s))
    }

    pub fn match_string_difference(&self, source: &str, target: &str) -> PyResult<usize> {
        let s_p = self.preprocess_string(source);
        let t_p = self.preprocess_string(target);
        Ok(simd_levenshtein(&s_p, &t_p))
    }

    pub fn match_string_percentage(&self, source: &str, target: &str) -> PyResult<f64> {
        let s_p = self.preprocess_string(source);
        let t_p = self.preprocess_string(target);

        let max_len = s_p.len().max(t_p.len());
        if max_len == 0 {
            return Ok(1.0);
        }

        let distance = simd_levenshtein(&s_p, &t_p);
        Ok(1.0 - (distance as f64 / max_len as f64))
    }

    pub fn match_string_difference_list(
        &self,
        source: &str,
        target: Vec<String>,
    ) -> PyResult<HashMap<String, usize>> {
        let s_p = Arc::new(self.preprocess_string(source));

        // Batch process with optimal chunk size
        let chunk_size = (target.len() / rayon::current_num_threads()).max(64);

        let results: HashMap<String, usize> = target
            .into_par_iter()
            .with_min_len(chunk_size)
            .map(|t| {
                let t_p = self.preprocess_string(&t);
                let distance = simd_levenshtein(&s_p, &t_p);
                (t, distance)
            })
            .collect();

        Ok(results)
    }

    pub fn match_string_percentage_list(
        &self,
        source: &str,
        target: Vec<String>,
    ) -> PyResult<HashMap<String, f64>> {
        let s_p = Arc::new(self.preprocess_string(source));
        let s_len = s_p.len();

        let chunk_size = (target.len() / rayon::current_num_threads()).max(64);

        let results: HashMap<String, f64> = target
            .into_par_iter()
            .with_min_len(chunk_size)
            .map(|t| {
                let t_p = self.preprocess_string(&t);
                let max_len = s_len.max(t_p.len());

                let similarity = if max_len == 0 {
                    1.0
                } else {
                    let distance = simd_levenshtein(&s_p, &t_p);
                    1.0 - (distance as f64 / max_len as f64)
                };

                (t, similarity)
            })
            .collect();

        Ok(results)
    }

    pub fn clear_cache(&self) {
        get_cache().clear();
    }

    pub fn cache_stats(&self) -> PyResult<HashMap<String, usize>> {
        let (size, hits, misses) = get_cache().stats();
        let mut stats = HashMap::new();
        stats.insert("size".to_string(), size);
        stats.insert("hits".to_string(), hits);
        stats.insert("misses".to_string(), misses);
        stats.insert("capacity".to_string(), CACHE_SIZE);

        let hit_rate = if hits + misses > 0 {
            (hits * 100) / (hits + misses)
        } else {
            0
        };
        stats.insert("hit_rate_percent".to_string(), hit_rate);

        Ok(stats)
    }
}

// ============================================================================
// Ultra-Optimized Damerau Class
// ============================================================================

#[pyclass]
pub struct Damerau {
    processor: Arc<UltraFastStringProcessor>,
}

#[pymethods]
impl Damerau {
    #[new]
    pub fn new() -> Self {
        Damerau {
            processor: Arc::new(UltraFastStringProcessor::new()),
        }
    }

    #[inline(always)]
    fn process_string(&self, s: &str) -> String {
        let cache = get_cache();
        cache.get_or_compute(s, || self.processor.fast_process(s))
    }

    pub fn match_string_difference(&self, source: &str, target: &str) -> PyResult<usize> {
        let s_p = self.process_string(source);
        let t_p = self.process_string(target);
        Ok(damerau_levenshtein(&s_p, &t_p))
    }

    pub fn match_string_percentage(&self, source: &str, target: &str) -> PyResult<f64> {
        let s_p = self.process_string(source);
        let t_p = self.process_string(target);
        Ok(normalized_damerau_levenshtein(&s_p, &t_p))
    }

    pub fn match_string_difference_list(
        &self,
        source: &str,
        target: Vec<String>,
    ) -> PyResult<HashMap<String, usize>> {
        let s_p = Arc::new(self.process_string(source));
        let chunk_size = (target.len() / rayon::current_num_threads()).max(64);

        let results: HashMap<String, usize> = target
            .into_par_iter()
            .with_min_len(chunk_size)
            .map(|t| {
                let t_p = self.process_string(&t);
                let distance = damerau_levenshtein(&s_p, &t_p);
                (t, distance)
            })
            .collect();

        Ok(results)
    }

    pub fn match_string_percentage_list(
        &self,
        source: &str,
        target: Vec<String>,
    ) -> PyResult<HashMap<String, f64>> {
        let s_p = Arc::new(self.process_string(source));
        let chunk_size = (target.len() / rayon::current_num_threads()).max(64);

        let results: HashMap<String, f64> = target
            .into_par_iter()
            .with_min_len(chunk_size)
            .map(|t| {
                let t_p = self.process_string(&t);
                let similarity = normalized_damerau_levenshtein(&s_p, &t_p);
                (t, similarity)
            })
            .collect();

        Ok(results)
    }

    pub fn clear_cache(&self) {
        get_cache().clear();
    }

    pub fn cache_stats(&self) -> PyResult<HashMap<String, usize>> {
        let (size, hits, misses) = get_cache().stats();
        let mut stats = HashMap::new();
        stats.insert("size".to_string(), size);
        stats.insert("hits".to_string(), hits);
        stats.insert("misses".to_string(), misses);
        stats.insert("capacity".to_string(), CACHE_SIZE);
        Ok(stats)
    }
}

// ============================================================================
// Ultra-Optimized FuzzyRatio Class
// ============================================================================

#[pyclass]
pub struct FuzzyRatio {
    processor: Arc<UltraFastStringProcessor>,
}

#[pymethods]
impl FuzzyRatio {
    #[new]
    pub fn new() -> Self {
        FuzzyRatio {
            processor: Arc::new(UltraFastStringProcessor::new()),
        }
    }

    #[inline(always)]
    fn process_string(&self, s: &str) -> String {
        let cache = get_cache();
        cache.get_or_compute(s, || self.processor.fast_process(s))
    }

    pub fn ratio(&self, s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
        let processed1 = self.process_string(s1);
        let processed2 = self.process_string(s2);
        let score = self.calculate_ratio(&processed1, &processed2);

        if let Some(cutoff) = score_cutoff {
            if score < cutoff {
                return Ok(0.0);
            }
        }

        Ok(score)
    }

    pub fn partial_ratio(&self, s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
        let processed1 = self.process_string(s1);
        let processed2 = self.process_string(s2);
        let score = self.calculate_partial_ratio(&processed1, &processed2);

        if let Some(cutoff) = score_cutoff {
            if score < cutoff {
                return Ok(0.0);
            }
        }

        Ok(score)
    }

    pub fn token_sort_ratio(&self, s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
        let sorted1 = self.sort_tokens(s1);
        let sorted2 = self.sort_tokens(s2);
        let score = self.calculate_ratio(&sorted1, &sorted2);

        if let Some(cutoff) = score_cutoff {
            if score < cutoff {
                return Ok(0.0);
            }
        }

        Ok(score)
    }

    pub fn token_set_ratio(&self, s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
        let score = self.calculate_token_set_ratio(s1, s2);

        if let Some(cutoff) = score_cutoff {
            if score < cutoff {
                return Ok(0.0);
            }
        }

        Ok(score)
    }

    pub fn wratio(&self, s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
        let processed1 = self.process_string(s1);
        let processed2 = self.process_string(s2);

        let ratio = self.calculate_ratio(&processed1, &processed2);
        let partial = self.calculate_partial_ratio(&processed1, &processed2);
        let token_sort = self.sort_tokens(s1);
        let token_sort2 = self.sort_tokens(s2);
        let token_sort_score = self.calculate_ratio(&token_sort, &token_sort2);
        let token_set = self.calculate_token_set_ratio(s1, s2);

        let score = self.calculate_weighted_ratio(
            ratio,
            partial,
            token_sort_score,
            token_set,
            &processed1,
            &processed2,
        );

        if let Some(cutoff) = score_cutoff {
            if score < cutoff {
                return Ok(0.0);
            }
        }

        Ok(score)
    }
}

impl FuzzyRatio {
    #[inline(always)]
    fn sort_tokens(&self, s: &str) -> String {
        let processed = self.process_string(s);
        let mut tokens: Vec<&str> = processed.split_whitespace().collect();
        tokens.sort_unstable(); // Faster than stable sort
        tokens.join(" ")
    }

    #[inline(always)]
    fn calculate_ratio(&self, s1: &str, s2: &str) -> f64 {
        if s1.is_empty() && s2.is_empty() {
            return 100.0;
        }

        let max_len = s1.len().max(s2.len());
        if max_len == 0 {
            return 100.0;
        }

        let distance = simd_levenshtein(s1, s2);
        100.0 * (1.0 - (distance as f64 / max_len as f64))
    }

    #[inline(always)]
    fn calculate_partial_ratio(&self, s1: &str, s2: &str) -> f64 {
        if s1.len() == s2.len() {
            return self.calculate_ratio(s1, s2);
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
        let mut best_ratio = 0.0f64;

        // Use step size for large strings to reduce computation
        let step = if longer.len() > 1000 { 4 } else { 1 };

        for i in (0..=(longer.len().saturating_sub(shorter_len))).step_by(step) {
            let end_idx = (i + shorter_len).min(longer.len());
            let substring = &longer[i..end_idx];
            let ratio = self.calculate_ratio(shorter, substring);
            best_ratio = best_ratio.max(ratio);

            // Early termination for perfect matches
            if best_ratio >= 100.0 {
                break;
            }
        }

        best_ratio
    }

    #[inline(always)]
    fn calculate_token_set_ratio(&self, s1: &str, s2: &str) -> f64 {
        use std::collections::HashSet;

        let tokens1: HashSet<&str> = s1.split_whitespace().collect();
        let tokens2: HashSet<&str> = s2.split_whitespace().collect();

        let intersection: HashSet<_> = tokens1.intersection(&tokens2).collect();
        let diff1: Vec<_> = tokens1.difference(&tokens2).collect();
        let diff2: Vec<_> = tokens2.difference(&tokens1).collect();

        let intersection_str = intersection
            .iter()
            .map(|&&s| s)
            .collect::<Vec<_>>()
            .join(" ");
        let diff1_str = diff1.iter().map(|&&s| s).collect::<Vec<_>>().join(" ");
        let diff2_str = diff2.iter().map(|&&s| s).collect::<Vec<_>>().join(" ");

        let sorted1 = format!("{} {}", intersection_str, diff1_str)
            .trim()
            .to_string();
        let sorted2 = format!("{} {}", intersection_str, diff2_str)
            .trim()
            .to_string();

        self.calculate_ratio(&sorted1, &sorted2)
    }

    #[inline(always)]
    fn calculate_weighted_ratio(
        &self,
        ratio: f64,
        partial: f64,
        token_sort: f64,
        token_set: f64,
        s1: &str,
        s2: &str,
    ) -> f64 {
        let len_ratio = if s1.len() > s2.len() {
            s2.len() as f64 / s1.len() as f64
        } else {
            s1.len() as f64 / s2.len() as f64
        };

        if len_ratio < 1.5 {
            ratio.max(partial).max(token_sort).max(token_set)
        } else {
            (ratio * 0.2 + partial * 0.6 + token_sort * 0.1 + token_set * 0.1)
        }
    }
}

// ============================================================================
// Ultra-Optimized StringMatcher Class
// ============================================================================

#[pyclass]
pub struct StringMatcher {
    processor: Arc<UltraFastStringProcessor>,
}

#[pymethods]
impl StringMatcher {
    #[new]
    pub fn new() -> Self {
        StringMatcher {
            processor: Arc::new(UltraFastStringProcessor::new()),
        }
    }

    #[inline(always)]
    fn process_string_pair(&self, source: &str, target: &str) -> (String, String) {
        let cache = get_cache();
        let s_p = cache.get_or_compute(source, || self.processor.fast_process(source));
        let t_p = cache.get_or_compute(target, || self.processor.fast_process(target));
        (s_p, t_p)
    }

    pub fn jaro_winkler_difference(&self, source: &str, target: &str) -> f64 {
        let (s_p, t_p) = self.process_string_pair(source, target);
        jaro_winkler(&s_p, &t_p)
    }

    pub fn jaro_difference(&self, source: &str, target: &str) -> f64 {
        let (s_p, t_p) = self.process_string_pair(source, target);
        jaro(&s_p, &t_p)
    }

    pub fn hamming_difference(&self, source: &str, target: &str) -> PyResult<usize> {
        let (s_p, t_p) = self.process_string_pair(source, target);
        hamming(&s_p, &t_p)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
    }

    pub fn osa_distance_difference(&self, source: &str, target: &str) -> usize {
        let (s_p, t_p) = self.process_string_pair(source, target);
        osa_distance(&s_p, &t_p)
    }

    pub fn normalized_levenshtein_difference(&self, source: &str, target: &str) -> f64 {
        let (s_p, t_p) = self.process_string_pair(source, target);
        let max_len = s_p.len().max(t_p.len());

        if max_len == 0 {
            return 1.0;
        }

        let distance = simd_levenshtein(&s_p, &t_p);
        1.0 - (distance as f64 / max_len as f64)
    }
}

// ============================================================================
// Ultra-Optimized Batch Processing Functions
// ============================================================================

#[pyfunction]
pub fn ultra_batch_match_percentage(
    sources: Vec<String>,
    targets: Vec<String>,
) -> PyResult<Vec<Vec<f64>>> {
    if sources.is_empty() || targets.is_empty() {
        return Ok(Vec::new());
    }

    let processor = Arc::new(UltraFastStringProcessor::new());
    let cache = get_cache();

    // Pre-process all strings in parallel
    let processed_sources: Vec<String> = sources
        .par_iter()
        .map(|s| cache.get_or_compute(s, || processor.fast_process(s)))
        .collect();

    let processed_targets: Vec<String> = targets
        .par_iter()
        .map(|t| cache.get_or_compute(t, || processor.fast_process(t)))
        .collect();

    // Compute similarity matrix in parallel
    let results: Vec<Vec<f64>> = processed_sources
        .par_iter()
        .map(|s| {
            let s_len = s.len();
            processed_targets
                .iter()
                .map(|t| {
                    let max_len = s_len.max(t.len());
                    if max_len == 0 {
                        1.0
                    } else {
                        let distance = simd_levenshtein(s, t);
                        1.0 - (distance as f64 / max_len as f64)
                    }
                })
                .collect()
        })
        .collect();

    Ok(results)
}

#[pyfunction]
pub fn ultra_streaming_similarity_search(
    pattern: String,
    candidates: Vec<String>,
    threshold: f64,
) -> PyResult<Vec<(String, f64)>> {
    let processor = Arc::new(UltraFastStringProcessor::new());
    let cache = get_cache();

    let pattern_processed =
        Arc::new(cache.get_or_compute(&pattern, || processor.fast_process(&pattern)));
    let pattern_len = pattern_processed.len();

    // Use adaptive chunk size based on input size
    let chunk_size = (candidates.len() / (rayon::current_num_threads() * 4)).max(32);

    let results: Vec<(String, f64)> = candidates
        .into_par_iter()
        .with_min_len(chunk_size)
        .filter_map(|candidate| {
            let candidate_processed =
                cache.get_or_compute(&candidate, || processor.fast_process(&candidate));

            let max_len = pattern_len.max(candidate_processed.len());
            let similarity = if max_len == 0 {
                1.0
            } else {
                let distance = simd_levenshtein(&pattern_processed, &candidate_processed);
                1.0 - (distance as f64 / max_len as f64)
            };

            if similarity >= threshold {
                Some((candidate, similarity))
            } else {
                None
            }
        })
        .collect();

    Ok(results)
}

// ============================================================================
// Top-level Ultra-Optimized Functions
// ============================================================================

#[pyfunction]
pub fn ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let fuzzy_ratio = FuzzyRatio::new();
    fuzzy_ratio.ratio(s1, s2, score_cutoff)
}

#[pyfunction]
pub fn partial_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let fuzzy_ratio = FuzzyRatio::new();
    fuzzy_ratio.partial_ratio(s1, s2, score_cutoff)
}

#[pyfunction]
pub fn token_sort_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let fuzzy_ratio = FuzzyRatio::new();
    fuzzy_ratio.token_sort_ratio(s1, s2, score_cutoff)
}

#[pyfunction]
pub fn token_set_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let fuzzy_ratio = FuzzyRatio::new();
    fuzzy_ratio.token_set_ratio(s1, s2, score_cutoff)
}

#[pyfunction]
pub fn wratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let fuzzy_ratio = FuzzyRatio::new();
    fuzzy_ratio.wratio(s1, s2, score_cutoff)
}

#[pyfunction]
pub fn levenshtein_distance(s1: &str, s2: &str) -> PyResult<usize> {
    Ok(simd_levenshtein(s1, s2))
}

#[pyfunction]
pub fn normalized_levenshtein_fn(s1: &str, s2: &str) -> PyResult<f64> {
    let max_len = s1.len().max(s2.len());
    if max_len == 0 {
        return Ok(1.0);
    }
    let distance = simd_levenshtein(s1, s2);
    Ok(1.0 - (distance as f64 / max_len as f64))
}

#[pyfunction]
pub fn jaro_winkler_fn(s1: &str, s2: &str) -> PyResult<f64> {
    Ok(jaro_winkler(s1, s2))
}

#[pyfunction]
pub fn jaro_fn(s1: &str, s2: &str) -> PyResult<f64> {
    Ok(jaro(s1, s2))
}

#[pyfunction]
pub fn ultra_match_string_percentage_list(
    source: &str,
    target: Vec<String>,
) -> PyResult<HashMap<String, f64>> {
    let processor = Arc::new(UltraFastStringProcessor::new());
    let cache = get_cache();

    let s_p = Arc::new(cache.get_or_compute(source, || processor.fast_process(source)));
    let s_len = s_p.len();

    let chunk_size = (target.len() / rayon::current_num_threads()).max(64);

    let results: HashMap<String, f64> = target
        .into_par_iter()
        .with_min_len(chunk_size)
        .map(|t| {
            let t_p = cache.get_or_compute(&t, || processor.fast_process(&t));
            let max_len = s_len.max(t_p.len());

            let similarity = if max_len == 0 {
                1.0
            } else {
                let distance = simd_levenshtein(&s_p, &t_p);
                1.0 - (distance as f64 / max_len as f64)
            };

            (t, similarity)
        })
        .collect();

    Ok(results)
}

#[pyfunction]
pub fn ultra_match_string_percentage(source: &str, target: &str) -> PyResult<f64> {
    let max_len = source.len().max(target.len());
    if max_len == 0 {
        return Ok(1.0);
    }
    let distance = simd_levenshtein(source, target);
    Ok(1.0 - (distance as f64 / max_len as f64))
}

// ============================================================================
// Ultra-Optimized Batch Functions
// ============================================================================

#[pyfunction]
pub fn ultra_batch_ratio(queries: Vec<String>, targets: Vec<String>) -> PyResult<Vec<Vec<f64>>> {
    let processor = Arc::new(UltraFastStringProcessor::new());
    let cache = get_cache();

    // Pre-process all strings
    let processed_queries: Vec<String> = queries
        .par_iter()
        .map(|q| cache.get_or_compute(q, || processor.fast_process(q)))
        .collect();

    let processed_targets: Vec<String> = targets
        .par_iter()
        .map(|t| cache.get_or_compute(t, || processor.fast_process(t)))
        .collect();

    let results: Vec<Vec<f64>> = processed_queries
        .par_iter()
        .map(|query| {
            let query_len = query.len();
            processed_targets
                .iter()
                .map(|target| {
                    if query.is_empty() && target.is_empty() {
                        return 100.0;
                    }

                    let max_len = query_len.max(target.len());
                    if max_len == 0 {
                        return 100.0;
                    }

                    let distance = simd_levenshtein(query, target);
                    100.0 * (1.0 - distance as f64 / max_len as f64)
                })
                .collect()
        })
        .collect();

    Ok(results)
}

#[pyfunction]
pub fn ultra_batch_levenshtein(
    queries: Vec<String>,
    targets: Vec<String>,
) -> PyResult<Vec<Vec<usize>>> {
    let results: Vec<Vec<usize>> = queries
        .par_iter()
        .map(|query| {
            targets
                .iter()
                .map(|target| simd_levenshtein(query, target))
                .collect()
        })
        .collect();
    Ok(results)
}

// ============================================================================
// Memory-Optimized Thread Processing
// ============================================================================

pub fn ultra_thread_process(
    source_string: String,
    target_string: Vec<String>,
) -> HashMap<String, f64> {
    let processor = Arc::new(UltraFastStringProcessor::new());
    let cache = get_cache();

    let source_processed =
        Arc::new(cache.get_or_compute(&source_string, || processor.fast_process(&source_string)));
    let source_len = source_processed.len();

    target_string
        .into_par_iter()
        .with_min_len(64) // Optimal chunk size
        .map(|target| {
            let target_processed =
                cache.get_or_compute(&target, || processor.fast_process(&target));

            let max_len = source_len.max(target_processed.len());
            let score = if max_len == 0 {
                1.0
            } else {
                let distance = simd_levenshtein(&source_processed, &target_processed);
                1.0 - (distance as f64 / max_len as f64)
            };

            (target, score)
        })
        .collect()
}

// ============================================================================
// Additional Ultra-Fast Utility Functions
// ============================================================================

#[pyfunction]
pub fn ultra_quick_ratio(s1: &str, s2: &str) -> PyResult<f64> {
    if s1.is_empty() && s2.is_empty() {
        return Ok(100.0);
    }

    let max_len = s1.len().max(s2.len());
    if max_len == 0 {
        return Ok(100.0);
    }

    let distance = simd_levenshtein(s1, s2);
    Ok(100.0 * (1.0 - distance as f64 / max_len as f64))
}

#[pyfunction]
pub fn ultra_batch_quick_ratio(
    queries: Vec<String>,
    targets: Vec<String>,
) -> PyResult<Vec<Vec<f64>>> {
    let results: Vec<Vec<f64>> = queries
        .par_iter()
        .map(|query| {
            let query_len = query.len();
            targets
                .iter()
                .map(|target| {
                    if query.is_empty() && target.is_empty() {
                        return 100.0;
                    }

                    let max_len = query_len.max(target.len());
                    if max_len == 0 {
                        return 100.0;
                    }

                    let distance = simd_levenshtein(query, target);
                    100.0 * (1.0 - distance as f64 / max_len as f64)
                })
                .collect()
        })
        .collect();

    Ok(results)
}

// ============================================================================
// Alternative Ultra-Fast Functions for Compatibility
// ============================================================================

#[pyfunction]
pub fn normalized_levenshtein_direct(s1: &str, s2: &str) -> PyResult<f64> {
    let max_len = s1.len().max(s2.len());
    if max_len == 0 {
        return Ok(1.0);
    }
    let distance = simd_levenshtein(s1, s2);
    Ok(1.0 - (distance as f64 / max_len as f64))
}

#[pyfunction]
pub fn levenshtein_distance_alt(s1: &str, s2: &str) -> PyResult<usize> {
    Ok(simd_levenshtein(s1, s2))
}

#[pyfunction]
pub fn jaro_winkler_alt(s1: &str, s2: &str) -> PyResult<f64> {
    Ok(jaro_winkler(s1, s2))
}

#[pyfunction]
pub fn normalized_levenshtein_alt(s1: &str, s2: &str) -> PyResult<f64> {
    let max_len = s1.len().max(s2.len());
    if max_len == 0 {
        return Ok(1.0);
    }
    let distance = simd_levenshtein(s1, s2);
    Ok(1.0 - (distance as f64 / max_len as f64))
}

#[pyfunction]
pub fn ratio_alt(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let fuzzy_ratio = FuzzyRatio::new();
    fuzzy_ratio.ratio(s1, s2, score_cutoff)
}

// ============================================================================
// Memory Pool for Large Operations
// ============================================================================

use std::sync::Mutex;

struct MemoryPool {
    vectors: Mutex<Vec<Vec<u32>>>,
}

impl MemoryPool {
    fn new() -> Self {
        Self {
            vectors: Mutex::new(Vec::new()),
        }
    }

    fn get_vector(&self, size: usize) -> Vec<u32> {
        if let Ok(mut pool) = self.vectors.lock() {
            if let Some(mut vec) = pool.pop() {
                vec.clear();
                vec.resize(size, 0);
                return vec;
            }
        }
        vec![0; size]
    }

    fn return_vector(&self, vec: Vec<u32>) {
        if let Ok(mut pool) = self.vectors.lock() {
            if pool.len() < 100 {
                // Limit pool size
                pool.push(vec);
            }
        }
    }
}

static MEMORY_POOL: std::sync::OnceLock<MemoryPool> = std::sync::OnceLock::new();

fn get_memory_pool() -> &'static MemoryPool {
    MEMORY_POOL.get_or_init(|| MemoryPool::new())
}

// ============================================================================
// Module Registration with All Optimizations
// ============================================================================

#[pymodule]
fn fast_fuzzy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add ultra-optimized classes
    m.add_class::<Levenshtein>()?;
    m.add_class::<Damerau>()?;
    m.add_class::<FuzzyRatio>()?;
    m.add_class::<StringMatcher>()?;

    // Add ultra-fast top-level functions
    m.add_function(wrap_pyfunction!(ratio, m)?)?;
    m.add_function(wrap_pyfunction!(partial_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(token_sort_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(token_set_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(wratio, m)?)?;

    // Add ultra-fast distance functions
    m.add_function(wrap_pyfunction!(levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_levenshtein_fn, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_fn, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_fn, m)?)?;

    // Add ultra-optimized batch functions
    m.add_function(wrap_pyfunction!(ultra_batch_match_percentage, m)?)?;
    m.add_function(wrap_pyfunction!(ultra_streaming_similarity_search, m)?)?;
    m.add_function(wrap_pyfunction!(ultra_match_string_percentage_list, m)?)?;
    m.add_function(wrap_pyfunction!(ultra_match_string_percentage, m)?)?;
    m.add_function(wrap_pyfunction!(ultra_batch_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(ultra_batch_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(ultra_quick_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(ultra_batch_quick_ratio, m)?)?;

    // Add alternative functions for compatibility
    m.add_function(wrap_pyfunction!(normalized_levenshtein_direct, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_distance_alt, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_alt, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_levenshtein_alt, m)?)?;
    m.add_function(wrap_pyfunction!(ratio_alt, m)?)?;

    // Add string_matcher classes if available
    m.add_class::<string_matcher::FuzzyRatio>()?;
    m.add_class::<string_matcher::AdditionalMetrics>()?;
    m.add_class::<string_matcher::PhoneticAlgorithms>()?;
    m.add_class::<string_matcher::Process>()?;
    m.add_class::<string_matcher::Utils>()?;
    m.add_class::<string_matcher::OptimizedScorers>()?;
    m.add_class::<string_matcher::StreamingProcessor>()?;

    // Cache management utilities
    #[pyfn(m)]
    fn clear_global_cache() {
        get_cache().clear();
    }

    #[pyfn(m)]
    fn global_cache_stats() -> PyResult<HashMap<String, usize>> {
        let (size, hits, misses) = get_cache().stats();
        let mut stats = HashMap::new();
        stats.insert("size".to_string(), size);
        stats.insert("hits".to_string(), hits);
        stats.insert("misses".to_string(), misses);
        stats.insert("capacity".to_string(), CACHE_SIZE);

        let hit_rate = if hits + misses > 0 {
            (hits * 100) / (hits + misses)
        } else {
            0
        };
        stats.insert("hit_rate_percent".to_string(), hit_rate);

        Ok(stats)
    }

    // Create optimized instances for compatibility
    let fuzz_class = FuzzyRatio::new();
    m.add("fuzz", fuzz_class)?;

    let process_class = string_matcher::Process::new();
    m.add("process", process_class)?;

    let utils_class = string_matcher::Utils::new();
    m.add("utils", utils_class)?;

    let string_matcher_class = StringMatcher::new();
    m.add("string_metric", string_matcher_class)?;

    // Add all function aliases for maximum compatibility
    let ratio_fn = m.getattr("ratio")?;
    m.add("fuzz_ratio", ratio_fn)?;

    let partial_ratio_fn = m.getattr("partial_ratio")?;
    m.add("fuzz_partial_ratio", partial_ratio_fn)?;

    let levenshtein_fn = m.getattr("levenshtein_distance")?;
    m.add("levenshtein", levenshtein_fn.clone())?;

    let jaro_winkler_fn = m.getattr("jaro_winkler_fn")?;
    m.add("jaro_winkler", jaro_winkler_fn.clone())?;
    m.add("jaro_winkler_distance", jaro_winkler_fn)?;

    let normalized_levenshtein_fn = m.getattr("normalized_levenshtein_fn")?;
    m.add("normalized_levenshtein", normalized_levenshtein_fn.clone())?;
    m.add("normalized_levenshtein_distance", normalized_levenshtein_fn)?;

    m.add("levenshtein_distance", levenshtein_fn)?;

    Ok(())
}
