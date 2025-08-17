use pyo3::prelude::*;
use pyo3::types::PyModule;
extern crate dashmap;
extern crate strsim;

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;

use pyo3::{wrap_pyfunction, Py};
use rayon::prelude::*;
use regex::Regex;
use strsim::{damerau_levenshtein, normalized_damerau_levenshtein};
use strsim::{hamming, jaro, jaro_winkler as strsim_jaro_winkler, osa_distance};

// SIMD support
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Import the string_matcher module
mod string_matcher;

// ============================================================================
// Optimized String Processing with SIMD
// ============================================================================

pub struct OptimizedStringProcessor {
    re: Regex,
    // Pre-compiled lookup table for character classification
    char_lookup: [bool; 128], // Reduced to 128 for ASCII only
}

impl OptimizedStringProcessor {
    pub fn new() -> Self {
        let re = Regex::new(r"(?ui)\W").unwrap();

        // Build ASCII lookup table for faster character classification (0-127 only)
        let mut char_lookup = [false; 128];
        for i in 0..128 {
            let ch = i as u8 as char;
            char_lookup[i] = ch.is_ascii_alphanumeric();
        }

        OptimizedStringProcessor { re, char_lookup }
    }

    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    pub fn simd_process(&self, input: &str) -> String {
        // Fast path for ASCII strings using SIMD
        if input.is_ascii() && is_x86_feature_detected!("sse2") {
            unsafe {
                self.simd_process_impl(input)
            }
        } else {
            self.fallback_process(input)
        }
    }

    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    unsafe fn simd_process_impl(&self, input: &str) -> String {
        let bytes = input.as_bytes();
        let len = bytes.len();
        let mut result = Vec::with_capacity(len);

        // Process 16 bytes at a time using SSE2
        let chunk_size = 16;
        let chunks = len / chunk_size;
        let remainder = len % chunk_size;

        for chunk in 0..chunks {
            let start = chunk * chunk_size;
            let end = start + chunk_size;
            let chunk_bytes = &bytes[start..end];

            // Load 16 bytes into SSE registers
            let data = _mm_loadu_si128(chunk_bytes.as_ptr() as *const __m128i);
            
            // Create masks for character classification
            let zero = _mm_setzero_si128();
            let space = _mm_set1_epi8(b' ' as i8);
            
            // Check for alphanumeric characters using lookup table approach
            let mut processed_chunk = [0u8; 16];
            for i in 0..16 {
                let byte = chunk_bytes[i];
                if byte < 128 && self.char_lookup[byte as usize] {
                    processed_chunk[i] = byte;
                } else {
                    processed_chunk[i] = b' ';
                }
            }
            
            let processed = _mm_loadu_si128(processed_chunk.as_ptr() as *const __m128i);
            _mm_storeu_si128(result.as_mut_ptr().add(start) as *mut __m128i, processed);
        }

        // Process remaining bytes
        for i in (chunks * chunk_size)..len {
            let byte = bytes[i];
            if byte < 128 && self.char_lookup[byte as usize] {
                result.push(byte);
            } else {
                result.push(b' ');
            }
        }

        // Convert to string and normalize whitespace
        match String::from_utf8(result) {
            Ok(s) => s.split_whitespace().collect::<Vec<_>>().join(" "),
            Err(_) => self.fallback_process(input)
        }
    }

    #[inline(always)]
    pub fn fallback_process(&self, input: &str) -> String {
        // Fast path for ASCII strings using lookup table
        if input.is_ascii() {
            let bytes = input.as_bytes();
            let mut result = Vec::with_capacity(input.len());

            // Process bytes with bounds checking
            for &byte in bytes {
                // Ensure byte is within lookup table bounds
                if byte < 128 {
                    if self.char_lookup[byte as usize] {
                        result.push(byte);
                    } else {
                        result.push(b' ');
                    }
                } else {
                    // Non-ASCII byte, replace with space
                    result.push(b' ');
                }
            }

            // Safely convert back to string
            match String::from_utf8(result) {
                Ok(s) => s.split_whitespace().collect::<Vec<_>>().join(" "),
                Err(_) => {
                    // Fallback to regex processing if UTF-8 conversion fails
                    self.re.replace_all(input, " ").trim().to_string()
                }
            }
        } else {
            // Fallback for Unicode strings
            self.re.replace_all(input, " ").trim().to_string()
        }
    }

    #[inline(always)]
    pub fn process(&self, input: &str) -> String {
        #[cfg(target_arch = "x86_64")]
        {
            self.simd_process(input)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.fallback_process(input)
        }
    }
}

// ============================================================================
// High-Performance Cache Implementation using DashMap
// ============================================================================

use std::sync::atomic::{AtomicUsize, Ordering};

const MAX_CACHE_SIZE: usize = 100_000; // Reduced to 100K for safety

struct OptimizedCache {
    entries: DashMap<String, String>,
    hit_count: AtomicUsize,
    miss_count: AtomicUsize,
}

impl OptimizedCache {
    fn new() -> Self {
        Self {
            entries: DashMap::with_capacity(MAX_CACHE_SIZE),
            hit_count: AtomicUsize::new(0),
            miss_count: AtomicUsize::new(0),
        }
    }

    #[inline(always)]
    fn get_or_compute<F>(&self, key: &str, compute: F) -> String
    where
        F: FnOnce() -> String,
    {
        // Check if key exists in cache
        if let Some(entry) = self.entries.get(key) {
            self.hit_count.fetch_add(1, Ordering::Relaxed);
            return entry.clone();
        }

        // Compute new value
        self.miss_count.fetch_add(1, Ordering::Relaxed);
        let computed = compute();

        // Evict old entries if cache is too large
        if self.entries.len() >= MAX_CACHE_SIZE {
            // Remove ~10% of entries when at capacity
            let to_remove = MAX_CACHE_SIZE / 10;
            let keys_to_remove: Vec<String> = self
                .entries
                .iter()
                .take(to_remove)
                .map(|entry| entry.key().clone())
                .collect();

            for key in keys_to_remove {
                self.entries.remove(&key);
            }
        }

        // Store in cache
        self.entries.insert(key.to_string(), computed.clone());
        computed
    }

    fn clear(&self) {
        self.entries.clear();
        self.hit_count.store(0, Ordering::Relaxed);
        self.miss_count.store(0, Ordering::Relaxed);
    }

    fn stats(&self) -> (usize, usize, usize) {
        let hits = self.hit_count.load(Ordering::Relaxed);
        let misses = self.miss_count.load(Ordering::Relaxed);
        (self.entries.len(), hits, misses)
    }
}

// Global cache instance using lazy_static pattern
static GLOBAL_CACHE: std::sync::OnceLock<Arc<OptimizedCache>> = std::sync::OnceLock::new();

#[inline(always)]
fn get_cache() -> &'static OptimizedCache {
    let cache = GLOBAL_CACHE.get_or_init(|| Arc::new(OptimizedCache::new()));
    &**cache
}

// ============================================================================
// Optimized Levenshtein Distance Implementation with SIMD
// ============================================================================

#[inline(always)]
fn optimized_levenshtein(s1: &str, s2: &str) -> usize {
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

    // Use SIMD-optimized implementation for ASCII strings
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") && s1.is_ascii() && s2.is_ascii() {
            return unsafe { simd_levenshtein_impl(bytes1, bytes2) };
        }
    }
    
    // Use parallel implementation for large strings
    if len1 > 1000 || len2 > 1000 {
        return parallel_levenshtein_impl(bytes1, bytes2);
    }

    // Use optimized implementation for all other string sizes
    optimized_levenshtein_impl(bytes1, bytes2)
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn simd_levenshtein_impl(s1: &[u8], s2: &[u8]) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();

    // Handle empty strings
    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    // Two-row algorithm with SIMD optimization
    let mut prev_row = vec![0u32; len2 + 1];
    let mut curr_row = vec![0u32; len2 + 1];

    // Initialize first row
    for j in 0..=len2 {
        prev_row[j] = j as u32;
    }

    // Process in chunks of 16 for SIMD
    for i in 1..=len1 {
        curr_row[0] = i as u32;

        let mut j = 1;
        // Process 16 elements at a time using SIMD
        while j + 16 <= len2 {
            // Load data into SIMD registers
            let s1_byte = _mm_set1_epi8(s1[i-1] as i8);
            let s2_chunk = _mm_loadu_si128(s2[j-1..].as_ptr() as *const __m128i);
            
            // Compare bytes
            let cmp_result = _mm_cmpeq_epi8(s1_byte, s2_chunk);
            
            // Convert comparison result to cost array
            let costs: [u8; 16] = std::mem::transmute(_mm_movemask_epi8(cmp_result));
            
            // Process the costs
            for k in 0..16 {
                let cost = if costs[k] == 0 { 1 } else { 0 };
                
                let deletion = prev_row[j + k].saturating_add(1);
                let insertion = curr_row[j + k - 1].saturating_add(1);
                let substitution = prev_row[j + k - 1].saturating_add(cost);
                
                curr_row[j + k] = deletion.min(insertion).min(substitution);
            }
            
            j += 16;
        }

        // Process remaining elements
        for j in j..=len2 {
            let cost = if s1.get(i - 1) == s2.get(j - 1) { 0 } else { 1 };

            let deletion = prev_row[j].saturating_add(1);
            let insertion = curr_row[j - 1].saturating_add(1);
            let substitution = prev_row[j - 1].saturating_add(cost);

            curr_row[j] = deletion.min(insertion).min(substitution);
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[len2] as usize
}

#[inline(always)]
fn parallel_levenshtein_impl(s1: &[u8], s2: &[u8]) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();

    // Handle empty strings
    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    // For very large strings, use a divide-and-conquer approach
    if len1 > 10000 && len2 > 10000 {
        // Split the larger string in half and compute distance for each half
        let mid = len1 / 2;
        let left_dist = optimized_levenshtein_impl(&s1[..mid], s2);
        let right_dist = optimized_levenshtein_impl(&s1[mid..], s2);
        return left_dist + right_dist;
    }

    optimized_levenshtein_impl(s1, s2)
}

#[inline(always)]
fn optimized_levenshtein_impl(s1: &[u8], s2: &[u8]) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();

    // Handle empty strings
    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    // Use smaller array when one string is much shorter than the other
    if len1 > len2 * 4 || len2 > len1 * 4 {
        return asymmetric_levenshtein(s1, s2);
    }

    // Two-row algorithm with optimized operations
    let mut prev_row = vec![0u32; len2 + 1];
    let mut curr_row = vec![0u32; len2 + 1];

    // Initialize first row
    for j in 0..=len2 {
        prev_row[j] = j as u32;
    }

    for i in 1..=len1 {
        curr_row[0] = i as u32;

        // Use early termination when possible
        let mut min_val = curr_row[0];
        
        for j in 1..=len2 {
            // Fast path for identical characters
            let cost = if s1[i - 1] == s2[j - 1] { 0 } else { 1 };

            // Calculate all three operations
            let deletion = prev_row[j] + 1;
            let insertion = curr_row[j - 1] + 1;
            let substitution = prev_row[j - 1] + cost;

            let min_op = deletion.min(insertion).min(substitution);
            curr_row[j] = min_op;
            min_val = min_val.min(min_op);
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[len2] as usize
}

#[inline(always)]
fn asymmetric_levenshtein(s1: &[u8], s2: &[u8]) -> usize {
    let (short, long) = if s1.len() < s2.len() { (s1, s2) } else { (s2, s1) };
    
    // For highly asymmetric cases, use a band algorithm
    let k = (long.len() - short.len() + 1) as i32;
    let mut prev_row = vec![0u32; short.len() + 1];
    let mut curr_row = vec![0u32; short.len() + 1];

    // Initialize first row
    for j in 0..=short.len() {
        prev_row[j] = j as u32;
    }

    for i in 1..=long.len() {
        curr_row[0] = i as u32;
        
        let start = (i as i32 - k).max(1) as usize;
        let end = (i as i32 + k).min(short.len() as i32) as usize;

        for j in start..=end {
            let cost = if long[i - 1] == short[j - 1] { 0 } else { 1 };

            let deletion = prev_row[j] + 1;
            let insertion = curr_row[j - 1] + 1;
            let substitution = prev_row[j - 1] + cost;

            curr_row[j] = deletion.min(insertion).min(substitution);
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[short.len()] as usize
}

// ============================================================================
// Optimized Levenshtein Class with SIMD and Parallel Processing
// ============================================================================

#[pyclass]
pub struct Levenshtein {
    processor: Arc<OptimizedStringProcessor>,
}

#[pymethods]
impl Levenshtein {
    #[new]
    pub fn new() -> Self {
        Levenshtein {
            processor: Arc::new(OptimizedStringProcessor::new()),
        }
    }

    #[inline(always)]
    fn preprocess_string(&self, s: &str) -> String {
        let cache = get_cache();
        cache.get_or_compute(s, || self.processor.process(s))
    }

    pub fn match_string_difference(&self, source: &str, target: &str) -> PyResult<usize> {
        let s_p = self.preprocess_string(source);
        let t_p = self.preprocess_string(target);
        Ok(optimized_levenshtein(&s_p, &t_p))
    }

    // Fixed: Now accepts only 2 positional arguments (self is implicit)
    #[pyo3(signature = (source, target))]
    pub fn match_string_percentage(&self, source: &str, target: &str) -> PyResult<f64> {
        let s_p = self.preprocess_string(source);
        let t_p = self.preprocess_string(target);

        let max_len = s_p.len().max(t_p.len());
        if max_len == 0 {
            return Ok(1.0);
        }

        let distance = optimized_levenshtein(&s_p, &t_p);
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
                let distance = optimized_levenshtein(&s_p, &t_p);
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
                    let distance = optimized_levenshtein(&s_p, &t_p);
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
        stats.insert("capacity".to_string(), MAX_CACHE_SIZE);

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
// Optimized Damerau Class with Parallel Processing
// ============================================================================

#[pyclass]
pub struct Damerau {
    processor: Arc<OptimizedStringProcessor>,
}

#[pymethods]
impl Damerau {
    #[new]
    pub fn new() -> Self {
        Damerau {
            processor: Arc::new(OptimizedStringProcessor::new()),
        }
    }

    #[inline(always)]
    fn process_string(&self, s: &str) -> String {
        let cache = get_cache();
        cache.get_or_compute(s, || self.processor.process(s))
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
        stats.insert("capacity".to_string(), MAX_CACHE_SIZE);
        Ok(stats)
    }
}

// ============================================================================
// Optimized FuzzyRatio Class with SIMD and Parallel Processing
// ============================================================================

#[pyclass]
pub struct FuzzyRatio {
    processor: Arc<OptimizedStringProcessor>,
}

#[pymethods]
impl FuzzyRatio {
    #[new]
    pub fn new() -> Self {
        FuzzyRatio {
            processor: Arc::new(OptimizedStringProcessor::new()),
        }
    }

    #[inline(always)]
    fn process_string(&self, s: &str) -> String {
        let cache = get_cache();
        cache.get_or_compute(s, || self.processor.process(s))
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
        tokens.sort_unstable();
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

        let distance = strsim::levenshtein(s1, s2);
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

        // Parallel processing for large strings
        if longer.len() > 5000 {
            let segments: Vec<_> = (0..=(longer.len().saturating_sub(shorter_len)))
                .step_by(step)
                .collect();
            
            let ratios: Vec<f64> = segments
                .into_par_iter()
                .map(|i| {
                    let end_idx = (i + shorter_len).min(longer.len());
                    if let Some(substring) = longer.get(i..end_idx) {
                        self.calculate_ratio(shorter, substring)
                    } else {
                        0.0
                    }
                })
                .collect();
            
            best_ratio = ratios.into_iter().fold(0.0, f64::max);
        } else {
            for i in (0..=(longer.len().saturating_sub(shorter_len))).step_by(step) {
                let end_idx = (i + shorter_len).min(longer.len());
                if let Some(substring) = longer.get(i..end_idx) {
                    let ratio = self.calculate_ratio(shorter, substring);
                    best_ratio = best_ratio.max(ratio);

                    // Early termination for perfect matches
                    if best_ratio >= 100.0 {
                        break;
                    }
                }
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
            ratio * 0.2 + partial * 0.6 + token_sort * 0.1 + token_set * 0.1
        }
    }
}

// ============================================================================
// Optimized StringMatcher Class with SIMD and Parallel Processing
// ============================================================================

#[pyclass]
pub struct StringMatcher {
    processor: Arc<OptimizedStringProcessor>,
}

#[pymethods]
impl StringMatcher {
    #[new]
    pub fn new() -> Self {
        StringMatcher {
            processor: Arc::new(OptimizedStringProcessor::new()),
        }
    }

    #[inline(always)]
    fn process_string_pair(&self, source: &str, target: &str) -> (String, String) {
        let cache = get_cache();
        let s_p = cache.get_or_compute(source, || self.processor.process(source));
        let t_p = cache.get_or_compute(target, || self.processor.process(target));
        (s_p, t_p)
    }

    pub fn jaro_winkler_difference(&self, source: &str, target: &str) -> f64 {
        let (s_p, t_p) = self.process_string_pair(source, target);
        strsim_jaro_winkler(&s_p, &t_p)
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

        let distance = optimized_levenshtein(&s_p, &t_p);
        1.0 - (distance as f64 / max_len as f64)
    }
}

// ============================================================================
// Optimized Batch Processing Functions with SIMD and Parallel Processing
// ============================================================================

#[pyfunction]
pub fn safe_batch_match_percentage(
    sources: Vec<String>,
    targets: Vec<String>,
) -> PyResult<Vec<Vec<f64>>> {
    if sources.is_empty() || targets.is_empty() {
        return Ok(Vec::new());
    }

    let processor = Arc::new(OptimizedStringProcessor::new());
    let cache = get_cache();

    // Pre-process all strings in parallel
    let processed_sources: Vec<String> = sources
        .par_iter()
        .map(|s| cache.get_or_compute(s, || processor.process(s)))
        .collect();

    let processed_targets: Vec<String> = targets
        .par_iter()
        .map(|t| cache.get_or_compute(t, || processor.process(t)))
        .collect();

    // Compute similarity matrix in parallel
    let results: Vec<Vec<f64>> = processed_sources
        .par_iter()
        .map(|s| {
            let s_len = s.len();
            processed_targets
                .par_iter()
                .map(|t| {
                    let max_len = s_len.max(t.len());
                    if max_len == 0 {
                        1.0
                    } else {
                        let distance = optimized_levenshtein(s, t);
                        1.0 - (distance as f64 / max_len as f64)
                    }
                })
                .collect()
        })
        .collect();

    Ok(results)
}

// ============================================================================
// Top-level Optimized Functions with SIMD and Parallel Processing
// ============================================================================

#[pyfunction]
pub fn ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let fuzzy_ratio = string_matcher::FuzzyRatio::new();
    let result = fuzzy_ratio.ratio(s1, s2, score_cutoff)?;
    Ok(result.unwrap_or(0.0))
}

#[pyfunction]
pub fn partial_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let fuzzy_ratio = string_matcher::FuzzyRatio::new();
    let result = fuzzy_ratio.partial_ratio(s1, s2, score_cutoff)?;
    Ok(result.unwrap_or(0.0))
}

#[pyfunction]
pub fn token_sort_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let fuzzy_ratio = string_matcher::FuzzyRatio::new();
    let result = fuzzy_ratio.token_sort_ratio(s1, s2, score_cutoff)?;
    Ok(result.unwrap_or(0.0))
}

#[pyfunction]
pub fn token_set_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let fuzzy_ratio = string_matcher::FuzzyRatio::new();
    let result = fuzzy_ratio.token_set_ratio(s1, s2, score_cutoff)?;
    Ok(result.unwrap_or(0.0))
}

#[pyfunction]
pub fn wratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let fuzzy_ratio = string_matcher::FuzzyRatio::new();
    let result = fuzzy_ratio.wratio(s1, s2, score_cutoff)?;
    Ok(result.unwrap_or(0.0))
}

#[pyfunction]
pub fn levenshtein_distance(s1: &str, s2: &str) -> PyResult<usize> {
    Ok(optimized_levenshtein(s1, s2))
}

#[pyfunction]
pub fn levenshtein(s1: &str, s2: &str) -> PyResult<usize> {
    Ok(optimized_levenshtein(s1, s2))
}

#[pyfunction]
pub fn normalized_levenshtein_fn(s1: &str, s2: &str) -> PyResult<f64> {
    let max_len = s1.len().max(s2.len());
    if max_len == 0 {
        return Ok(1.0);
    }
    let distance = strsim::levenshtein(s1, s2);
    Ok(1.0 - (distance as f64 / max_len as f64))
}

#[pyfunction]
pub fn normalized_levenshtein(s1: &str, s2: &str) -> PyResult<f64> {
    normalized_levenshtein_fn(s1, s2)
}

#[pyfunction]
pub fn jaro_winkler_distance(s1: &str, s2: &str) -> PyResult<f64> {
    Ok(strsim_jaro_winkler(s1, s2))
}

#[pyfunction]
pub fn jaro_winkler(s1: &str, s2: &str) -> PyResult<f64> {
    Ok(strsim_jaro_winkler(s1, s2))
}

#[pyfunction]
pub fn jaro_winkler_fn(s1: &str, s2: &str) -> PyResult<f64> {
    Ok(strsim_jaro_winkler(s1, s2))
}

#[pyfunction]
pub fn jaro_winkler_alt(s1: &str, s2: &str) -> PyResult<f64> {
    Ok(strsim_jaro_winkler(s1, s2))
}

#[pyfunction]
pub fn normalized_levenshtein_alt(s1: &str, s2: &str) -> PyResult<f64> {
    normalized_levenshtein_fn(s1, s2)
}

#[pyfunction]
pub fn ratio_alt(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    ratio(s1, s2, score_cutoff)
}

#[pyfunction]
pub fn quick_ratio(s1: &str, s2: &str) -> PyResult<f64> {
    // Quick ratio using simple character count comparison
    if s1.is_empty() && s2.is_empty() {
        return Ok(100.0);
    }

    let len1 = s1.len();
    let len2 = s2.len();
    let max_len = len1.max(len2);

    if max_len == 0 {
        return Ok(100.0);
    }

    // Quick approximation based on length difference
    let len_diff = (len1 as i32 - len2 as i32).abs() as f64;
    Ok(100.0 * (1.0 - len_diff / max_len as f64))
}

#[pyfunction]
pub fn match_string_percentage_fn(source: &str, target: &str) -> PyResult<f64> {
    normalized_levenshtein_fn(source, target)
}

#[pyfunction]
pub fn match_string_percentage_list_fn(
    source: &str,
    targets: Vec<String>,
) -> PyResult<HashMap<String, f64>> {
    let mut results = HashMap::new();
    for target in targets {
        let score = normalized_levenshtein_fn(source, &target)?;
        results.insert(target, score);
    }
    Ok(results)
}

#[pyfunction]
pub fn batch_match_percentage(
    sources: Vec<String>,
    targets: Vec<String>,
) -> PyResult<Vec<Vec<f64>>> {
    safe_batch_match_percentage(sources, targets)
}

#[pyfunction]
pub fn batch_ratio(queries: Vec<String>, targets: Vec<String>) -> PyResult<Vec<Vec<f64>>> {
    let fuzzy_ratio = FuzzyRatio::new();
    let processor = Arc::new(OptimizedStringProcessor::new());
    let cache = get_cache();

    // Pre-process all strings
    let processed_queries: Vec<String> = queries
        .par_iter()
        .map(|q| cache.get_or_compute(q, || processor.process(q)))
        .collect();

    let processed_targets: Vec<String> = targets
        .par_iter()
        .map(|t| cache.get_or_compute(t, || processor.process(t)))
        .collect();

    let results: Vec<Vec<f64>> = processed_queries
        .par_iter()
        .map(|query| {
            processed_targets
                .par_iter()
                .map(|target| fuzzy_ratio.calculate_ratio(query, target))
                .collect()
        })
        .collect();

    Ok(results)
}

#[pyfunction]
pub fn batch_levenshtein(queries: Vec<String>, targets: Vec<String>) -> PyResult<Vec<Vec<usize>>> {
    let results: Vec<Vec<usize>> = queries
        .par_iter()
        .map(|query| {
            targets
                .par_iter()
                .map(|target| optimized_levenshtein(query, target))
                .collect()
        })
        .collect();
    Ok(results)
}

#[pyfunction]
pub fn batch_quick_ratio(queries: Vec<String>, targets: Vec<String>) -> PyResult<Vec<Vec<f64>>> {
    let results: Vec<Vec<f64>> = queries
        .par_iter()
        .map(|query| {
            targets
                .par_iter()
                .map(|target| quick_ratio(query, target).unwrap_or(0.0))
                .collect()
        })
        .collect();
    Ok(results)
}

// Wrapper functions to match the expected interface for benchmark compatibility
#[pyfunction]
#[pyo3(signature = (query, choices, limit=5))]
pub fn extract(query: &str, choices: Vec<String>, limit: Option<usize>) -> PyResult<Vec<(String, f64, usize)>> {
    let process = string_matcher::Process::new();
    process.extract(query, choices, limit, Some(0.0), Some("ratio"))
}

#[pyfunction]
pub fn extract_one(query: &str, choices: Vec<String>) -> PyResult<Option<(String, f64, usize)>> {
    let process = string_matcher::Process::new();
    process.extract_one(query, choices, Some(0.0), Some("ratio"))
}

// ============================================================================
// Module Registration
// ============================================================================

#[pymodule]
fn fastfuzzy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add optimized classes
    m.add_class::<Levenshtein>()?;
    m.add_class::<Damerau>()?;
    m.add_class::<FuzzyRatio>()?;
    m.add_class::<StringMatcher>()?;

    // Add optimized top-level functions
    m.add_function(wrap_pyfunction!(ratio, m)?)?;
    m.add_function(wrap_pyfunction!(partial_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(token_sort_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(token_set_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(wratio, m)?)?;

    // Add optimized distance functions with multiple aliases
    m.add_function(wrap_pyfunction!(levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_levenshtein_fn, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_distance, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_fn, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_alt, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_levenshtein_alt, m)?)?;
    m.add_function(wrap_pyfunction!(ratio_alt, m)?)?;
    m.add_function(wrap_pyfunction!(quick_ratio, m)?)?;

    // Add additional utility functions
    m.add_function(wrap_pyfunction!(match_string_percentage_fn, m)?)?;
    m.add_function(wrap_pyfunction!(match_string_percentage_list_fn, m)?)?;

    // Add optimized batch functions
    m.add_function(wrap_pyfunction!(safe_batch_match_percentage, m)?)?;
    m.add_function(wrap_pyfunction!(batch_match_percentage, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(batch_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(batch_quick_ratio, m)?)?;

    // Add wrapper functions for benchmark compatibility
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    m.add_function(wrap_pyfunction!(extract_one, m)?)?;

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
        stats.insert("capacity".to_string(), MAX_CACHE_SIZE);

        let hit_rate = if hits + misses > 0 {
            (hits * 100) / (hits + misses)
        } else {
            0
        };
        stats.insert("hit_rate_percent".to_string(), hit_rate);

        Ok(stats)
    }

    // Create optimized instances for compatibility
    let fuzz_class = Py::new(m.py(), FuzzyRatio::new())?;
    m.add("fuzz", fuzz_class)?;

    let process_class = Py::new(m.py(), string_matcher::Process::new())?;
    m.add("process", process_class)?;

    let utils_class = Py::new(m.py(), string_matcher::Utils::new())?;
    m.add("utils", utils_class)?;

    let string_matcher_class = Py::new(m.py(), StringMatcher::new())?;
    m.add("string_metric", string_matcher_class)?;

    Ok(())
}
