extern crate crossbeam;
extern crate crossbeam_channel;
extern crate num_cpus;

use std::collections::HashMap;
use std::thread;
use std::time::Duration;

use crossbeam_channel::bounded;
use regex::Regex;
use strsim::normalized_levenshtein;

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

#[test]
fn check_string_processing() {
    let strings = ["new york mets - atlanta braves".to_string(),
        "Cães danados".to_string(),
        "New York //// Mets $$$".to_string(), "Ça va?".to_string()];
    let obj = StringProcessing::new();
    let regex = Regex::new(r"(?ui)[\W]").unwrap();
    for i in strings.iter() {
        let proc_string = obj.replace_non_letters_non_numbers_with_whitespace(i.to_string());
        for expr in regex.find_iter(&*proc_string) {
            let i = expr.as_str();
            assert_eq!(i.to_string(), " ");
        }
    }
}

#[test]
fn test_dont_condense_whitespace() {
    let s1 = "new york mets - atlanta braves";
    let s2 = "new york mets atlanta braves";
    let mut obj = StringProcessing::new();
    let p1 = obj.replace_non_letters_non_numbers_with_whitespace(s1.to_string());
    let p2 = obj.replace_non_letters_non_numbers_with_whitespace(s2.to_string());
    assert_ne!(p1, p2);
}

pub(crate) fn thread_process(source_string: String, target_string: Vec<String>) -> HashMap<String, f64> {
    let (snd1, rcv1) = bounded(1);
    let (snd2, rcv2) = bounded(1);
    let n_workers = num_cpus::get();
    let mut temp_hash_map = HashMap::new();
    let mut res_hash_map = HashMap::new();
    for i in target_string.iter() {
        temp_hash_map.insert(i.clone(), source_string.clone());
    }

    crossbeam::scope(|s| {
        // Producer thread
        s.spawn(|_| {
            for (i, j) in temp_hash_map.iter() {
                snd1.send((i.clone(), j.clone())).unwrap();
            }
            // Close the channel - this is necessary to exit
            // the for-loop in the worker
            drop(snd1);
        });

        // Parallel processing by 2 threads
        for _ in 0..n_workers {
            // Send to sink, receive from source
            let (sendr, recvr) = (snd2.clone(), rcv1.clone());
            // Spawn workers in separate threads
            s.spawn(move |_| {
                thread::sleep(Duration::from_millis(500));
                // Receive until channel closes
                for (a, b) in recvr.iter() {
                    let rez = normalized_levenshtein(a.as_str(), b.as_str());
                    sendr.send((a, rez)).unwrap();
                }
            });
        }
        // Close the channel, otherwise sink will never
        // exit the for-loop
        drop(snd2);

        // Sink
        for (msg, val) in rcv2.iter() {
            res_hash_map.insert(msg, val);
        }
    }).unwrap();
    res_hash_map
}
