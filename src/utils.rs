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
