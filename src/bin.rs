use ngrams::Ngram;

fn main() {
    let grams: Vec<_> = "one two three".split(' ').ngrams(2).collect();
    let mut temp_str :Vec<_> = vec![];
    println!("{:?}", grams);

    for i in grams.iter(){
        let mut t = "".to_string();
        for j in i.to_vec(){
            t.push_str(j);
            t.push_str(" ");
        }
        t = t.trim().to_string();
        temp_str.push(t);
    }
    println!("{:?}", temp_str);
}