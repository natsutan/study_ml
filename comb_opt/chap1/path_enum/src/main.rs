use std::collections::HashSet;

type Pi = Vec<u32>;

fn calc_k(pi:&Pi, i:u32, n:u32) -> u32 {

    // pi(i) + 1 .. n
    let g0: HashSet<u32> = (pi[i as usize - 1] + 1.. n + 2).collect();
    //println!("g0 = {:?}", g0);
    // pi(1) .. pi(i-1)
    let g1: HashSet<u32> = pi[0 .. (i - 1) as usize].to_vec().into_iter().collect();    
    //println!("g1 = {:?}", g1);

    let diff = &g0 - &g1;
    match diff.into_iter().min() {
        Some(k) => k,
        _ => 0
    }   
}

fn enumlation(n: u32) -> Vec<Pi> {
    let mut pi : Pi = (1..n+1).collect();
    let mut i = n - 1;  
    let mut result :  Vec<Vec<u32>> = Vec::new();
    let mut k = calc_k(&pi, i , n);

    result.push(pi.clone());

    // k == n + 1, i == 1
    while k != n + 1 || i != 1  {
        if k <= n {
            pi[i as usize - 1] = k;
            if i == n {
                result.push(pi.clone());
            }
            if i < n  {
                pi[i as usize] = 0;
                i = i + 1;
            }
        }
        if k == n + 1 {
            i = i - 1;
        }
        k = calc_k(&pi, i , n);
    }
    
    result
        
}

#[test]
fn test_k() {
    assert_eq!(calc_k(&vec![1,2,3,4,5,6], 5, 6), 6);
    assert_eq!(calc_k(&vec![1,2,3,4,6,5], 5, 6), 7);
    assert_eq!(calc_k(&vec![1,2,3,4,6,5], 4, 6), 5);
}


fn main() {
    let ret = enumlation(4);
    println!("len = {}", ret.len());
    println!("{:?}", ret);
}
