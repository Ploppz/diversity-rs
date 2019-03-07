use ndarray::prelude::*;


macro_rules! max {
    {$x:expr $(, $y:expr)+} => {
        $x.max(max!($($y),+))
    };
    {$x:expr} => {
        $x
    }
}
macro_rules! min {
    {$x:expr $(, $y:expr)+} => {
        $x.min(min!($($y),+))
    };
    {$x:expr} => {
        $x
    }
}

// pub fn unanchored_l2_discrepancy<T: Index<usize, Output=f64>, I: Iterator<Item=T>>(points: I) -> f64 {
// ^^ TODO: actually can generalize using ExactSizeIterator, because it provides len()
pub fn unanchored_l2_discrepancy(points: &Array2<f32>) -> f32 {
    assert!(points.len() > 0);
    let n_points = points.len_of(Axis(0));
    let n_dim = points.len_of(Axis(1));
    let dims = n_dim as f32;
    assert!(points.iter().all(|x| *x >= 0.0 && *x <= 1.0));
    if n_points == 0 {
        return 0.0
    }
    
    let t3 = 12.0_f32.powf(-dims as f32);
    let mut t2 = 0.0;

    for n in 0..n_points {
        let mut t2_temp = 1.0;
        for d in 0..n_dim {
            t2_temp *= points[[n,d]] * (1.0 - points[[n,d]]) ;
        }
        t2 += t2_temp;
    }
    t2 = 2.0_f32.powf(1.0 - dims as f32) * t2 / n_points as f32;

    let mut t1 = 0.0;
    for n in 0..n_points {
        for m in 0..n_points {
            let mut t1_temp = 1.0;
            for d in 0..n_dim {
                t1_temp *= (1.0 - max!(points[[n,d]],points[[m,d]])) * min!(points[[n,d]], points[[m,d]]);
            }
            t1 += t1_temp;
        }
    }
    t1 = t1 / n_points.pow(2) as f32;
    (t1 - t2 + t3).sqrt()
}

/*
#[cfg(test)]
mod tests {
    use ndarray::prelude::*;
    use rand::{Rng};
    use crate::*;
    #[test]
    fn uniform() {
        let mut rng = rand::thread_rng();
        let N = 1000;
        let D = 2;
        let mut points = Array2::<f32>::zeros((N, D));
        points.iter_mut().map(|_| rng.gen::<f32>());
        let a = unanchored_L2_discrepancy(points);
        assert!(a > 1.0/N as f32);
        assert!(a < 0.09);
        println!("{}", a);
    }

    #[test]
    fn concentrated() {
        let mut rng = rand::thread_rng();
        let N = 1000;
        let D = 2;
        let mut points = Array2::<f32>::from_elem((N, D), 0.5);
        let a = unanchored_L2_discrepancy(points);
        println!("{}", a);
    }
}
*/
