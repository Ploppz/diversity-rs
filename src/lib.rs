use ndarray::prelude::*;

#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;


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
pub fn unanchored_l2_discrepancy(points: &Array2<f64>) -> f64 {
    assert!(points.len() > 0);
    let n_points = points.len_of(Axis(0));
    let n_dim = points.len_of(Axis(1));
    let dims = n_dim as f64;
    assert!(points.iter().all(|x| *x >= 0.0 && *x <= 1.0));
    if n_points == 0 {
        return 0.0
    }
    
    let t3 = 12.0_f64.powf(-dims as f64);
    let mut t2 = 0.0;

    for n in 0..n_points {
        let mut t2_temp = 1.0;
        for d in 0..n_dim {
            t2_temp *= points[[n,d]] * (1.0 - points[[n,d]]) ;
        }
        t2 += t2_temp;
    }
    t2 = 2.0_f64.powf(1.0 - dims as f64) * t2 / n_points as f64;

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
    t1 = t1 / n_points.pow(2) as f64;
    (t1 - t2 + t3).sqrt()
}

#[cfg(test)]
mod tests {
    use ndarray::prelude::*;
    use rand::{Rng};
    use crate::*;
    use ndarray_rand::RandomExt;
    #[test]
    fn uniform() {
        let mut rng = rand::thread_rng();
        const N: usize = 1000;
        const D: usize = 2;
        let points = Array2::<f64>::random((N, D), rand::distributions::Uniform::from(0.0..0.1));
        let a = unanchored_l2_discrepancy(&points);
        assert!(a < 0.09);
        assert!(a > 1.0/N as f64);
    }

    #[test]
    fn concentrated() {
        const N: usize = 1000;
        const D: usize = 2;
        let points = Array2::<f64>::from_elem((N, D), 0.5);
        let a = unanchored_l2_discrepancy(&points);
        assert_approx_eq!(a, 0.195434);
    }
    #[test]
    fn stability() {
        let mut points = Array2::zeros((3, 2));
        points[[0,0]] = 0.5; points[[0, 1]] = 0.5;
        points[[1,0]] = 0.2; points[[1, 1]] = 0.8;
        points[[2,0]] = 0.6; points[[2, 1]] = 0.3;
        let a = unanchored_l2_discrepancy(&points);
        assert_approx_eq!(a, 0.0959456);
    }
}
