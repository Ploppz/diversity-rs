#![feature(test)]

extern crate test;
use ndarray::prelude::*;
use diversity::unanchored_l2_discrepancy;
use ndarray_rand::RandomExt;

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_discrepancy(b: &mut Bencher) {
        const N_POINTS: usize = 400;
        const N_DIM: usize = 5;
        let points = Array::random((N_POINTS, N_DIM), rand::distributions::Uniform::from(0.0..0.1));
        b.iter(|| {
            unanchored_l2_discrepancy(&points)
        });
    }
}
