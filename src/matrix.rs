
use num::Num;

use std::ops::*;
use crate::ops::mat_mul;

pub struct Matrix<S: Num> {
    data: Vec<S>,
    rows: usize,
    cols: usize,
}

impl<S: Num + Copy> Mul<Self> for &Matrix<S> {
    type Output = Matrix<S>;

    fn mul(self, rhs: Self) -> Matrix<S> {
        let data = mat_mul(
            &self.data,
            self.rows,
            self.cols,
            &rhs.data,
            rhs.rows,
            rhs.cols
        );

        Matrix {
            data,
            rows: self.rows,
            cols: rhs.cols
        }
    }
}

