use crate::{matrix::Matrix, traits::Scalar, vector::Vector};

/// Multiplies two row major matrices
pub fn mat_mul<S: Scalar>(lhs: &Matrix<S>, rhs: &Matrix<S>) -> Matrix<S> {
    assert!(
        lhs.cols_len() == rhs.rows_len(),
        "Matrix dimensions must agree"
    );
    let rhs = transpose(rhs);

    let out_rows = lhs.rows_len();
    let out_cols = rhs.rows_len();
    let mut out = vec![Vector::filled(S::zero(), out_cols); out_rows];

    for (i, lhs_row) in lhs.iter_rows().enumerate() {
        for (j, rhs_col) in rhs.iter_rows().enumerate() {
            out[i][j] = dot(lhs_row, rhs_col);
        }
    }

    Matrix { rows: out }
}

/// Multiplies a matrix with a column vector
pub fn mat_vec_mul<S: Scalar>(lhs: &Matrix<S>, rhs: &Vector<S>) -> Vector<S> {
    assert!(lhs.cols_len() == rhs.len(), "Matrix dimensions must agree");

    let out_len = lhs.rows_len();
    let mut out = Vector::filled(S::zero(), out_len);

    for (i, lhs_row) in lhs.iter_rows().enumerate() {
        out[i] = dot(lhs_row, rhs);
    }

    out
}

pub fn transpose<S: Scalar>(mat: &Matrix<S>) -> Matrix<S> {
    let rows = mat.rows_len();
    let cols = mat.cols_len();

    let mut out = vec![Vector::zeros(rows); cols];

    for i in 0..rows {
        for j in 0..cols {
            out[i][j] = mat[j][i];
        }
    }

    Matrix { rows: out }
}

pub fn dot<S: Scalar>(a: &Vector<S>, b: &Vector<S>) -> S {
    assert!(a.len() == b.len(), "Vectors must be of equal length");

    let mut sum = S::zero();
    for i in 0..a.len() {
        sum = sum + a[i] * b[i];
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_small_odd() {
        let a = Vector::from(vec![1, 3, 7]);
        let b = Vector::from(vec![1, -2, 2]);

        let result = dot(&a, &b);
        assert_eq!(result, 9);
    }

    #[test]
    fn matrix_multiplication_small_rect() {
        let a = mat![
            1, 2, 3;
            4, 5, 6
        ];
        let b = mat![
            1, 2, 3;
            4, 5, 6;
            7, 8, 9
        ];

        let result = mat_mul(&a, &b);

        assert_eq!(
            result,
            mat![
                30, 36, 42;
                66, 81, 96
            ]
        );
    }
}
