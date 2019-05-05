//! Matrices are layed out with rows-first in memory.

use num::{traits::*, *};

pub fn mat_mul<S: Num + Copy>(
    a: &[S],
    a_rows: usize,
    a_cols: usize,
    b: &[S],
    b_rows: usize,
    b_cols: usize,
) -> Vec<S> {
    let out_rows = a_rows;
    let out_cols = b_cols;
    let mut out = vec![S::zero(); out_rows * out_cols];

    let b = &transpose(b, b_rows, b_cols);
    write_mat_mul_transposed(&mut out, a, a_rows, a_cols, b, b_cols, b_rows);

    out
}

pub fn write_mat_mul<S: Num + Copy>(
    out: &mut [S],
    a: &[S],
    a_rows: usize,
    a_cols: usize,
    b: &[S],
    b_rows: usize,
    b_cols: usize,
) {
    debug_assert_eq!(a_cols, b_rows);
    debug_assert_eq!(a.len(), a_rows * a_cols);
    debug_assert_eq!(b.len(), b_rows * b_cols);
    debug_assert_eq!(out.len(), a_rows * b_cols);

    let out_rows = a_rows;
    let out_cols = b_cols;
    for row in 0..out_rows {
        for col in 0..out_cols {
            let mut sum = S::zero();

            for i in 0..a_cols {
                sum = sum + a[row * a_cols + i] * b[i * b_cols + col];
            }

            out[row * out_cols + col] = sum;
        }
    }
}

/// Perform matrix multiplication between a "normal" matrix `a` and a transposed matrix `b`
pub fn write_mat_mul_transposed<S: Num + Copy>(
    out: &mut [S],
    a: &[S],
    a_rows: usize,
    a_cols: usize,
    b: &[S],
    b_rows: usize,
    b_cols: usize,
) {
    debug_assert_eq!(a_cols, b_cols);
    debug_assert_eq!(a.len(), a_rows * a_cols);
    debug_assert_eq!(b.len(), b_rows * b_cols);
    debug_assert_eq!(out.len(), a_rows * b_rows);

    let out_rows = a_rows;
    let out_cols = b_rows;

    for row in 0..out_rows {
        let s = row * a_cols;
        let a_row = &a[s..s + a_cols];

        for col in 0..out_cols {
            let s = col * b_cols;
            let b_row = &b[s..s + b_cols];

            out[row * out_cols + col] = dot(a_row, b_row);
        }
    }
}

pub fn transpose<S: Num + Copy>(a: &[S], rows: usize, cols: usize) -> Vec<S> {
    debug_assert_eq!(a.len(), rows * cols);
    let mut out = vec![S::zero(); rows * cols];

    write_transpose(&mut out, a, rows, cols);

    out
}

pub fn write_transpose<S: Num + Copy>(out: &mut [S], a: &[S], rows: usize, cols: usize) {
    debug_assert_eq!(a.len(), rows * cols);
    debug_assert_eq!(out.len(), rows * cols);

    for col in 0..cols {
        for row in 0..rows {
            out[col * rows + row] = a[row * cols + col];
        }
    }
}

pub fn dot<S: Num + Copy>(a: &[S], b: &[S]) -> S {
    debug_assert!(a.len() == b.len());

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
        let a = &[1, 3, 7];
        let b = &[1, -2, 2];

        let result = dot(a, b);
        assert_eq!(result, 9);
    }

    #[test]
    fn transpose_write_small_odd_square() {
        let matrix = &mut [0, 1, 2, 3, 4, 5, 6, 7, 8];

        let out = &mut [0; 3 * 3];
        write_transpose(out, matrix, 3, 3);

        assert_eq!(out, &[0, 3, 6, 1, 4, 7, 2, 5, 8,])
    }

    #[test]
    fn transpose_write_small_odd_rect() {
        let matrix = &mut [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        let out = &mut [0; 2 * 5];
        write_transpose(out, matrix, 2, 5);

        assert_eq!(out, &[0, 5, 1, 6, 2, 7, 3, 8, 4, 9,])
    }

    #[test]
    fn transpose_small_odd_square() {
        let matrix = &mut [0, 1, 2, 3, 4, 5, 6, 7, 8];

        let out = transpose(matrix, 3, 3);

        assert_eq!(&out, &[0, 3, 6, 1, 4, 7, 2, 5, 8,])
    }

    #[test]
    fn transpose_small_odd_rect() {
        let matrix = &mut [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        let out = transpose(matrix, 2, 5);

        assert_eq!(&out, &[0, 5, 1, 6, 2, 7, 3, 8, 4, 9,])
    }

    #[test]
    fn mat_mul_write_small_rect() {
        let a = &[1, 2, 3, 4, 5, 6];
        let b = &[7, 8, 9, 10, 11, 12];

        let out = &mut [0; 2 * 2];
        write_mat_mul(out, a, 2, 3, b, 3, 2);

        assert_eq!(out, &[58, 64, 139, 154])
    }
    #[test]
    fn mat_mul_transposed_write_small_rect() {
        let a = &[1, 2, 3, 4, 5, 6];
        let b = transpose(&[7, 8, 9, 10, 11, 12], 3, 2);

        let out = &mut [0; 2 * 2];
        write_mat_mul_transposed(out, a, 2, 3, &b, 2, 3);

        assert_eq!(out, &[58, 64, 139, 154])
    }
}

#[cfg(all(feature = "unstable", test))]
mod benches {
    extern crate test;
    use super::*;
    use test::*;

    #[bench]
    fn transpose_matrix_write(b: &mut Bencher) {
        const ROWS: usize = 256;
        const COLS: usize = 319;

        let mut out = &mut [0; ROWS * COLS];
        let matrix: Vec<_> = (0..ROWS * COLS).collect();

        b.iter(|| {
            write_transpose(out, &matrix, ROWS, COLS);
            black_box(&out);
        })
    }

    #[bench]
    fn transpose_matrix_allocate(b: &mut Bencher) {
        const ROWS: usize = 256;
        const COLS: usize = 319;

        let matrix: Vec<_> = (0..ROWS * COLS).collect();

        b.iter(|| {
            let out = transpose(&matrix, ROWS, COLS);
            black_box(out);
        })
    }

    #[bench]
    fn mat_mul_write(bencher: &mut Bencher) {
        const A_ROWS: usize = 52;
        const A_COLS: usize = 127;
        const B_ROWS: usize = A_COLS;
        const B_COLS: usize = 42;

        let mut out = &mut [0f64; A_ROWS * B_COLS];

        let a: Vec<_> = (0..A_ROWS * A_COLS).map(|i| i as f64).collect();
        let b: Vec<_> = (0..B_ROWS * B_COLS).map(|i| i as f64).collect();

        bencher.iter(|| {
            write_mat_mul(out, &a, A_ROWS, A_COLS, &b, B_ROWS, B_COLS);
            black_box(&out);
        })
    }

    #[bench]
    fn mat_mul_transposed_write(bencher: &mut Bencher) {
        const A_ROWS: usize = 52;
        const A_COLS: usize = 127;
        const B_ROWS: usize = A_COLS;
        const B_COLS: usize = 42;

        let mut out = &mut [0f64; A_ROWS * B_COLS];

        let a: Vec<_> = (0..A_ROWS * A_COLS).map(|i| i as f64).collect();
        let b: Vec<_> = (0..B_ROWS * B_COLS).map(|i| i as f64).collect();

        bencher.iter(|| {
            let b = transpose(&b, B_ROWS, B_COLS);
            write_mat_mul_transposed(out, &a, A_ROWS, A_COLS, &b, B_COLS, B_ROWS);
            black_box(&out);
        })
    }

    #[bench]
    fn mat_mul_alloc(bencher: &mut Bencher) {
        const A_ROWS: usize = 52;
        const A_COLS: usize = 127;
        const B_ROWS: usize = A_COLS;
        const B_COLS: usize = 42;

        let a: Vec<_> = (0..A_ROWS * A_COLS).map(|i| i as f64).collect();
        let b: Vec<_> = (0..B_ROWS * B_COLS).map(|i| i as f64).collect();

        bencher.iter(|| {
            let b = transpose(&b, B_ROWS, B_COLS);
            let out = mat_mul(&a, A_ROWS, A_COLS, &b, B_ROWS, B_COLS);
            black_box(out);
        })
    }
}
