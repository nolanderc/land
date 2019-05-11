use super::*;

impl<S> Matrix<S>
where
    S: Scalar,
{
    /// Returns the transpose of a matrix
    pub fn transpose(&self) -> Matrix<S> {
        let mut out = Matrix::zeros(self.dimensions.transpose());

        for col in 0..self.dimensions.cols {
            for row in 0..self.dimensions.rows {
                out[col][row] = self[row][col];
            }
        }

        out
    }
}

pub fn dot<S>(lhs: &[S], rhs: &[S]) -> S
where
    S: Scalar,
{
    lhs.iter()
        .zip(rhs)
        .map(|(&a, &b)| a * b)
        .fold(S::zero(), |acc, t| acc + t)
}

// Multiplication
impl<S> Mul<Self> for &Matrix<S>
where
    S: Scalar,
{
    type Output = Matrix<S>;

    // Standard matrix multiplication
    fn mul(self, rhs: Self) -> Matrix<S> {
        let rhs_transpose = rhs.transpose();

        let out_dimensions = Dimensions {
            rows: self.dimensions.rows,
            cols: rhs.dimensions.cols,
        };
        let mut out = Matrix::zeros(out_dimensions);

        for row in 0..out_dimensions.rows {
            let lhs_row = &self[row];

            for col in 0..out_dimensions.cols {
                let rhs_col = &rhs_transpose[col];

                out[row][col] = dot(lhs_row, rhs_col);
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_square() {
        let a = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];

        let result = a.transpose();

        assert_eq!(result, mat![[1, 4, 7], [2, 5, 8], [3, 6, 9]]);
    }

    #[test]
    fn transpose_rect() {
        let a = mat![[1, 2, 3], [4, 5, 6]];

        let result = a.transpose();

        assert_eq!(result, mat![[1, 4], [2, 5], [3, 6]]);
    }

    #[test]
    fn matrix_multiplication_small_rect() {
        let a = mat![[1, 2, 3], [4, 5, 6]];
        let b = mat![[1, 2, 3], [4, 5, 6], [7, 8, 9]];

        let result = &a * &b;

        assert_eq!(result, mat![[30, 36, 42], [66, 81, 96]]);
    }

    #[test]
    fn matrix_multiplication_small_square() {
        let a = mat![[1, 2], [3, 4]];
        let b = mat![[5, 6], [7, 8]];

        let result = &a * &b;

        assert_eq!(
            result,
            mat![
                [1 * 5 + 2 * 7, 1 * 6 + 2 * 8],
                [3 * 5 + 4 * 7, 3 * 6 + 4 * 8]
            ]
        );
    }
}
