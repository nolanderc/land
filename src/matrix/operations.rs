use super::*;
use crate::vector::{operations::dot, Vector};

macro_rules! assert_equal_dimensions {
    ($lhs:expr, $rhs:expr) => {
        let lhs = $lhs.dimensions;
        let rhs = $rhs.dimensions;
        assert!(
            lhs == rhs,
            "Matrix dimensions must agree. Left hand side is of size {} and right hand side is of size {}",
            lhs,
            rhs
        )
    }
}

impl<S> Matrix<S>
where
    S: Scalar,
{
    /// Returns the transpose of a matrix
    pub fn transpose(&self) -> Matrix<S> {
        let mut out = Matrix::zeros(self.dimensions.transpose());

        for col in 0..self.dimensions.cols {
            for row in 0..self.dimensions.rows {
                out[[col, row]] = self[[row, col]];
            }
        }

        out
    }
}

// Matrix-Matrix Multiplication
impl<S> Mul<Self> for &Matrix<S>
where
    S: Scalar,
{
    type Output = Matrix<S>;

    // Standard matrix multiplication
    fn mul(self, rhs: Self) -> Matrix<S> {
        assert!(
            self.dimensions.cols == rhs.dimensions.rows,
            "Matrix dimensions must agree. Left hand side is {} and right hand side is {}",
            self.dimensions,
            rhs.dimensions,
        );

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

// Matrix-Vector Multiplication
impl<S> Mul<&Vector<S>> for &Matrix<S>
where
    S: Scalar,
{
    type Output = Vector<S>;

    // Standard matrix multiplication
    fn mul(self, rhs: &Vector<S>) -> Vector<S> {
        assert!(
            self.dimensions.cols == rhs.len(),
            "Matrix dimensions must agree. Left hand side is {} and right hand side has length {}",
            self.dimensions,
            rhs.len(),
        );

        let mut out = vec![S::zero(); self.dimensions.rows];

        for row in 0..self.dimensions.rows {
            let lhs_row = &self[row];

            out[row] = dot(lhs_row, &rhs);
        }

        out.into()
    }
}

impl<S: Scalar> Mul<Vector<S>> for Matrix<S> {
    type Output = Vector<S>;
    fn mul(self, rhs: Vector<S>) -> Vector<S> {
        (&self).mul(&rhs)
    }
}

impl<S: Scalar> Mul<&Vector<S>> for Matrix<S> {
    type Output = Vector<S>;
    fn mul(self, rhs: &Vector<S>) -> Vector<S> {
        (&self).mul(rhs)
    }
}

impl<S: Scalar> Mul<Vector<S>> for &Matrix<S> {
    type Output = Vector<S>;
    fn mul(self, rhs: Vector<S>) -> Vector<S> {
        self.mul(&rhs)
    }
}

macro_rules! impl_elementwise_operator {
    ($trait:ident, $fn:ident, $fn_assign:ident) => {
        impl<S> $trait<&Matrix<S>> for Matrix<S>
        where
            S: Scalar,
        {
            type Output = Matrix<S>;

            fn $fn(mut self, rhs: &Matrix<S>) -> Self::Output {
                self.$fn_assign(rhs);
                self
            }
        }

        impl<S> $trait<&Matrix<S>> for &Matrix<S>
        where
            S: Scalar,
        {
            type Output = Matrix<S>;

            fn $fn(self, rhs: &Matrix<S>) -> Self::Output {
                let mut tmp = self.clone();
                tmp.$fn_assign(rhs);
                tmp
            }
        }

        impl<S> $trait<Matrix<S>> for Matrix<S>
        where
            S: Scalar,
        {
            type Output = Matrix<S>;

            fn $fn(mut self, rhs: Matrix<S>) -> Self::Output {
                self.$fn_assign(&rhs);
                self
            }
        }

        impl<S> $trait<Matrix<S>> for &Matrix<S>
        where
            S: Scalar,
        {
            type Output = Matrix<S>;

            fn $fn(self, rhs: Matrix<S>) -> Self::Output {
                let mut tmp = self.clone();
                tmp.$fn_assign(&rhs);
                tmp
            }
        }
    };
}

impl_elementwise_operator!(Add, add, add_assign);
impl_elementwise_operator!(Sub, sub, sub_assign);

macro_rules! impl_elementwise_assign {
    ($trait:ident, $fn:ident) => {
        impl<S> $trait<Matrix<S>> for Matrix<S>
        where
            S: Scalar,
        {
            fn $fn(&mut self, rhs: Matrix<S>) {
                self.$fn(&rhs)
            }
        }

        impl<S> $trait<&Matrix<S>> for Matrix<S>
        where
            S: Scalar,
        {
            fn $fn(&mut self, rhs: &Matrix<S>) {
                assert_equal_dimensions!(self, rhs);

                for (a, b) in self.elements.iter_mut().zip(rhs.elements.iter()) {
                    a.$fn(*b)
                }
            }
        }
    };
}

impl_elementwise_assign!(AddAssign, add_assign);
impl_elementwise_assign!(SubAssign, sub_assign);

macro_rules! impl_scalar_operators {
    ($trait:ident, $fn:ident, ($($scalar:ty),+)) => (
        $(
            impl_scalar_operators!($trait, $fn, $scalar);
        )+
    );

    ($trait:ident, $fn:ident, $scalar:ty) => (
        impl $trait<$scalar> for Matrix<$scalar> {
            type Output = Matrix<$scalar>;

            fn $fn(mut self, rhs: $scalar) -> Self::Output {
                self.elements.iter_mut().for_each(|e| *e = e.$fn(rhs));
                self
            }
        }

        impl $trait<$scalar> for &Matrix<$scalar> {
            type Output = Matrix<$scalar>;

            fn $fn(self, rhs: $scalar) -> Self::Output {
                let mut tmp = self.clone();
                tmp.elements.iter_mut().for_each(|e| *e = e.$fn(rhs));
                tmp
            }
        }

        impl $trait<Matrix<$scalar>> for $scalar {
            type Output = Matrix<$scalar>;

            fn $fn(self, mut rhs: Matrix<$scalar>) -> Self::Output {
                rhs.elements.iter_mut().for_each(|e| *e = e.$fn(self));
                rhs
            }
        }

        impl $trait<&Matrix<$scalar>> for $scalar {
            type Output = Matrix<$scalar>;

            fn $fn(self, rhs: &Matrix<$scalar>) -> Self::Output {
                let mut rhs = rhs.clone();
                rhs.elements.iter_mut().for_each(|e| *e = e.$fn(self));
                rhs
            }
        }
    );
}

impl_scalar_operators!(
    Add,
    add,
    (i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64)
);
impl_scalar_operators!(
    Sub,
    sub,
    (i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64)
);
impl_scalar_operators!(
    Mul,
    mul,
    (i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64)
);
impl_scalar_operators!(
    Div,
    div,
    (i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64)
);

#[cfg(test)]
mod tests {
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

    #[test]
    fn matrix_vector_multiplication() {
        let a = mat![[1, 2, 3], [4, 5, 6]];
        let b = mat![1, 2, 3];

        let result = &a * &b;

        assert_eq!(result, mat![1 * 1 + 2 * 2 + 3 * 3, 4 * 1 + 5 * 2 + 6 * 3]);
    }
}
