use super::*;
use crate::matrix::Matrix;

macro_rules! assert_equal_length {
    ($lhs:expr, $rhs:expr) => {
        let lhs = $lhs.len();
        let rhs = $rhs.len();
        assert!(
            lhs == rhs,
            "Vectors must be of same length. Left hand side has length {} and right hand side has length {}",
            lhs,
            rhs
        )
    }
}

pub fn dot<S>(lhs: &[S], rhs: &[S]) -> S
where
    S: Scalar,
{
    assert_equal_length!(lhs, rhs);

    lhs.iter()
        .zip(rhs)
        .map(|(&a, &b)| a * b)
        .fold(S::zero(), |acc, t| acc + t)
}

impl<S> Vector<S>
where
    S: Scalar,
{
    pub fn dot(&self, other: &Self) -> S {
        dot(&self, &other)
    }

    /// Perform matrix multiplication between a column and row vector so that for
    /// `let m = a.mul_transpose(b)` the resulting matrix `m` fulfills `m[r][c] = a[r]*b[c]`
    pub fn mul_transpose(&self, other: &Vector<S>) -> Matrix<S> {
        let mut out = Matrix::zeros([self.len(), other.len()].into());

        for i in 0..self.len() {
            for j in 0..other.len() {
                out[i][j] = self[i] * other[j];
            }
        }

        out
    }
}

macro_rules! impl_elementwise {
    ($trait:ident, $fn:ident) => {
        impl<S> $trait<&Vector<S>> for Vector<S>
        where
            S: Scalar,
        {
            type Output = Vector<S>;

            fn $fn(mut self, rhs: &Vector<S>) -> Self::Output {
                assert_equal_length!(self, rhs);

                for i in 0..self.len() {
                    self[i] = self[i].$fn(rhs[i]);
                }

                self
            }
        }

        impl<S> $trait<Vector<S>> for &Vector<S>
        where
            S: Scalar,
        {
            type Output = Vector<S>;

            fn $fn(self, mut rhs: Vector<S>) -> Self::Output {
                assert_equal_length!(self, rhs);

                for i in 0..self.len() {
                    rhs[i] = self[i].$fn(rhs[i]);
                }

                rhs
            }
        }

        impl<S> $trait<&Vector<S>> for &Vector<S>
        where
            S: Scalar,
        {
            type Output = Vector<S>;

            fn $fn(self, rhs: &Vector<S>) -> Self::Output {
                self.clone().$fn(rhs)
            }
        }

        impl<S> $trait<Vector<S>> for Vector<S>
        where
            S: Scalar,
        {
            type Output = Vector<S>;

            fn $fn(self, rhs: Vector<S>) -> Self::Output {
                self.$fn(&rhs)
            }
        }
    };
}

impl_elementwise!(Add, add);
impl_elementwise!(Sub, sub);
impl_elementwise!(Mul, mul);
impl_elementwise!(Div, div);

macro_rules! impl_scalar_ops {
    ($trait:ident, $fn:ident, ($($scalar:ty),+)) => (
        $(
            impl_scalar_ops!($trait, $fn, $scalar);
        )+
    );

    ($trait:ident, $fn:ident, $scalar:ty) => (
        impl $trait<$scalar> for Vector<$scalar> {
            type Output = Vector<$scalar>;

            fn $fn(mut self, rhs: $scalar) -> Self::Output {
                for i in 0..self.len() {
                    self[i] = self[i].$fn(rhs);
                }

                self
            }
        }

        impl $trait<Vector<$scalar>> for $scalar {
            type Output = Vector<$scalar>;

            fn $fn(self, mut rhs: Vector<$scalar>) -> Self::Output {
                for i in 0..rhs.len() {
                    rhs[i] = self.$fn(rhs[i]);
                }

                rhs
            }
        }
    );
}

impl_scalar_ops!(
    Add,
    add,
    (i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64)
);
impl_scalar_ops!(
    Sub,
    sub,
    (i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64)
);
impl_scalar_ops!(
    Mul,
    mul,
    (i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64)
);
impl_scalar_ops!(
    Div,
    div,
    (i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64)
);

impl<S> Neg for Vector<S>
where
    S: Scalar,
{
    type Output = Vector<S>;
    fn neg(mut self) -> Vector<S> {
        self.iter_mut().for_each(|s| *s = -*s);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_small() {
        let a = mat![1, 2, 3];
        let b = mat![1, 2, 2];

        let result = a.dot(&b);

        assert_eq!(result, 1 * 1 + 2 * 2 + 3 * 2);
    }

    #[test]
    fn mul_transpose() {
        let a = mat![1, 2, 3];
        let b = mat![4, 5];

        let result = a.mul_transpose(&b);

        assert_eq!(result, mat![[1 * 4, 1 * 5], [2 * 4, 2 * 5], [3 * 4, 3 * 5]]);
    }

    #[test]
    fn add_small() {
        let a = mat![1, 2, 3];
        let b = mat![1, 2, 2];

        let ref_ref = &a + &b;
        let move_ref = a.clone() + &b;
        let ref_move = &a + b.clone();
        let move_move = a + b;

        assert_eq!(move_ref, ref_ref);
        assert_eq!(move_ref, ref_move);
        assert_eq!(move_ref, move_move);
        assert_eq!(move_ref, mat![1 + 1, 2 + 2, 3 + 2]);
    }

    #[test]
    fn sub_small() {
        let a = mat![1, 2, 3];
        let b = mat![1, 2, 2];

        let ref_ref = &a - &b;
        let move_ref = a.clone() - &b;
        let ref_move = &a - b.clone();
        let move_move = a - b;

        assert_eq!(move_ref, ref_ref);
        assert_eq!(move_ref, ref_move);
        assert_eq!(move_ref, move_move);
        assert_eq!(move_ref, mat![1 - 1, 2 - 2, 3 - 2]);
    }

    #[test]
    fn vec_add_scalar() {
        let a = mat![1, 2, 3];
        let result = a + 1;
        assert_eq!(result, mat![2, 3, 4]);
    }

    #[test]
    fn scalar_add_vec() {
        let a = mat![1, 2, 3];
        let result = 1 + a;
        assert_eq!(result, mat![2, 3, 4]);
    }

    #[test]
    fn vec_mul_scalar() {
        let a = mat![1, 2, 3];
        let result = a * 2;
        assert_eq!(result, mat![2, 4, 6]);
    }

    #[test]
    fn scalar_mul_vec() {
        let a = mat![1, 2, 3];
        let result = 2 * a;
        assert_eq!(result, mat![2, 4, 6]);
    }
}
