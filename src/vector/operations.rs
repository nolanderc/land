use super::*;

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
}
