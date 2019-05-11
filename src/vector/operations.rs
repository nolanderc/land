use super::*;

macro_rules! assert_equal_length {
    ($lhs:expr, $rhs:expr) => {
        let lhs = $lhs;
        let rhs = $rhs;
        assert!(
            lhs.len() == rhs.len(),
            "Vectors must be of same length. Left hand side has length {} and right hand side has length {}",
            lhs.len(),
            rhs.len()
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

impl<S> Add<Self> for &Vector<S>
where
    S: Scalar,
{
    type Output = Vector<S>;

    fn add(self, other: Self) -> Self::Output {
        assert_equal_length!(self, other);

        let mut out = vec![S::zero(); self.len()];

        for i in 0..self.len() {
            out[i] = self[i] + other[i];
        }

        out.into()
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
    fn add_small() {
        let a = mat![1, 2, 3];
        let b = mat![1, 2, 2];

        let result = &a + &b;

        assert_eq!(result, mat![1+1, 2+2, 3+2]);
    }
}
