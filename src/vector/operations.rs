use super::*;

pub fn dot<S>(lhs: &[S], rhs: &[S]) -> S
where
    S: Scalar,
{
    assert!(
        lhs.len() == rhs.len(),
        "Vectors must be of same length. Left hand side has length {} and right hand side has length {}",
        lhs.len(),
        rhs.len()
    );

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
}
