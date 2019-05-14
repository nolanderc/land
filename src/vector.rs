pub(crate) mod operations;

use crate::traits::{FloatScalar, Scalar};
use std::{fmt, ops::*};

#[derive(Debug, Clone)]
pub struct Vector<S> {
    elements: Vec<S>,
}

impl<S> Vector<S> {
    pub fn new(elements: Vec<S>) -> Vector<S> {
        Vector { elements }
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &S> {
        self.elements.iter()
    }

    pub fn into_iter(self) -> impl Iterator<Item = S> {
        self.elements.into_iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut S> {
        self.elements.iter_mut()
    }

    pub fn to_vec(self) -> Vec<S> {
        self.elements
    }
}

impl<S> Vector<S>
where
    S: Clone,
{
    /// A vector filled with a value.
    pub fn filled(value: S, len: usize) -> Vector<S> {
        Vector {
            elements: vec![value; len],
        }
    }
}

impl<S> Vector<S>
where
    S: Copy,
{
    pub fn map<F>(mut self, mut f: F) -> Vector<S>
    where
        F: FnMut(S) -> S,
    {
        for elem in self.elements.iter_mut() {
            *elem = f(*elem);
        }

        self
    }
}

impl<S> Vector<S>
where
    S: Scalar,
{
    /// A vector filled with zeros.
    pub fn zeros(len: usize) -> Vector<S> {
        Self::filled(S::zero(), len)
    }

    /// A vector filled with ones.
    pub fn ones(len: usize) -> Vector<S> {
        Self::filled(S::one(), len)
    }
}

macro_rules! impl_elementwise_operation {
    ($name:ident ($($arg:ident: $type:ty),*)) => (
        pub fn $name(self $(, $arg: $type)*) -> Vector<F> {
            self.map(|e| e.$name($($arg),*))
        }
    )
}

impl<F> Vector<F>
where
    F: FloatScalar,
{
    /// Elementwise absolute value
    impl_elementwise_operation!(abs());

    /// Elementwise signum:
    /// - `1.0` if the number is positive, `+0.0` or `F::infinity()`.
    /// - `-1.0` if the number is negative, `-0.0` or `F::neg_infinity()`.
    /// - `F::nan()` if the number is `F::nan()`.
    impl_elementwise_operation!(signum());

    /// Elementwise maximum of two values
    impl_elementwise_operation!(max(other: F));

    /// Elementwise minimum of two values
    impl_elementwise_operation!(min(other: F));

    /// Elementwise natural logarithm
    impl_elementwise_operation!(ln());

    /// Elementwise base of arbitrary base
    impl_elementwise_operation!(log(base: F));

    /// Elementwise base 2 logarithm
    impl_elementwise_operation!(log2());

    /// Elementwise base 10 logarithm
    impl_elementwise_operation!(log10());

    /// Elementwise reciprocal: `1/(self)`
    impl_elementwise_operation!(recip());

    /// Raise to the power of an integer power elementwise
    impl_elementwise_operation!(powi(n: i32));

    /// Raise to the power of an integer power elementwise
    impl_elementwise_operation!(powf(n: F));

    /// Elementwise square root
    impl_elementwise_operation!(sqrt());

    /// Elementwise exponential function, `e^(self)`.
    impl_elementwise_operation!(exp());

    /// Elementwise `2^(self)`.
    impl_elementwise_operation!(exp2());

    /// Elementwise sine in radians
    impl_elementwise_operation!(sin());

    /// Elementwise cosine in radians
    impl_elementwise_operation!(cos());

    /// Elementwise tangent in radians
    impl_elementwise_operation!(tan());

    /// Elementwise arcsine in radians
    impl_elementwise_operation!(asin());

    /// Elementwise arccosine in radians
    impl_elementwise_operation!(acos());

    /// Elementwise arctangent in radians
    impl_elementwise_operation!(atan());

    /// Elementwise hyperbolic sine in radians
    impl_elementwise_operation!(sinh());

    /// Elementwise hyperbolic cosine in radians
    impl_elementwise_operation!(cosh());

    /// Elementwise hyperbolic tangent in radians
    impl_elementwise_operation!(tanh());

    /// Elementwise inverse hyperbolic sine in radians
    impl_elementwise_operation!(asinh());

    /// Elementwise inverse hyperbolic cosine in radians
    impl_elementwise_operation!(acosh());

    /// Elementwise inverse hyperbolic tangent in radians
    impl_elementwise_operation!(atanh());
}

impl<S> PartialEq<Self> for Vector<S>
where
    S: Scalar + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.elements == other.elements
    }
}

impl<S> From<Vec<S>> for Vector<S> {
    fn from(elements: Vec<S>) -> Self {
        Vector { elements }
    }
}

impl<S> Into<Vec<S>> for Vector<S> {
    fn into(self) -> Vec<S> {
        self.elements
    }
}

impl<S> Deref for Vector<S> {
    type Target = Vec<S>;

    fn deref(&self) -> &Self::Target {
        &self.elements
    }
}

impl<S> AsRef<[S]> for Vector<S> {
    fn as_ref(&self) -> &[S] {
        &self.elements
    }
}

impl<S> Index<usize> for Vector<S> {
    type Output = S;

    fn index(&self, index: usize) -> &S {
        self.elements.index(index)
    }
}

impl<S> IndexMut<usize> for Vector<S> {
    fn index_mut(&mut self, index: usize) -> &mut S {
        self.elements.index_mut(index)
    }
}

impl<S> fmt::Display for Vector<S>
where
    S: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..self.len() {
            self[i].fmt(f)?;

            if i != self.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_filled_vector() {
        assert_eq!(Vector::<i32>::filled(2, 3), mat![2, 2, 2])
    }

    #[test]
    fn create_zero_vector() {
        assert_eq!(Vector::<i32>::zeros(3), mat![0, 0, 0])
    }

    #[test]
    fn create_one_vector() {
        assert_eq!(Vector::<i32>::ones(3), mat![1, 1, 1])
    }

    #[test]
    fn map_add_one() {
        let a = mat![1, 2, 3];

        let result = a.map(|e| e + 1);

        assert_eq!(result, mat![2, 3, 4])
    }

    #[test]
    fn display_vector() {
        let mat = mat![1.234, 1.0/3.0, 7.0001];

        let out = format!("{:.2}", mat);

        assert_eq!(
            out,
            "[1.23, 0.33, 7.00]"
        )
    }
}
