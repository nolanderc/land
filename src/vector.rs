pub(crate) mod operations;

use crate::traits::Scalar;
use std::ops::*;

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

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut S> {
        self.elements.iter_mut()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_filled_vector() {
        assert_eq!(
            Vector::<i32>::filled(2, 3),
            Vector {
                elements: vec![2; 3]
            }
        )
    }

    #[test]
    fn create_zero_vector() {
        assert_eq!(
            Vector::<i32>::zeros(3),
            Vector {
                elements: vec![0; 3]
            }
        )
    }

    #[test]
    fn create_one_vector() {
        assert_eq!(
            Vector::<i32>::ones(3),
            Vector {
                elements: vec![1; 3]
            }
        )
    }
}
