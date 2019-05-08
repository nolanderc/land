use std::ops::*;

use crate::traits::Scalar;

#[derive(Debug, Clone)]
pub struct Vector<S> {
    pub(crate) data: Vec<S>,
}

impl<S> Vector<S> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> impl Iterator<Item=&S> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut S> {
        self.data.iter_mut()
    }
}

impl<S> Vector<S>
where
    S: Clone,
{
    /// A vector filled with a value.
    pub fn filled(value: S, len: usize) -> Vector<S> {
        Vector {
            data: vec![value; len],
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
        self.data == other.data
    }
}

impl<S, V> From<V> for Vector<S> where V: Into<Vec<S>> {
    fn from(data: V) -> Self {
        Vector { data: data.into() }
    }
}

impl<S> AsRef<[S]> for Vector<S> {
    fn as_ref(&self) -> &[S] {
        &self.data
    }
}


impl<S> Index<usize> for Vector<S> {
    type Output = S;

    fn index(&self, index: usize) -> &S {
        self.data.index(index)
    }
}

impl<S> IndexMut<usize> for Vector<S> {
    fn index_mut(&mut self, index: usize) -> &mut S {
        self.data.index_mut(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_filled_vector() {
        assert_eq!(Vector::<i32>::filled(2, 3), Vector { data: vec![2; 3] })
    }

    #[test]
    fn create_zero_vector() {
        assert_eq!(Vector::<i32>::zeros(3), Vector { data: vec![0; 3] })
    }

    #[test]
    fn create_one_vector() {
        assert_eq!(Vector::<i32>::ones(3), Vector { data: vec![1; 3] })
    }
}
