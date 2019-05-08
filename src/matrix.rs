use std::ops::*;

use crate::{ops::*, traits::Scalar, vector::Vector};

#[derive(Debug, Clone)]
pub struct Matrix<S> {
    pub(crate) rows: Vec<Vector<S>>,
}

impl<S> Matrix<S> {
    pub fn new<V>(rows: V) -> Matrix<S>
    where
        V: Into<Vec<Vector<S>>>,
    {
        let rows = rows.into();

        for row in rows.iter() {
            assert!(
                row.len() == rows[0].len(),
                "All rows in matrix need to be of same length"
            );
        }

        Matrix { rows }
    }

    pub fn rows_len(&self) -> usize {
        self.rows.len()
    }

    pub fn cols_len(&self) -> usize {
        self.rows.get(0).map(Vector::len).unwrap_or(0)
    }

    /// Return an iterator which goes through all rows of the matrix.
    pub fn iter_rows(&self) -> impl Iterator<Item = &Vector<S>> {
        self.rows.iter()
    }

    /// Return an iterator which mutably goes through all rows of the matrix.
    pub fn iter_rows_mut(&mut self) -> impl Iterator<Item = &mut Vector<S>> {
        self.rows.iter_mut()
    }

    /// Return an iterator which, row by row, goes through all elements of the matrix.
    pub fn iter(&self) -> impl Iterator<Item = &S> {
        self.rows.iter().flat_map(|row| row.iter())
    }

    /// Return an iterator which, row by row, *mutably* goes through all elements of the matrix.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut S> {
        self.rows.iter_mut().flat_map(|row| row.iter_mut())
    }
}

impl<S> Matrix<S>
where
    S: Clone,
{
    /// A matrix filled with a value.
    pub fn filled(value: S, rows: usize, cols: usize) -> Matrix<S> {
        Matrix {
            rows: vec![Vector::filled(value, cols); rows],
        }
    }
}

impl<S> Matrix<S>
where
    S: Scalar,
{
    /// A square matrix with a value along the diagonal and zeros everywhere else.
    pub fn diagonal(value: S, size: usize) -> Matrix<S> {
        let mut mat = Self::filled(S::zero(), size, size);

        for i in 0..size {
            mat[i][i] = value.clone();
        }

        mat
    }

    /// A matrix filled with zeros
    pub fn zeros(rows: usize, cols: usize) -> Matrix<S> {
        Self::filled(S::zero(), rows, cols)
    }

    /// A matrix filled with ones
    pub fn ones(rows: usize, cols: usize) -> Matrix<S> {
        Self::filled(S::one(), rows, cols)
    }

    /// A square matrix with ones along the diagonal and zeros everywhere else.
    pub fn identity(size: usize) -> Matrix<S> {
        Self::diagonal(S::one(), size)
    }

    /// Get a single row of the matrix.
    pub fn row(&self, index: usize) -> &Vector<S> {
        self.rows.index(index)
    }

    /// Get a single row of the matrix.
    pub fn row_mut(&mut self, index: usize) -> &mut Vector<S> {
        self.rows.index_mut(index)
    }
}

impl<S> Index<usize> for Matrix<S>
where
    S: Scalar,
{
    type Output = Vector<S>;

    fn index(&self, index: usize) -> &Self::Output {
        self.row(index)
    }
}

impl<S> IndexMut<usize> for Matrix<S>
where
    S: Scalar,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.row_mut(index)
    }
}

impl<S> Mul<Self> for &Matrix<S>
where
    S: Scalar,
{
    type Output = Matrix<S>;

    fn mul(self, rhs: Self) -> Matrix<S> {
        mat_mul(self, rhs)
    }
}

impl<S> Mul<&Vector<S>> for &Matrix<S>
where
    S: Scalar,
{
    type Output = Vector<S>;

    fn mul(self, rhs: &Vector<S>) -> Vector<S> {
        mat_vec_mul(self, rhs)
    }
}

impl<S> PartialEq<Self> for Matrix<S>
where
    S: Scalar + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_filled_matrix() {
        assert_eq!(
            Matrix::<i32>::filled(2, 3, 4),
            Matrix {
                rows: vec![vec![2; 4].into(); 3],
            }
        )
    }

    #[test]
    fn create_zero_matrix() {
        assert_eq!(
            Matrix::<i32>::zeros(3, 4),
            Matrix {
                rows: vec![vec![0; 4].into(); 3],
            }
        )
    }

    #[test]
    fn create_one_matrix() {
        assert_eq!(
            Matrix::<i32>::ones(3, 4),
            Matrix {
                rows: vec![vec![1; 4].into(); 3],
            }
        )
    }

    #[test]
    fn create_diagonal_matrix() {
        assert_eq!(
            Matrix::<i32>::diagonal(3, 4),
            Matrix {
                rows: vec![
                    vec![3, 0, 0, 0].into(),
                    vec![0, 3, 0, 0].into(),
                    vec![0, 0, 3, 0].into(),
                    vec![0, 0, 0, 3].into(),
                ],
            }
        )
    }

    #[test]
    fn create_identity_matrix() {
        assert_eq!(
            Matrix::<i32>::identity(4),
            mat![
                1, 0, 0, 0;
                0, 1, 0, 0;
                0, 0, 1, 0;
                0, 0, 0, 1
            ]
        )
    }
}
