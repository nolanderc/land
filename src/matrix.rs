mod constructors;
mod dimensions;
mod index;
mod operations;

pub use self::dimensions::*;
use crate::traits::Scalar;
use std::{fmt, ops::*};

/// A row major matrix
#[derive(Debug, Clone)]
pub struct Matrix<S> {
    elements: Vec<S>,
    dimensions: Dimensions,
}

impl<S> Matrix<S> {
    /// Get the dimensions of a matrix
    pub fn dim(&self) -> Dimensions {
        self.dimensions
    }

    /// Get the dimensions of a matrix
    pub fn size(&self) -> Dimensions {
        self.dimensions
    }

    /// Get a row of the matrix
    pub fn row(&self, row: usize) -> &[S] {
        let row_start = self.dimensions.row_major(row, 0);
        let row_len = self.dimensions.cols;

        &self.elements[row_start..row_start + row_len]
    }

    /// Get a row of the matrix mutably
    pub fn row_mut(&mut self, row: usize) -> &mut [S] {
        let row_start = self.dimensions.row_major(row, 0);
        let row_len = self.dimensions.cols;

        &mut self.elements[row_start..row_start + row_len]
    }
}

impl<S> PartialEq<Self> for Matrix<S>
where
    S: Scalar + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.dimensions == other.dimensions && self.elements == other.elements
    }
}

impl<S> fmt::Display for Matrix<S>
where
    S: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[")?;
        for row in 0..self.dimensions.rows {
            write!(f, "    [")?;
            for col in 0..self.dimensions.cols {
                write!(f, "{}", self.row(row)[col])?;

                if col != self.dimensions.cols - 1 {
                    write!(f, ", ")?;
                }
            }
            if row == self.dimensions.rows - 1 {
                writeln!(f, "]")?;
            } else {
                writeln!(f, "],")?;
            }
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_filled_matrix() {
        assert_eq!(
            Matrix::<i32>::filled(2, (3, 4).into()),
            mat![[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
        )
    }

    #[test]
    fn create_zero_matrix() {
        assert_eq!(
            Matrix::<i32>::zeros((3, 4).into()),
            mat![[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
    }

    #[test]
    fn create_one_matrix() {
        assert_eq!(
            Matrix::<i32>::ones((3, 4).into()),
            mat![[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        )
    }

    #[test]
    fn create_diagonal_matrix() {
        assert_eq!(
            Matrix::<i32>::diagonal(3, 4),
            mat![[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]]
        )
    }

    #[test]
    fn create_identity_matrix() {
        assert_eq!(
            Matrix::<i32>::identity(4),
            mat![[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
    }

    #[test]
    fn display_matrix() {
        let mat = mat![[1, 2, 3], [4, 5, 6]];

        let out = format!("{}", mat);

        assert_eq!(
            out,
            "[
    [1, 2, 3],
    [4, 5, 6]
]"
        )
    }
}
