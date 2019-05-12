use super::*;

impl<S> Matrix<S> {
    /// Create a new matrix from a list of rows
    pub fn new<V>(rows: V) -> Matrix<S>
    where
        V: Into<Vec<Vec<S>>>,
    {
        let rows = rows.into();

        for row in rows.iter() {
            assert!(
                row.len() == rows[0].len(),
                "All rows in matrix need to be of same length"
            );
        }

        let dimensions = Dimensions {
            rows: rows.len(),
            cols: rows.get(0).map(Vec::len).unwrap_or(0),
        };

        let elements = rows.into_iter().map(Vec::into_iter).flatten().collect();

        Matrix {
            elements,
            dimensions,
        }
    }

    /// Create a new matrix from a row-major format.
    /// ```
    /// # use land::{Dimensions, Matrix, mat};
    /// # fn main() {
    /// let elements = vec![1, 2, 3, 4, 5, 6];
    /// let dimensions = Dimensions { rows: 2, cols: 3 };
    /// let matrix = Matrix::from_row_major(dimensions, elements);
    ///
    /// assert_eq!(
    ///     matrix,
    ///     mat![
    ///         [1, 2, 3],
    ///         [4, 5, 6]
    ///     ]
    /// );
    /// # }
    /// ```
    pub fn from_row_major(dimensions: Dimensions, elements: Vec<S>) -> Matrix<S> {
        assert!(
            elements.len() == dimensions.elements(),
            "Number of elements must match matrix dimensions. Number of elements was {} and dimensions {}",
            elements.len(),
            dimensions.elements()
        );

        Matrix {
            elements,
            dimensions,
        }
    }
}

impl<S> Matrix<S>
where
    S: Clone,
{
    /// A matrix filled with a value.
    pub fn filled(value: S, dimensions: Dimensions) -> Matrix<S> {
        Matrix {
            elements: vec![value; dimensions.elements()],
            dimensions: dimensions,
        }
    }
}

impl<S> Matrix<S>
where
    S: Scalar,
{
    /// A matrix filled with zeros
    pub fn zeros(dimensions: Dimensions) -> Matrix<S> {
        Self::filled(S::zero(), dimensions)
    }

    /// A matrix filled with ones
    pub fn ones(dimensions: Dimensions) -> Matrix<S> {
        Self::filled(S::one(), dimensions)
    }

    /// A square matrix with a value along the diagonal and zeros everywhere else.
    pub fn diagonal(value: S, size: usize) -> Matrix<S> {
        let mut mat = Self::filled(S::zero(), Dimensions::square(size));

        for i in 0..size {
            mat[i][i] = value.clone();
        }

        mat
    }

    /// A square matrix with ones along the diagonal and zeros everywhere else.
    pub fn identity(size: usize) -> Matrix<S> {
        Self::diagonal(S::one(), size)
    }
}
