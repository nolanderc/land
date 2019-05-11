use super::*;

impl<S> Matrix<S> {
    /// Create a new matrix from a row-major format
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
