use std::fmt;

/// The dimensions of a matrix
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Dimensions {
    pub rows: usize,
    pub cols: usize,
}

impl Dimensions {
    pub fn square(size: usize) -> Dimensions {
        Dimensions {
            rows: size,
            cols: size,
        }
    }

    pub fn elements(&self) -> usize {
        self.rows * self.cols
    }

    pub fn transpose(&self) -> Dimensions {
        Dimensions {
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Convert a coordinate to an index into a row-major matrix.
    pub(crate) fn row_major(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }
}

impl From<(usize, usize)> for Dimensions {
    fn from((rows, cols): (usize, usize)) -> Dimensions {
        Dimensions { rows, cols }
    }
}

impl From<[usize; 2]> for Dimensions {
    fn from([rows, cols]: [usize; 2]) -> Dimensions {
        Dimensions { rows, cols }
    }
}

impl fmt::Display for Dimensions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}x{}", self.rows, self.cols)
    }
}
