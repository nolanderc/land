use super::*;

impl<S> Index<usize> for Matrix<S>
where
    S: Scalar,
{
    type Output = [S];

    /// Get a row of the matrix.
    fn index(&self, row: usize) -> &Self::Output {
        self.row(row)
    }
}

impl<S> IndexMut<usize> for Matrix<S>
where
    S: Scalar,
{
    /// Get a row of the matrix mutably
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.row_mut(index)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_rect() {
        let a: Matrix<i32> = mat![
            [1, 2, 3],
            [4, 5, 6]
        ];

        assert_eq!(a[0][0], 1);
        assert_eq!(a[0][1], 2);
        assert_eq!(a[0][2], 3);
        assert_eq!(a[1][0], 4);
        assert_eq!(a[1][1], 5);
        assert_eq!(a[1][2], 6);
    }
}

