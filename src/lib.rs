#[macro_use]
mod macros;

mod matrix;
mod vector;

mod ops;
mod traits;

pub use matrix::Matrix;
pub use vector::Vector;
pub use traits::Scalar;

