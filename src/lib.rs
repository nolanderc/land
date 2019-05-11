#[macro_use]
mod macros;
mod matrix;
mod traits;
mod vector;

pub use matrix::*;
pub use traits::*;
pub use vector::*;

mod prelude {
    pub use crate::matrix::*;
    pub use crate::traits::*;
    pub use crate::vector::*;
}
