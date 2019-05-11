use num::{Num, traits::NumAssign};
use std::fmt::Debug;

/// A scalar value
pub trait Scalar: Num + NumAssign + Debug + Copy + Clone + Sized {}

impl<S> Scalar for S where S: Num + NumAssign + Debug + Copy + Clone + Sized {}
