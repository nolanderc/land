use num::Num;
use std::fmt::Debug;

/// A scalar value
pub trait Scalar: Num + Debug + Copy + Clone + Sized {}

impl<S> Scalar for S where S: Num + Debug + Copy + Clone + Sized {}

