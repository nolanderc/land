use num::{Num, Float, traits::NumAssign};
use std::fmt::Debug;

/// A scalar value
pub trait Scalar: Num + NumAssign + Debug + Copy + Clone + Sized {}

impl<S> Scalar for S where S: Num + NumAssign + Debug + Copy + Clone + Sized {}


/// A floating point scalar value
pub trait FloatScalar: Scalar + Float {}

impl<S> FloatScalar for S where S: Scalar + Float {}

