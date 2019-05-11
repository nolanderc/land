use num::{traits::NumAssign, Float, Num};
use std::{ops::Neg, fmt::Debug};

/// A scalar value
pub trait Scalar: Num + Neg<Output = Self> + NumAssign + Debug + Copy + Clone + Sized {}

impl<S> Scalar for S where S: Num + Neg<Output = Self> + NumAssign + Debug + Copy + Clone + Sized {}

/// A floating point scalar value
pub trait FloatScalar: Scalar + Float {}

impl<S> FloatScalar for S where S: Scalar + Float {}
