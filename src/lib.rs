#![cfg_attr(feature = "unstable", feature(test))]

mod matrix;
mod ops;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

