#[macro_export]
macro_rules! mat {
    [$([$($elem:expr),+]),+] => {
        $crate::Matrix::new(vec![
            $(
                vec![$($elem),+].into()
            ),+
        ])
    };

    [$($elem:expr),+] => {
        $crate::Vector::new(vec![ $( $elem ),+ ])
    }
}
