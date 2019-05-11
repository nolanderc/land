#[macro_export]
macro_rules! mat {
    [$([$($elem:expr),+]),+] => {
        Matrix::new(vec![
            $(
                vec![$($elem),+].into()
            ),+
        ])
    };

    [$($elem:expr),+] => {
        Vector::new(vec![ $( $elem ),+ ])
    }
}
