
use fast_linalg::Matrix;


pub fn ref_matmul<const N: usize>(a: &Matrix<N, N>, b: &Matrix<N, N>) -> Matrix<N, N> {
    Matrix::from_fn(|i, j| (0..N).map(|k| a[(i, k)] * b[(k, j)]).sum())
}

pub fn ref_identity<const N: usize>() -> Matrix<N, N> {
    Matrix::from_fn(|i, j| if i == j { 1.0 } else { 0.0 })
}


// -------------Error metrics---------------




/// element wise max absolute error
pub fn max_abs_error<const N: usize, const M: usize>(
    a: &Matrix<N, M>,
    b: &Matrix<N, M>,
) -> f64 {
    let mut max_err = 0.0f64;
    for i in 0..N {
        for j in 0..M {
            let err = (a[(i, j)] - b[(i, j)]).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

// трейс
pub fn ref_trace<const N: usize>(m: &Matrix<N, N>) -> f64 {
    (0..N).map(|i| m[(i, i)]).sum()
}

pub fn eigenvalue_trace_sum<const N: usize>(re: &[f64; N]) -> f64 {
    re.iter().sum()
}