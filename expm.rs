// основано на методе из статьи The Scaling and Squaring Method for the Matrix Exponential Revisited
// http://eprints.ma.man.ac.uk/1300/1/covered/MIMS_ep2009_9.pdf


use crate::matrix::Matrix;

const PADE_COEFS: [f64; 14] = [
    1.0,
    0.5,
    0.12,
    1.833333333333333e-2,   // 11/600
    1.992753623188406e-3,   // 11/5520
    1.630434782608696e-4,   // 3/18400
    1.035196687370600e-5,
    5.175983438685100e-7,
    2.043151683173277e-8,
    6.306659613335512e-10,
    1.483525522051185e-11,
    2.529153491597966e-13,
    2.810170546428544e-15,
    1.544049750670308e-17, 
];

const THETA_13: f64 = 5.371920351148152; 

// only god and Higham know how this works
impl<const N: usize> Matrix<N, N> {
    pub fn expm(&self) -> Matrix<N, N> {
        let norm = self.max_column_sum();

        let s = if norm > THETA_13 {
            (norm / THETA_13).log2().ceil() as u32
        } else { 0 };

        let mut a = self.clone();
        if s > 0 {
            a *= 1.0 / (1u64 << s) as f64;
        }

        let a2 = &a * &a;
        let a4 = &a2 * &a2;
        let a6 = &a2 * &a4;
        let id = Matrix::<N, N>::identity();

        // w1 = c13*a6 + c11*a4 + c9*a2 + c7*I
        let mut w1 = Matrix::<N, N>::from_scaled(PADE_COEFS[13], &a6);
        w1.axpy(PADE_COEFS[11], &a4);
        w1.axpy(PADE_COEFS[ 9], &a2);
        w1.axpy(PADE_COEFS[ 7], &id);

        // w2 = c12*a6 + c10*a4 + c8*a2 + c6*I
        let mut w2 = Matrix::<N, N>::from_scaled(PADE_COEFS[12], &a6);
        w2.axpy(PADE_COEFS[10], &a4);
        w2.axpy(PADE_COEFS[ 8], &a2);
        w2.axpy(PADE_COEFS[ 6], &id);

        // u_inner = a6*w1 + c5*a4 + c3*a2 + c1*I
        let mut u_inner = &a6 * &w1;
        u_inner.axpy(PADE_COEFS[5], &a4);
        u_inner.axpy(PADE_COEFS[3], &a2);
        u_inner.axpy(PADE_COEFS[1], &id);
        let u = &a * &u_inner;

        // v = a6*w2 + c4*a4 + c2*a2 + c0*I  (matmul then 3 axpy)
        let mut v = &a6 * &w2;
        v.axpy(PADE_COEFS[4], &a4);
        v.axpy(PADE_COEFS[2], &a2);
        v.axpy(PADE_COEFS[0], &id);

        let p = &v + &u;
        let q = v - u;

        let mut r = &q.inverse().expect("singular matrix in expm") * &p;

        for _ in 0..s {
            r = &r * &r;
        }

        r
    }
}


#[test]
fn expm_inverse() {
    let a = Matrix::from_rows([
        [1.0, 2.0],
        [3.0, 4.0],
    ]);
    let neg_a = a.clone() * -1.0;
    let prod = a.expm() * neg_a.expm();
    println!("🟢 TEST 1 🟢");

    println!("{} \n{}", a, neg_a);
    println!("{} \n{}", a.expm(), neg_a.expm());
    println!("{}", prod);


    assert!(prod.approx_eq(&Matrix::identity(), 1e-10));
}