


mod tests{
    use fast_linalg::{LU, Matrix};

    #[test]
    fn lu_1_4x4_reconstruction() {
        let a = Matrix::from_rows([
            [1.0, 2.0, 3.0, 4.0],
            [2.5, 2.6, 3.7, 4.8],
            [3.1, 3.9, 3.3, 3.4],
            [4.1, 4.2, 4.0, 4.4],
        ]);
        println!("🟢 TEST LU.1 🟢 - LU reconstruction: P*A = L*U");

        let lu = LU::factor(&a).expect("factorization failed");
        let (l, u, piv) = lu.unpack();

        // восстановить P*A
        let mut pa = a.clone();
        for k in 0..4 {
            pa.swap_rows(k, piv[k]);
        }

        let lu_product = &l * &u;
        println!("P*A =\n{}", pa);
        println!("L*U =\n{}", lu_product);
        assert!(pa.approx_eq(&lu_product, 1e-10));
    }
}