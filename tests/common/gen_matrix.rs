use arbitrary::Unstructured;
use fast_linalg::Matrix;
pub fn uniform_f64(u: &mut Unstructured, lo: f64, hi: f64) -> arbitrary::Result<f64> {
    let bytes: &[u8] = u.bytes(8)?;
    let raw = u64::from_le_bytes(bytes.try_into().unwrap());
    Ok((raw as f64 / u64::MAX as f64) * (hi - lo) + lo)
}

pub fn uniform_matrix<const N: usize, const M: usize>(
    u: &mut Unstructured,
    lo: f64,
    hi: f64,
) -> arbitrary::Result<Matrix<N, M>> {
    let mut m = Matrix::<N, M>::zeros();
    for i in 0..N {
        for j in 0..M {
            m[(i, j)] = uniform_f64(u, lo, hi)?;
        }
    }
    Ok(m)
}


// used for expm semigroup tests
pub fn rate_matrix<const N: usize>(u: &mut Unstructured) -> arbitrary::Result<Matrix<N, N>> {
    let mut m = Matrix::<N, N>::zeros();
    for i in 0..N {
        let mut row_sum = 0.0f64;
        for j in 0..N {
            if i != j {
                let val = uniform_f64(u, 0.0, 10.0)?;
                m[(i, j)] = val;
                row_sum += val;
            }
        }
        m[(i, i)] = -row_sum;
    }
    Ok(m)
}

#[derive(Clone, Copy, Debug)]
pub enum Range { Unit, Hundred, Million }

impl Range {
    pub fn pick(u: &mut Unstructured) -> arbitrary::Result<Self> {
        Ok(match u.arbitrary::<u8>()? % 3 {
            0 => Range::Unit,
            1 => Range::Hundred,
            _ => Range::Million,
        })
    }

    pub fn hi(&self) -> f64 {
        match self { Range::Unit => 1.0, Range::Hundred => 100.0, Range::Million => 1.0e6 }
    }

    ///tolerance scaled
    pub fn tol(&self, n: usize) -> f64 {
        1e-10 * self.hi() * n as f64
    }

    pub fn label(&self) -> &'static str {
        match self { Range::Unit => "[0,1]", Range::Hundred => "[0,100]", Range::Million => "[0,1e6]" }
    }
}