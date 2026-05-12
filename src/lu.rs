use crate::matrix::Matrix;


// ─────────────────────────────────────────────────────────────
// ─── LU-разложение с частичным выбором ведущего элемента ─────
// ─────────────────────────────────────────────────────────────
// PA = LU: P - перестановка, L - нижний унитреугольный, U - верхний треугольный
//
// Хранение: L и U пакуем в одну матрицу!
//    -  U - верхний треугольник, включая диагональ)
//    -  L - нижний треугольник
//а единичная диагональ L нигде не хранится
//
//
//
// Зачем пивотинг: без него деление на околонулевой ведущий элемент катастрофически усиливает ошибки
// поэтьои я меняю строки так, чтобы на диагонали стоял максимальный |pivot|.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LuError {
    Singular,
}


//           Результат LU-разложения 
// 
// lu            - упакованные L\U
// piv           - вектор перестановок: piv[k] = строка, на которую менялась k
// sign_negative - нечётность перестановки, нужна для знака детерминанта

#[derive(Debug, Clone)]
pub struct LU<const N: usize> {
    pub(crate) lu: Matrix<N, N>,
    pub(crate) piv: [usize; N],
    pub(crate) sign_negative: bool,
}

////// EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
impl<const N: usize> LU<N> {

    //факторизует матрицу a; Err(Singular) если матрица вырождена.
    pub fn factor(a: &Matrix<N, N>) -> Result<Self, LuError> {
        let mut lu = a.clone();
        let mut piv = [0usize; N];

        let max_abs = a.data.iter().flatten().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let tol = (f64::EPSILON * max_abs * N as f64).max(f64::MIN_POSITIVE);
        let mut sign_negative = false;

        for k in 0..N {
            let mut p = k;
            let mut max_col = lu.data[k][k].abs();
            for i in k + 1..N {
                let v = lu.data[i][k].abs();
                if v > max_col { max_col = v; p = i; }
            }

            if max_col < tol { return Err(LuError::Singular); }

            piv[k] = p;
            if p != k {
                lu.swap_rows(k, p);
                sign_negative = !sign_negative;
            }

            let pivot_inv = 1.0 / lu.data[k][k];
            let row_k = lu.data[k];     // копируем строку k для Schur-обновления
            for i in k + 1..N {
                lu.data[i][k] *= pivot_inv;
                let factor = lu.data[i][k];
                for j in k + 1..N {
                    lu.data[i][j] -= factor * row_k[j];
                }
            }
        }

        Ok(LU { lu, piv, sign_negative })
    }


    ///решает Ax = b. 
    // Три шага: P*b -> прямая замена (L) -> обратная замена (U)
    pub fn lu_solve(&self, b: &[f64; N]) -> [f64; N] {
        let mut y = *b;
        apply_permutation(&self.piv, &mut y);
        forward_substitute(&self.lu, &mut y);
        back_substitute(&self.lu, &mut y);
        y
    }

    //Вычисляет A^-1 решая AX = I
    pub fn inverse(&self) -> Matrix<N, N> {
        //старт с единичной матрицы, сразу применяем перестановку ко всем N столбцам
        let mut work = Matrix::<N, N>::identity();
        for k in 0..N {
            work.swap_rows(k, self.piv[k]);
        }

        //прямая замена
        for i in 0..N {
            for j in 0..i {
                let factor = self.lu.data[i][j];
                if factor == 0.0 { continue; }
                let wj = work.data[j];
                let wi = &mut work.data[i];
                for col in 0..N {
                    wi[col] -= factor * wj[col];
                }
            }
        }

        // обратная замена
        for i in (0..N).rev() {
            for j in i + 1..N {
                let factor = self.lu.data[i][j];
                if factor == 0.0 { continue; }
                let wj = work.data[j];
                let wi = &mut work.data[i];
                for col in 0..N {
                    wi[col] -= factor * wj[col];
                }
            }
            let diag_inv = 1.0 / self.lu.data[i][i];
            let wi = &mut work.data[i];
            for col in 0..N {
                wi[col] *= diag_inv;
            }
        }

        work
    }

    pub fn det(&self) -> f64 {
        let mut det = 1.0;
        for i in 0..N {
            let d = self.lu.data[i][i];
            if d == 0.0 { return 0.0; }
            det *= d;
        }
        if self.sign_negative { -det } else { det }
    }

    // Распаковывает в отдельные L, U и вектор перестановок.
    pub fn unpack(&self) -> (Matrix<N, N>, Matrix<N, N>, [usize; N]) {
        let mut l = Matrix::identity();
        let mut u = Matrix::zeros();
        for i in 0..N {
            for j in 0..N {
                if i > j { l[(i,j)] = self.lu[(i,j)]; }
                else      { u[(i,j)] = self.lu[(i,j)]; }
            }
        }
        (l, u, self.piv)
    }
}


//  обёртки вокруг LU::factor для Matrix<N, N> 
// (когда нужна одна операция и нет смысла держать LU-объект)

impl<const N: usize> Matrix<N, N> {
    //Обратная матрица. Err если вырождена
    pub fn inverse(&self) -> Result<Matrix<N, N>, LuError> {
        Ok(LU::factor(self)?.inverse())
    }

    //Определитель; 0.0 если вырождена
    pub fn det(&self) -> f64 {
        match LU::factor(self) {
            Ok(lu) => lu.det(),
            Err(_) => 0.0,
        }
    }

    // Решает Ax = b, Err если вырождена
    pub fn lu_solve(&self, b: &[f64; N]) -> Result<[f64; N], LuError> {
        Ok(LU::factor(self)?.lu_solve(b))
    }
}

/// _____________________________________________
/// ─── Вспомогательные функции подстановки ─────
/// ---------------------------------------------

// применяем перестановку b <- Pb: для каждого i меняем b[i] с b[piv[i]]
fn apply_permutation<const N: usize>(piv: &[usize; N], b: &mut [f64; N]) {
    for i in 0..N {
        b.swap(i, piv[i]);
    }
}

// прямая замена
// y[i] = b[i] - sum_{j<i} L[i][j]*y[j]
fn forward_substitute<const N: usize>(lu: &Matrix<N, N>, b: &mut [f64; N]) {
    for i in 0..N {
        let mut acc = b[i];
        for j in 0..i {
            acc -= lu.data[i][j] * b[j];
        }
        b[i] = acc;
    }
}

// обратная замена: Ux = y
// x[i] = (y[i] - sum_{j>i} U[i][j]*x[j]) / U[i][i]
fn back_substitute<const N: usize>(lu: &Matrix<N, N>, b: &mut [f64; N]) {
    for i in (0..N).rev() {
        let mut acc = b[i];
        let row = &lu.data[i];
        for j in i + 1..N {
            acc -= row[j] * b[j];
        }
        b[i] = acc / lu.data[i][i];
    }
}