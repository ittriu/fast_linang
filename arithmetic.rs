use std::ops::{Add, Sub, Neg, Mul, AddAssign, SubAssign, MulAssign, DivAssign};
use crate::matrix::Matrix;



// ─────────────────────────────────────────────────
// ─── Сложение и вычитание матриц с друг другом ───
// ─────────────────────────────────────────────────
// Матрица + матрица
// к сожалению нужно 4 impl чтобы поддерживать
// я пытался тут что-то придумать с макросами но ничего дельного не придумал
// все четыре вариации владения на вход (T+T, &T+T, T+&T, &T+&T)

impl<const N: usize, const M: usize> Add for &Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn add(self, matr2: &Matrix<N, M>) -> Matrix<N, M> {
        let mut data = self.data;
        for i in 0..N {
            for j in 0..M {
                data[i][j] += matr2[(i, j)];
            }
        }
        Matrix { data }
    }
}

// эти impl делегируют на основной
impl<const N: usize, const M: usize> Add for Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn add(self, matr2: Matrix<N, M>) -> Matrix<N, M> { &self + &matr2 }
}

impl<const N: usize, const M: usize> Add<&Matrix<N, M>> for Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn add(self, matr2: &Matrix<N, M>) -> Matrix<N, M> { &self + matr2 }
}

impl<const N: usize, const M: usize> Add<Matrix<N, M>> for &Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn add(self, matr2: Matrix<N, M>) -> Matrix<N, M> { self + &matr2 }
}



// ------------------------------------
// Матрица - матрица 
// тут все аналогично

impl<const N: usize, const M: usize> Sub for &Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn sub(self, matr2: &Matrix<N, M>) -> Matrix<N, M> {
        
        let mut data = self.data;
        
        for i in 0..N {
            for j in 0..M {
                data[i][j] -= matr2[(i, j)];
            }
        }

        Matrix { data }
    }
}

// эти impl делегируют на основной
impl<const N: usize, const M: usize> Sub for Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn sub(self, matr2: Matrix<N, M>) -> Matrix<N, M> { &self - &matr2 }
}

impl<const N: usize, const M: usize> Sub<&Matrix<N, M>> for Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn sub(self, matr2: &Matrix<N, M>) -> Matrix<N, M> { &self - matr2 }
}

impl<const N: usize, const M: usize> Sub<Matrix<N, M>> for &Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn sub(self, matr2: Matrix<N, M>) -> Matrix<N, M> { self - &matr2 }
}





// |───────────────────────────────────────────────────────────|
// |────────────────────── Минус матрица ──────────────────────|
// |───────────────────────────────────────────────────────────|


impl<const N: usize, const M: usize> Neg for &Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn neg(self) -> Matrix<N, M> {
        let mut data = self.data;
        for i in 0..N {
            for j in 0..M {
                data[i][j] = -data[i][j];
            }
        }
        Matrix { data }
    }
}

impl<const N: usize, const M: usize> Neg for Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn neg(self) -> Matrix<N, M> {
        let mut data = self.data;
        for i in 0..N {
            for j in 0..M {
                data[i][j] = -data[i][j];
            }
        }
        Matrix { data }
    }
}

// скалярное умножение

// матрица * скаляр
impl<const N: usize, const M: usize> Mul<f64> for &Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn mul(self, scalar: f64) -> Matrix<N, M> {
        let mut data = self.data;
        for i in 0..N {
            for j in 0..M {
                data[i][j] *= scalar;
            }
        }
        Matrix { data }
    }
}

// матрица * скаляр - делегирует
impl<const N: usize, const M: usize> Mul<f64> for Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn mul(self, scalar: f64) -> Matrix<N, M> { &self * scalar }
}

// скаляр * &матрица
impl<const N: usize, const M: usize> Mul<&Matrix<N, M>> for f64 {
    type Output = Matrix<N, M>;
    fn mul(self, mat: &Matrix<N, M>) -> Matrix<N, M> { mat * self }
}

// скаляр * матрица
impl<const N: usize, const M: usize> Mul<Matrix<N, M>> for f64 {
    type Output = Matrix<N, M>;
    fn mul(self, mat: Matrix<N, M>) -> Matrix<N, M> { &mat * self }
}

// &матрица / скаляр
impl<const N: usize, const M: usize> std::ops::Div<f64> for &Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn div(self, scalar: f64) -> Matrix<N, M> {
        self * (1.0 / scalar)  
    }
}

// impl<const N: usize, const M: usize> std::ops::Div<f64> for &Matrix<N, M> {
//     type Output = Matrix<N, M>;
//     fn div(self, scalar: f64) -> Matrix<N, M> {
//         let mut data = self.data;
//         for i in 0..N {
//             for j in 0..M {
//                 data[i][j] /= scalar;
//             }
//         }
//         Matrix { data }
//     }
// }

// матрица / скаляр - делегирует
impl<const N: usize, const M: usize> std::ops::Div<f64> for Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn div(self, scalar: f64) -> Matrix<N, M> { &self / scalar }
}



// ─── УМНОЖЕНИЕ МАТРИЦ ───
// Matrix<N, K> * Matrix<K, M> -> Matrix<N, M> 
// умножаем поэлементно
// ─── УМНОЖЕНИЕ МАТРИЦ ───




// TILE TILE TIILE
// https://johnnysswlab.com/memory-access-pattern-and-performance-the-example-of-matrix-multiplication/
// 
// умножение матриц с тайлингом
#[inline(always)]
fn mm<const N: usize, const K: usize, const M: usize>(
    a: &[[f64; K]; N],
    b: &[[f64; M]; K],
    c: &mut [[f64; M]; N],
) {
    const T: usize = 24;
    let mut ii = 0;
    while ii < N {
        let ie = (ii + T).min(N);
        let mut kk = 0;
        while kk < K {
            let ke = (kk + T).min(K);
            let mut jj = 0;
            while jj < M {
                let je = (jj + T).min(M);
                for i in ii..ie {
                    for k in kk..ke {
                        let a_ik = a[i][k];
                        let bk = &b[k];
                        let ci = &mut c[i];
                        for j in jj..je { ci[j] += a_ik * bk[j]; }
                    }
                }
                jj += T;
            }
            kk += T;
        }
        ii += T;
    }
}

impl<const N: usize, const K: usize, const M: usize> Mul<&Matrix<K, M>> for &Matrix<N, K> {
    type Output = Matrix<N, M>;
    #[inline(always)]
    fn mul(self, b: &Matrix<K, M>) -> Matrix<N, M> {
        let mut c = Matrix::<N, M>::zeros();
        mm(&self.data, &b.data, &mut c.data);
        c
    }
}






// а это старое поэлементное умножение
// impl<const N: usize, const K: usize, const M: usize> Mul<&Matrix<K, M>> for &Matrix<N, K> {
//     type Output = Matrix<N, M>;
//     fn mul(self, matr2: &Matrix<K, M>) -> Matrix<N, M> {
        
//         let mut product_matr = Matrix::<N, M>::zeros();

//         for i in 0..N {
//             for k in 0..K {
//                 let a_ik = self.data[i][k];           
//                 let row_k = &matr2.data[k];                 // row k of matr2
//                 let row_i = &mut product_matr.data[i];  // row i of result
//                 for j in 0..M {
//                     row_i[j] += a_ik * row_k[j];
//                 }
//             }
// }

//         product_matr
//     }
// }





// тоже делегируют
impl<const N: usize, const K: usize, const M: usize> Mul<Matrix<K, M>> for Matrix<N, K> {
    type Output = Matrix<N, M>;
    fn mul(self, matr2: Matrix<K, M>) -> Matrix<N, M> { &self * &matr2 }
}

impl<const N: usize, const K: usize, const M: usize> Mul<&Matrix<K, M>> for Matrix<N, K> {
    type Output = Matrix<N, M>;
    fn mul(self, matr2: &Matrix<K, M>) -> Matrix<N, M> { &self * matr2 }
}

impl<const N: usize, const K: usize, const M: usize> Mul<Matrix<K, M>> for &Matrix<N, K> {
    type Output = Matrix<N, M>;
    fn mul(self, matr2: Matrix<K, M>) -> Matrix<N, M> { self * &matr2 }
}






//  *= += -= /= [assign operators]


impl<const N: usize, const M: usize> AddAssign<&Matrix<N, M>> for Matrix<N, M> {
    fn add_assign(&mut self, matr2: &Matrix<N, M>) {
        for i in 0..N {
            for j in 0..M {
                self.data[i][j] += matr2.data[i][j];
            }
        }
    }
}

impl<const N: usize, const M: usize> SubAssign<&Matrix<N, M>> for Matrix<N, M> {
    fn sub_assign(&mut self, matr2: &Matrix<N, M>) {
        for i in 0..N {
            for j in 0..M {
                self.data[i][j] -= matr2.data[i][j];
            }
        }
    }
}

// я решил тут сделать только *= на скаляр потому что 
impl<const N: usize, const M: usize> MulAssign<f64> for Matrix<N, M> {
    fn mul_assign(&mut self, scalar: f64) {
        for i in 0..N {
            for j in 0..M {
                self.data[i][j] *= scalar;
            }
        }
    }
}

impl<const N: usize, const M: usize> DivAssign<f64> for Matrix<N, M> {
    fn div_assign(&mut self, scalar: f64) {
        let inv = 1.0 / scalar;  // 1 division
        for i in 0..N {
            for j in 0..M {
                self.data[i][j] *= inv;
            }
        }
    }
}


impl<const N: usize, const M: usize> Matrix<N, M> {

    /// сумма всех элементов по диагонали начиная с [0][0]; [1][1]; [2][2]...
    pub fn trace(&self) -> f64 {
        let mut trace_sum:f64 = 0.0;
        for i in 0..N.min(M){
            trace_sum += self.data[i][i];
        }
        trace_sum
    }

    /// Норма ФРОБЕНИУСА - квадратный корень из суммы квадратов всех её элементов
    pub fn frobenius_norm(&self) -> f64 {
        let mut frobenius_norm: f64 = 0.0;
        for i in 0..N {
            for j in 0..M {
                let x = self.data[i][j];
                frobenius_norm += x * x; 
            }
        }
        frobenius_norm.sqrt()
    }

    /// возвращает максимальную сумму ряда в матрице |a[i][0]| + |a[i][1]| + ... + |a[i][M-1]|
    pub fn max_row_sum(&self) -> f64 {
        let mut max_row_sum: f64 = 0.0;
        for i in 0..N {
            let row = &self.data[i];
            let mut row_sum: f64 = 0.0;
            for j in 0..M {
                row_sum += row[j].abs();   
            }
            if row_sum > max_row_sum {
                max_row_sum = row_sum;
            }
        }
        max_row_sum
    }

    /// возвращает максимальную сумму колонки в матрице |a[0][j]| + |a[1][j]| + ... + |a[N-1][j]|
    pub fn max_column_sum(&self) -> f64 {
        let mut col_sums = [0.0f64; M];
        for i in 0..N {
            let row = &self.data[i];        
            for j in 0..M {
                col_sums[j] += row[j].abs();
            }
        }
        col_sums.iter().cloned().fold(0.0_f64, f64::max)
    }
}

