use std::ops::{Index, IndexMut};
use std::fmt;

/// Я хранию матрицу как массив рядов (row-major order)
/// каждый ряд имеет длину M, всего их N штук
/// 
/// 
///   [ [1, 2, 3],
///     [4, 5, 6] ]      -> N = 2
///         |
///        \ /
///         V
///       M = 3
/// 
/// 

#[derive(Debug, Clone)]
pub struct Matrix<const N: usize, const M: usize> {
    pub(crate) data: [[f64; M]; N],
} 





/// Методы для всех матриц
impl<const N: usize, const M:usize> Matrix<N, M>{
    // занулить всю матрицу
    pub fn zeros() -> Matrix<N, M> {
        Matrix { 
            data: [[0.0; M]; N],
        }
    }


    // Сделать матрицу из списка рядов
    pub fn from_rows(data: [[f64; M]; N]) -> Matrix<N, M> {
        Matrix {data}
    }

    /// Сделать матрицу из функции, которая по foo(i, j) возвращает элемент
    pub fn from_fn(f: impl Fn(usize, usize) -> f64) -> Self {
        let mut data = [[0.0; M]; N];
        for i in 0..N { 
            for j in 0..M {
                data[i][j] = f(i, j);
            }
        }
        Matrix { data }
    }
    
    /// вернуть элемент по координатам
    pub fn get_element(&self, row:usize, col:usize) -> f64 {
        self.data[row][col]
    }

    //  отредактировать 1 элемент 
    // 
    pub fn set_element(&mut self, row: usize, col:usize, value:f64) {
        self.data[row][col] = value;
    }

    //сколько рядов
    pub const fn nrows(&self) -> usize {
        N
    }

    //сколько коллон
    pub const fn ncols(&self) -> usize {
        M
    }

    //matrix size 1
    pub const fn shape(&self) -> [usize; 2] {
        [N, M]
    }

    //return row
    pub fn row(&self, row: usize) -> [f64; M] {
        self.data[row]
    }

    //return coloumn - TODO fix
    pub fn col(&self, col:usize) -> [f64; N] {
        let mut column = [0.0; N];
        for i in 0..N {
            column[i] = self.data[i][col];
        }
        column
    }

    //ТРАНСПОНИРОВАНИЕ МАТРИЦЫ
    pub fn transpose(&self) -> Matrix<M, N> {
        let mut out = Matrix::<M, N>::zeros();
        for i in 0..N {
            for j in 0..M {
                out.data[j][i] = self.data[i][j];  
            }
        }
        out
    }

    pub fn swap_rows(&mut self, row1:usize, row2:usize){
        self.data.swap(row1, row2);
    }

    pub fn swap_cols(&mut self, row1:usize, row2:usize){
        for i in 0..N{
            self.data[i].swap(row1, row2);
        }
    }
    // NEW2027
    /// Fused scaled add: self += scalar * other
    /// Операция AXPY (сокр. от «a times x plus y»)
    /// сумму вектора \(y\) и вектора \(x\), умноженного на скаляр \(\a \).
    #[inline(always)]
    pub fn axpy(&mut self, scalar: f64, other: &Matrix<N, M>) {
        for i in 0..N {
            let row     = &mut self.data[i];
            let row_src = &other.data[i];
            for j in 0..M {
                row[j] += scalar * row_src[j];
            }
        }
    }

    // Returns scalar * other
    //старт apxy-цепочки
    #[inline(always)]
    pub fn from_scaled(scalar: f64, other: &Matrix<N, M>) -> Self {
        let mut data = other.data;
        for i in 0..N {
            for j in 0..M { data[i][j] *= scalar; }
        }
        Matrix { data }
    }
}

// методы для квадратных матриц
impl<const N: usize> Matrix<N, N>{
    pub fn identity() -> Self {
        let mut m = Self::zeros();
        for i in 0..N {
            m.data[i][i] = 1.0;
        }
        m
    }
    // TODO i need to add trace and determinant here


    //makes a diagonal matrix from the given values
    pub fn diagonal(diag: &[f64; N]) -> Self {
        let mut m = Self::zeros();
        for i in 0..N {
            m.data[i][i] = diag[i];
        }
        m
    }
}

// todo check
impl<const N: usize, const M: usize> Index<(usize, usize)> for Matrix<N, M> {
    type Output = f64;

    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i][j]
    }
}

//todo check
impl<const N: usize, const M: usize> IndexMut<(usize, usize)> for Matrix<N, M> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i][j]
    }
}

//display
impl<const N: usize, const M: usize> fmt::Display for Matrix<N, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..N {
            write!(f, "[")?;
            for j in 0..M {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:>10.4}", self.data[i][j])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

/// clone matrix needed for lu
// impl<const N: usize, const M: usize> Clone for Matrix<N, M> {
//     fn clone(&self) -> Self {
//         Matrix { data: self.data }
//     }
// }

impl<const N: usize, const M: usize> Matrix<N, M> {
    /// проверка матрицы от эталона на отличие +- tol.
    pub fn approx_eq(&self, other: &Self, tol: f64) -> bool {
        for i in 0..N {
            for j in 0..M {
                if (self.data[i][j] - other.data[i][j]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }
}




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let z = Matrix::<3, 4>::zeros();
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(z[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn test_identity() {
        let eye = Matrix::<4, 4>::identity();
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert_eq!(eye[(i, j)], 1.0);
                } else {
                    assert_eq!(eye[(i, j)], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_from_rows_and_get() {
        let a = Matrix::from_rows([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]);
        assert_eq!(a[(0, 0)], 1.0);
        assert_eq!(a[(0, 2)], 3.0);
        assert_eq!(a[(1, 1)], 5.0);
    }

    #[test]
    fn test_set_element() {
        let mut a = Matrix::<2, 2>::zeros();
        a.set_element(0, 1, 7.0);
        assert_eq!(a[(0, 1)], 7.0);
    }

    #[test]
    fn test_transpose() {
        let a = Matrix::from_rows([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]);
        let at = a.transpose();
        // at is Matrix<3, 2>
        assert_eq!(at[(0, 0)], 1.0);
        assert_eq!(at[(2, 0)], 3.0);
        assert_eq!(at[(0, 1)], 4.0);
        assert_eq!(at[(2, 1)], 6.0);
    }

    #[test]
    fn test_transpose_of_transpose() {
        let a = Matrix::from_rows([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]);
        let att = a.transpose().transpose();
        assert!(a.approx_eq(&att, 1e-15));
    }

    #[test]
    fn test_row_and_col() {
        let a = Matrix::from_rows([
            [1.0, 2.0],
            [3.0, 4.0],
        ]);
        assert_eq!(a.row(0), [1.0, 2.0]);
        assert_eq!(a.col(1), [2.0, 4.0]);
    }

    #[test]
    fn test_diagonal() {
        let d = Matrix::diagonal(&[2.0, 3.0, 5.0]);
        assert_eq!(d[(0, 0)], 2.0);
        assert_eq!(d[(1, 1)], 3.0);
        assert_eq!(d[(2, 2)], 5.0);
        assert_eq!(d[(0, 1)], 0.0);
    }

    #[test]
    fn test_clone_independence() {
        let a = Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);
        let mut b = a.clone();
        b.set_element(0, 0, 99.0);
        assert_eq!(a[(0, 0)], 1.0); 
        assert_eq!(b[(0, 0)], 99.0);
    }

    #[test]
    fn test_approx_eq() {
        let a = Matrix::from_rows([[1.0, 2.0]]);
        let b = Matrix::from_rows([[1.0 + 1e-10, 2.0 - 1e-10]]);
        assert!(a.approx_eq(&b, 1e-9));   
        assert!(!a.approx_eq(&b, 1e-11)); 
    }

    #[test]
    fn test_64x64_compiles() {
        let big = Matrix::<64, 64>::identity();
        assert_eq!(big[(0, 0)], 1.0);
        assert_eq!(big[(63, 63)], 1.0);
        assert_eq!(big[(0, 63)], 0.0);
    }
}