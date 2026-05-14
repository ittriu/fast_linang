use crate::matrix::Matrix;



// ──────────────────────────────────────────────
// ─── Результат собственного разложения матрицы ───
// ──────────────────────────────────────────────
// eigenvalues_real[i] + i·eigenvalues_imag[i] — i-е собственное значение
// eigenvectors хранит векторы в стиле LAPACK dgeev:
// вещественные — столбец i, комплексные — столбец i (Re) и i+1 (Im)

pub struct EigenDecomposition<const N: usize> {
    pub eigenvalues_real: [f64; N],
    pub eigenvalues_imag: [f64; N],
    pub eigenvectors:     Matrix<N, N>,
}


impl<const N: usize> Matrix<N, N> {
    //cобственное разложение: возвращает собственные значения и векторы матрицы
    //оно использует хассенберга и QR Фрэнсиса с двойным сдвигом
    pub fn eigen(&self) -> EigenDecomposition<N> {
        let (mut h, mut z) = hessenberg_reduce_with_z(self); //diff

        francis_qr(&mut h, &mut z); //diff

        let (eigenvalues_real, eigenvalues_imag) = tung_tung_tung_schur(&h);

        let mut evecs_schur = Matrix::<N, N>::zeros();
        let mut i = 0;
        while i < N {
            if eigenvalues_imag[i].abs() > 1e-10 {
                let (yr, yi) = schur_complex_eigenvec(
                    &h, i, eigenvalues_real[i], eigenvalues_imag[i]
                );
                for r in 0..N {
                    evecs_schur.data[r][i]     = yr[r];
                    evecs_schur.data[r][i + 1] = yi[r];
                }
                i += 2;
            } else {
                let y = schur_real_eigenvec(&h, i);
                for r in 0..N { evecs_schur.data[r][i] = y[r]; }
                i += 1;
            }
        }

        let eigenvectors = &z * &evecs_schur;

        EigenDecomposition { eigenvalues_real, eigenvalues_imag, eigenvectors }
    }
}


// T - форма Шура, p - индекс 1x1 блока
// решаем (T - lamb*I)y = 0 обратным ходом вверх от строки p-1


fn schur_real_eigenvec<const N: usize>(t: &Matrix<N, N>, p: usize) -> [f64; N] {
    let lambda = t.data[p][p];
    let mut y = [0.0f64; N];
    y[p] = 1.0;
    for i in (0..p).rev() {
        let mut acc = 0.0f64;
        for j in (i + 1)..=p {
            acc -= t.data[i][j] * y[j];
        }
        let diag = t.data[i][i] - lambda;
        y[i] = if diag.abs() > 1e-14 { acc / diag } else { 0.0 };
    }

    let norm = y.iter().map(|x| x * x).sum::<f64>().sqrt().max(f64::EPSILON);
    for x in y.iter_mut() { *x /= norm; }
    y
}


// решаем связанную систему для (yr, yi) по аналогии с LAPACK dtrevc
// yr и yi - вещественная и мнимая части вектора, нормируем в конце

fn schur_complex_eigenvec<const N: usize>(
    t: &Matrix<N, N>, p: usize, re: f64, im: f64
) -> ([f64; N], [f64; N]) {
    let mut yr = [0.0f64; N];
    let mut yi = [0.0f64; N];
    yr[p]     = 1.0;
    yi[p + 1] = 1.0;

    for i in (0..p).rev() {
        let mut accr = 0.0f64;
        let mut acci = 0.0f64;
        for j in (i + 1)..N {
            let tij = t.data[i][j];
            accr -= tij * yr[j];
            acci -= tij * yi[j];
        }
        let dre = t.data[i][i] - re;
        let det = dre * dre + im * im;
        if det > 1e-28 {
            yr[i] = (dre * accr + im * acci) / det;
            yi[i] = (dre * acci - im * accr) / det;
        }
    }

    let norm = (yr.iter().chain(yi.iter()).map(|x| x * x).sum::<f64>())
        .sqrt().max(f64::EPSILON);
    for x in yr.iter_mut() { *x /= norm; }
    for x in yi.iter_mut() { *x /= norm; }
    (yr, yi)
}


// 
// Редукция к форме Хессенберга + накопление Z 

fn hessenberg_reduce_with_z<const N: usize>(a: &Matrix<N, N>) -> (Matrix<N, N>, Matrix<N, N>) {
    let mut h = a.clone();
    let mut z = Matrix::<N, N>::identity();

    for k in 0..N.saturating_sub(2) {
        let mut norm_sq = 0.0f64;
        for i in (k + 1)..N { norm_sq += h.data[i][k] * h.data[i][k]; }
        let norm = norm_sq.sqrt();
        if norm < f64::EPSILON { continue; }

        let mut v = [0.0f64; N];
        for i in (k + 1)..N { v[i] = h.data[i][k]; }
        let sign = if v[k + 1] >= 0.0 { 1.0 } else { -1.0 };
        v[k + 1] += sign * norm;

        let vv: f64 = ((k + 1)..N).map(|i| v[i] * v[i]).sum();
        if vv < f64::EPSILON { continue; }
        let two_over_vv = 2.0 / vv;

        for j in k..N {
            let mut dot = 0.0f64;
            for i in (k + 1)..N { dot += v[i] * h.data[i][j]; }
            let f = two_over_vv * dot;
            for i in (k + 1)..N { h.data[i][j] -= f * v[i]; }
        }

        for i in 0..N {
            let mut dot = 0.0f64;
            for j in (k + 1)..N {
                dot += h.data[i][j] * v[j];
            }
            if dot == 0.0 { continue; }
            let f = two_over_vv * dot;
            for j in (k + 1)..N {
                h.data[i][j] -= f * v[j];
            }
        }

        for i in 0..N {
            let mut dot = 0.0f64;
            for j in (k + 1)..N { dot += z.data[i][j] * v[j]; }
            let f = two_over_vv * dot;
            for j in (k + 1)..N { z.data[i][j] -= f * v[j]; }
        }
    }

    (h, z)
}

//______________________________
//QR фрэнсиса с двойным сдвигом 
// Главный цикл: сжимаем активное окно пока не останется 0 или 1 строка.
// максимум 100*N итераций

fn francis_qr<const N: usize>(h: &mut Matrix<N, N>, z: &mut Matrix<N, N>) { //diff

    let mut zt = z.transpose(); //diff

    let mut active_end = N;
    let mut iter       = 0;
    let max_iter       = 100 * N;

    while active_end > 1 && iter < max_iter {
        let active_start = find_deflation_point(h, active_end);

        if active_end - active_start <= 2 {
            if active_end - active_start == 2 { //diff
                let p = active_start; //diff
                let a = h.data[p][p]; //diff
                let b = h.data[p][p + 1]; //diff
                let c = h.data[p + 1][p]; //diff
                let d = h.data[p + 1][p + 1]; //diff
                // если поддиагональный не нулевой и disc >= 0 — вещественные лямбды,
                // нужно привести блок к верхнетреугольному виду (иначе eigenvec solver сломается)
                if c.abs() > f64::EPSILON * (a.abs() + d.abs()) { //diff
                    let tr   = a + d; //diff
                    let det  = a * d - b * c; //diff
                    let disc = tr * tr - 4.0 * det; //diff
                    if disc >= 0.0 { //diff
                        standardize_2x2_real(h, &mut zt, p); //diff
                    } //diff
                } //diff
            } //diff
            active_end = active_start;
            continue;
        }

        let (sigma_sum, sigma_prod) = if iter % 10 == 9 {
            let s1 = h.data[active_end - 1][active_end - 2].abs();
            let s2 = if active_end >= 3 {
                h.data[active_end - 2][active_end - 3].abs()
            } else { 0.0 };
            let t = s1 + s2;
            (1.5 * t, 0.8125 * t * t)
        } else {
            wilkinson_shift(h, active_end)
        };

        francis_double_step(h, &mut zt, active_start, active_end, sigma_sum, sigma_prod); //diff
        iter += 1;
    }

    *z = zt.transpose(); //diff - транспонируем обратно в Z

    debug_assert!(active_end <= 1,
        "francis_qr: не сошлось за {} итераций", max_iter);
}

// приводим 2x2 блок с вещественными лямбдами к верхнетреугольному виду через Givens
// строим Q = [[cs, -sn],[sn, cs]], применяем Q^T*H*Q и накапливаем ZT <- Q^T*ZT

fn standardize_2x2_real<const N: usize>(h: &mut Matrix<N, N>, zt: &mut Matrix<N, N>, p: usize) { //diff
    let a    = h.data[p][p]; //diff
    let b    = h.data[p][p + 1]; //diff
    let c    = h.data[p + 1][p]; //diff
    let d    = h.data[p + 1][p + 1]; //diff
    let tr   = a + d; //diff
    let det  = a * d - b * c; //diff
    let disc = tr * tr - 4.0 * det; //diff
    let sq   = disc.max(0.0).sqrt(); //diff
    let lam1 = (tr + sq) * 0.5; //diff
    let (vx, vy) = if b.hypot(lam1 - a) >= (lam1 - d).hypot(c) { //diff
        (b, lam1 - a) //diff
    } else { //diff
        (lam1 - d, c) //diff
    }; //diff
    let norm = vx.hypot(vy); //diff
    if norm < f64::EPSILON { return; } //diff
    let cs = vx / norm; //diff
    let sn = vy / norm; //diff
    // левое умножение Q^T: строчки p и p+1, все столбцы j
    for j in 0..N { //diff
        let t0 =  cs * h.data[p][j] + sn * h.data[p + 1][j]; //diff
        let t1 = -sn * h.data[p][j] + cs * h.data[p + 1][j]; //diff
        h.data[p][j]     = t0; //diff
        h.data[p + 1][j] = t1; //diff
    } //diff
    // правое умножение Q: столбцы p и p+1, все строки i
    for i in 0..N { //diff
        let t0 =  cs * h.data[i][p] + sn * h.data[i][p + 1]; //diff
        let t1 = -sn * h.data[i][p] + cs * h.data[i][p + 1]; //diff
        h.data[i][p]     = t0; //diff
        h.data[i][p + 1] = t1; //diff
    } //diff
    h.data[p + 1][p] = 0.0; //diff
    // Z_new = Z_old*Q  =>  ZT_new = Q^T * ZT_old
    // обновляем строки p и p+1 матрицы ZT
    for j in 0..N { //diff
        let t0 =  cs * zt.data[p][j] + sn * zt.data[p + 1][j]; //diff
        let t1 = -sn * zt.data[p][j] + cs * zt.data[p + 1][j]; //diff
        zt.data[p][j]     = t0; //diff
        zt.data[p + 1][j] = t1; //diff
    } //diff
} //diff

// тут мы ищем самую нижнюю строку где поддиагональный элемент пренебрежимо мал.
// если нашли то обнуляем его на месте и возвращаем индекс
// если нет - возвращаем 0

fn find_deflation_point<const N: usize>(h: &mut Matrix<N, N>, active_end: usize) -> usize {
    for i in (0..active_end.saturating_sub(1)).rev() {
        let sub   = h.data[i + 1][i].abs();
        let scale = h.data[i][i].abs() + h.data[i + 1][i + 1].abs();
        if sub <= f64::EPSILON * scale {
            h.data[i + 1][i] = 0.0;
            return i + 1;
        }
    }
    0
}


// Сдвиг уилкинсона 
// берём нижний 2x2 блок активного окна, возвращаем (tr, det)
// они становятся коэффициентами в двойном шаге френсиса

fn wilkinson_shift<const N: usize>(h: &Matrix<N, N>, end: usize) -> (f64, f64) {
    let a = h.data[end - 2][end - 2];
    let d = h.data[end - 1][end - 1];
    (a + d, a * d - h.data[end - 2][end - 1] * h.data[end - 1][end - 2])
}


fn francis_double_step<const N: usize>(
    h: &mut Matrix<N, N>,
    zt: &mut Matrix<N, N>, //diff 
    start: usize, end: usize,
    shift_sum: f64, shift_prod: f64,
) {
    let s = start;
    let e = end;

    let sub = h.data[s + 1][s];
    let p0 = h.data[s][s] * h.data[s][s] + h.data[s][s + 1] * sub
        - shift_sum * h.data[s][s]
        + shift_prod;
    let p1 = sub * (h.data[s][s] + h.data[s + 1][s + 1] - shift_sum);
    let p2 = h.data[s + 2][s + 1] * sub;
    let mut x = [p0, p1, p2];

    for k in 0..(e - s - 2) {
        let r = s + k;

        let mut v        = x;
        let norm         = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        if norm < f64::EPSILON {
            if k + 1 < e - s - 2 {
                x = [h.data[r + 1][r], h.data[r + 2][r], h.data[r + 3][r]];
            }
            continue;
        }
        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0]    += sign * norm;
        let two_over_vv = 2.0 / (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);

        let col_start = if k == 0 { r } else { r - 1 };
        for j in col_start..e {
            let dot = v[0] * h.data[r][j] + v[1] * h.data[r + 1][j] + v[2] * h.data[r + 2][j];
            let f = two_over_vv * dot;
            h.data[r][j]     -= f * v[0];
            h.data[r + 1][j] -= f * v[1];
            h.data[r + 2][j] -= f * v[2];
        }

        let row_limit = (r + 4).min(e);
        for i in 0..row_limit {
            let dot = v[0] * h.data[i][r] + v[1] * h.data[i][r + 1] + v[2] * h.data[i][r + 2];
            let f = two_over_vv * dot;
            h.data[i][r]     -= f * v[0];
            h.data[i][r + 1] -= f * v[1];
            h.data[i][r + 2] -= f * v[2];
        }

        for j in 0..N { //diff
            let dot = v[0] * zt.data[r][j] + v[1] * zt.data[r + 1][j] + v[2] * zt.data[r + 2][j]; //diff
            let f = two_over_vv * dot; //diff
            zt.data[r][j]     -= f * v[0]; //diff
            zt.data[r + 1][j] -= f * v[1]; //diff
            zt.data[r + 2][j] -= f * v[2]; //diff
        } //diff

        if k + 1 < e - s - 2 {
            x = [h.data[r + 1][r], h.data[r + 2][r], h.data[r + 3][r]];
        }
    }

    let rt = e - 2;
    let rb = e - 1;
    let cf = e - 3;
    let x0 = h.data[rt][cf];
    let v1 = h.data[rb][cf];
    let n2  = (x0*x0 + v1*v1).sqrt();
    if n2 > f64::EPSILON {
        let sign        = if x0 >= 0.0 { 1.0 } else { -1.0 };
        let v0          = x0 + sign * n2;
        let two_over_vv = 2.0 / (v0*v0 + v1*v1);
        for j in cf..N {
            let dot = v0 * h.data[rt][j] + v1 * h.data[rb][j];
            let f   = two_over_vv * dot;
            h.data[rt][j] -= f * v0;
            h.data[rb][j] -= f * v1;
        }
        for i in 0..e {
            let dot = v0 * h.data[i][rt] + v1 * h.data[i][rb];
            let f   = two_over_vv * dot;
            h.data[i][rt] -= f * v0;
            h.data[i][rb] -= f * v1;
        }
        // trailing 2x2, последовательный доступ по j
        for j in 0..N { //diff
            let dot = v0 * zt.data[rt][j] + v1 * zt.data[rb][j]; //diff
            let f   = two_over_vv * dot; //diff
            zt.data[rt][j] -= f * v0; //diff
            zt.data[rb][j] -= f * v1; //diff
        } //diff
    }
}


// 
//  читаем собственные значения из формы Шура !!

fn tung_tung_tung_schur<const N: usize>(t: &Matrix<N, N>) -> ([f64; N], [f64; N]) {
    let mut re = [0.0f64; N];
    let mut im = [0.0f64; N];
    let mut i  = 0;

    while i < N {
        let is_2x2 = i + 1 < N
            && t.data[i+1][i].abs() > f64::EPSILON * (t.data[i][i].abs() + t.data[i+1][i+1].abs());

        if is_2x2 {
            let a   = t.data[i][i];
            let d   = t.data[i+1][i+1];
            let det = a * d - t.data[i][i+1] * t.data[i+1][i];
            let tr   = a + d;
            let disc = tr*tr - 4.0 * det;

            if disc >= 0.0 {
                let sq  = disc.sqrt();
                let big = if tr >= 0.0 { (tr + sq) * 0.5 } else { (tr - sq) * 0.5 };
                let sml = if big.abs() > f64::EPSILON { det / big } else { tr - big };
                re[i] = big;   im[i]   = 0.0;
                re[i+1] = sml; im[i+1] = 0.0;
            } else {
                let half_tr = tr * 0.5;
                let half_im = (-disc).sqrt() * 0.5;
                re[i]   = half_tr;
                im[i]   =  half_im;
                re[i+1] = half_tr;
                im[i+1] = -half_im;
            }
            i += 2;
        } else {
            re[i] = t.data[i][i];
            im[i] = 0.0;
            i += 1;
        }
    }

    (re, im)
}
