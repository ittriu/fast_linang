//   matmul    — N×N × N×N
//   lu        — LU::factor(&m)       (только факторизация)
//   inverse   — m.inverse()          (LU + обратный ход × N)
//   eigen     — m.eigen()            (хессенберг + QR)
//   expm      — m.expm()             (Паде 13,13; только biomat)
//
// размеры (стандартные группы):  4×4 - 20×20 - 64×64
//
// входные данные: случайные матрицы N(0.0,1.0), сид зафиксирован для воспроизводимости
// все три библиотеки получают одинаковые данные

//matmul, lu
const SAMPLES_FAST:  usize = 1000;
//inverse, expm
const SAMPLES_MED:   usize = 1000;
//eigen
const SAMPLES_EIGEN: usize = 1000;

use std::time::Duration;
const TIME_FAST:  Duration = Duration::from_secs(30);
const TIME_MED:   Duration = Duration::from_secs(45);
const TIME_EIGEN: Duration = Duration::from_secs(60);


use criterion::{black_box, criterion_group, BatchSize, BenchmarkGroup, Criterion};
use criterion::measurement::WallTime;

use fast_linalg::{Matrix, LU};

use nalgebra::DMatrix;
//чтобы не конфликтовало с fast_linalg::LU
use nalgebra::linalg::LU    as NaLU;
use nalgebra::linalg::Schur as NaSchur;

use faer::Mat as FaerMat;
use faer::prelude::Solve;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

const SEED: u64 = 0xdead_beef_cafe_babe;

fn make_rng() -> StdRng {
    StdRng::seed_from_u64(SEED)
}

fn normal_dist() -> Normal<f64> {
    Normal::new(0.0, 1.0).expect("N(0,1) is always valid")
}

use std::cell::RefCell;

fn rand_biomat<const N: usize>() -> Matrix<N, N> {
    let rng  = RefCell::new(make_rng());
    let dist = normal_dist();
    Matrix::from_fn(|_, _| dist.sample(&mut *rng.borrow_mut()))
}

fn rand_nalgebra(n: usize) -> DMatrix<f64> {
    let rng  = RefCell::new(make_rng());
    let dist = normal_dist();
    DMatrix::from_fn(n, n, |_, _| dist.sample(&mut *rng.borrow_mut()))
}

fn rand_faer(n: usize) -> FaerMat<f64> {
    let rng  = RefCell::new(make_rng());
    let dist = normal_dist();
    FaerMat::from_fn(n, n, |_, _| dist.sample(&mut *rng.borrow_mut()))
}

fn add_matmul_size<const N: usize>(g: &mut BenchmarkGroup<WallTime>) {
    let label = format!("{N}x{N}");

    let a_bm = rand_biomat::<N>();
    let b_bm = rand_biomat::<N>();
    g.bench_function(format!("{label}/biomat"), |b| {
        b.iter(|| black_box(a_bm.clone() * b_bm.clone()))
    });

    let a_na = rand_nalgebra(N);
    let b_na = rand_nalgebra(N);
    g.bench_function(format!("{label}/nalgebra"), |b| {
        b.iter(|| black_box(&a_na * &b_na))
    });

    let a_fa = rand_faer(N);
    let b_fa = rand_faer(N);
    g.bench_function(format!("{label}/faer"), |b| {
        b.iter(|| black_box(&a_fa * &b_fa))
    });
}

fn add_lu_size<const N: usize>(g: &mut BenchmarkGroup<WallTime>) {
    let label = format!("{N}x{N}");

    let m_bm = rand_biomat::<N>();
    g.bench_function(format!("{label}/biomat"), |b| {
        b.iter(|| black_box(LU::factor(&m_bm).ok()))
    });

    let m_na = rand_nalgebra(N);
    g.bench_function(format!("{label}/nalgebra"), |b| {
        b.iter_batched(
            || m_na.clone(),
            |m| black_box(NaLU::new(m)),
            BatchSize::SmallInput,
        )
    });

    let m_fa = rand_faer(N);
    g.bench_function(format!("{label}/faer"), |b| {
        b.iter(|| black_box(m_fa.partial_piv_lu()))
    });
}

fn add_inverse_size<const N: usize>(g: &mut BenchmarkGroup<WallTime>) {
    let label = format!("{N}x{N}");

    let m_bm = rand_biomat::<N>();
    g.bench_function(format!("{label}/biomat"), |b| {
        b.iter(|| black_box(m_bm.inverse().ok()))
    });

    let m_na = rand_nalgebra(N);
    g.bench_function(format!("{label}/nalgebra"), |b| {
        b.iter_batched(
            || m_na.clone(),
            |m| black_box(m.try_inverse()),
            BatchSize::SmallInput,
        )
    });

    // faer: LU + solve(I)
    let m_fa = rand_faer(N);
    g.bench_function(format!("{label}/faer"), |b| {
        b.iter(|| {
            let lu  = m_fa.partial_piv_lu();
            let rhs = FaerMat::<f64>::identity(N, N);
            black_box(lu.solve(rhs))
        })
    });
}

fn add_eigen_size<const N: usize>(g: &mut BenchmarkGroup<WallTime>) {
    let label = format!("{N}x{N}");

    let m_bm = rand_biomat::<N>();
    g.bench_function(format!("{label}/biomat"), |b| {
        b.iter(|| black_box(m_bm.eigen()))
    });

    let m_na = rand_nalgebra(N);
    g.bench_function(format!("{label}/nalgebra"), |b| {
        //shur::new берёт владение
        b.iter_batched(
            || m_na.clone(),
            |m| black_box(NaSchur::new(m)),
            BatchSize::SmallInput,
        )
    });

    let m_fa = rand_faer(N);
    g.bench_function(format!("{label}/faer"), |b| {
        b.iter_batched(
            || m_fa.clone(),
            |m| black_box(m.eigen().ok()),
            BatchSize::SmallInput,
        )
    });
}

fn add_expm_size<const N: usize>(g: &mut BenchmarkGroup<WallTime>) {
    let label = format!("{N}x{N}");
    let m = rand_biomat::<N>();
    // ни nalgebra ни faer не умеют матричную экспоненту
    g.bench_function(format!("{label}/biomat"), |b| {
        b.iter(|| black_box(m.expm()))
    });
}

fn bench_matmul(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul");
    g.sample_size(SAMPLES_FAST);
    g.measurement_time(TIME_FAST);
    add_matmul_size::<4>(&mut g);
    add_matmul_size::<20>(&mut g);
    add_matmul_size::<64>(&mut g);
    g.finish();
}

fn bench_lu(c: &mut Criterion) {
    let mut g = c.benchmark_group("lu");
    g.sample_size(SAMPLES_FAST);
    g.measurement_time(TIME_FAST);
    add_lu_size::<4>(&mut g);
    add_lu_size::<20>(&mut g);
    add_lu_size::<64>(&mut g);
    g.finish();
}

fn bench_inverse(c: &mut Criterion) {
    let mut g = c.benchmark_group("inverse");
    g.sample_size(SAMPLES_MED);
    g.measurement_time(TIME_MED);
    add_inverse_size::<4>(&mut g);
    add_inverse_size::<20>(&mut g);
    add_inverse_size::<64>(&mut g);
    g.finish();
}

fn bench_eigen(c: &mut Criterion) {
    let mut g = c.benchmark_group("eigen");
    g.sample_size(SAMPLES_EIGEN);
    g.measurement_time(TIME_EIGEN);
    add_eigen_size::<4>(&mut g);
    add_eigen_size::<20>(&mut g);
    add_eigen_size::<64>(&mut g);
    g.finish();
}

fn bench_expm(c: &mut Criterion) {
    let mut g = c.benchmark_group("expm");
    g.sample_size(SAMPLES_MED);
    g.measurement_time(TIME_MED);
    add_expm_size::<4>(&mut g);
    add_expm_size::<20>(&mut g);
    add_expm_size::<64>(&mut g);
    g.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_lu,
    bench_inverse,
    bench_eigen,
    bench_expm,
);

fn main() {
    let mut c = Criterion::default().configure_from_args();
    bench_matmul(&mut c);
    bench_lu(&mut c);
    bench_inverse(&mut c);
    bench_eigen(&mut c);
    bench_expm(&mut c);
    c.final_summary();
}