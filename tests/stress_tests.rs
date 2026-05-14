//   $env:BIOMAT_SECS=28800; cargo test --test stress_tests -- --ignored --nocapture



mod common;
use common::gen_matrix::{uniform_matrix, rate_matrix, Range};
use common::lapack::{
    ref_matmul, ref_identity,
    max_abs_error, ref_trace, eigenvalue_trace_sum,
};

use arbtest::arbtest;
use fast_linalg::Matrix;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

fn read_secs() -> u64 {
    std::env::var("BIOMAT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30)
}

macro_rules! progress {
    ($elapsed:expr, $pass:expr, $fail:expr, $last:expr) => {
        eprint!(
            "\r[{:02}:{:02}:{:02}]  iter={:<8}  pass={:<8}  fail={:<4}  last: {}   ",
            $elapsed / 3600,
            ($elapsed % 3600) / 60,
            $elapsed % 60,
            $pass + $fail,
            $pass,
            $fail,
            $last,
        );
    };
}

const DIMS: &[usize] = &[2, 4, 8, 20];

fn check_matmul<const N: usize>(
    u: &mut arbitrary::Unstructured,
    range: Range,
) -> arbitrary::Result<Option<String>> {
    let a = uniform_matrix::<N, N>(u, 0.0, range.hi())?;
    let b = uniform_matrix::<N, N>(u, 0.0, range.hi())?;

    let c_biomat = &a * &b;
    let c_ref    = ref_matmul::<N>(&a, &b);

    let err = max_abs_error::<N, N>(&c_biomat, &c_ref);
    let tol = range.tol(N);

    if err > tol {
        return Ok(Some(format!(
            "MATMUL  N={N}  range={}  err={err:.3e}  tol={tol:.3e}\nA={a}\nB={b}\nbiomat C={c_biomat}\nref C={c_ref}",
            range.label()
        )));
    }
    Ok(None)
}

fn check_inverse<const N: usize>(
    u: &mut arbitrary::Unstructured,
    range: Range,
) -> arbitrary::Result<Option<String>> {
    let m = uniform_matrix::<N, N>(u, 0.0, range.hi())?;

    let inv = match m.inverse() {
        Ok(inv) => inv,
        Err(_)  => return Ok(None), 
    };

    let product = &m * &inv;
    let eye     = ref_identity::<N>();

    let tol = 1e-6 * N as f64;
    let err = max_abs_error::<N, N>(&product, &eye);

    if err > tol {
        return Ok(Some(format!(
            "INVERSE  N={N}  range={}  err={err:.3e}  tol={tol:.3e}\nM={m}\nM·M⁻¹={product}",
            range.label()
        )));
    }
    Ok(None)
}

fn check_eigen_reconstruction<const N: usize>(
    u: &mut arbitrary::Unstructured,
    range: Range,
) -> arbitrary::Result<Option<String>> {
    let m  = uniform_matrix::<N, N>(u, 0.0, range.hi())?;
    let ed = m.eigen();
    let v  = &ed.eigenvectors;

    let mut vd = Matrix::<N, N>::zeros();
    let mut i = 0;
    while i < N {
        let a  = ed.eigenvalues_real[i];
        let bi = ed.eigenvalues_imag[i].abs(); //always use magnitude; sign convention varies
        if i + 1 < N && bi > 1e-10 {
            for row in 0..N {
                vd[(row, i)]     = a * v[(row, i)]     - bi * v[(row, i + 1)];
                vd[(row, i + 1)] = bi * v[(row, i)]    +  a * v[(row, i + 1)];
            }
            i += 2;
        } else {
            //real eigenvalue
            for row in 0..N {
                vd[(row, i)] = a * v[(row, i)];
            }
            i += 1;
        }
    }

    let av  = ref_matmul::<N>(&m, v);
    let tol = 1e-4 * range.hi() * N as f64;
    let err = max_abs_error::<N, N>(&av, &vd);

    if err > tol {
        return Ok(Some(format!(
            "EIGEN_RECON  N={N}  range={}  err={err:.3e}  tol={tol:.3e}\nM={m}",
            range.label()
        )));
    }
    Ok(None)
}

fn check_eigenvalue_trace<const N: usize>(
    u: &mut arbitrary::Unstructured,
    range: Range,
) -> arbitrary::Result<Option<String>> {
    let m  = uniform_matrix::<N, N>(u, 0.0, range.hi())?;
    let ed = m.eigen();

    let tr_matrix = ref_trace::<N>(&m);
    let tr_eigs   = eigenvalue_trace_sum::<N>(&ed.eigenvalues_real);

    let tol = range.tol(N);
    let err = (tr_matrix - tr_eigs).abs();

    if err > tol {
        return Ok(Some(format!(
            "EIGEN_TRACE  N={N}  range={}  tr(M)={tr_matrix:.6e}  Σλ={tr_eigs:.6e}  err={err:.3e}  tol={tol:.3e}\nM={m}",
            range.label()
        )));
    }
    Ok(None)
}

fn check_expm_semigroup<const N: usize>(
    u: &mut arbitrary::Unstructured,
) -> arbitrary::Result<Option<String>> {
    let q = rate_matrix::<N>(u)?;

    let t = (u.arbitrary::<u16>()? as f64 / u16::MAX as f64) * 2.0;
    let s = (u.arbitrary::<u16>()? as f64 / u16::MAX as f64) * 2.0;

    let qt  = &q * t;
    let qs  = &q * s;
    let qts = &q * (t + s);

    let lhs = qt.expm() * qs.expm();
    let rhs = qts.expm();

    let tol = 1e-7 * N as f64;
    let err = max_abs_error::<N, N>(&lhs, &rhs);

    if err > tol {
        return Ok(Some(format!(
            "EXPM_SEMIGROUP  N={N}  t={t:.4}  s={s:.4}  err={err:.3e}  tol={tol:.3e}"
        )));
    }
    Ok(None)
}


#[test]
#[ignore]
fn stress_matmul() {
    let secs = read_secs();
    let mut log = BufWriter::new(File::create("stress_matmul.log").unwrap());
    let (mut pass, mut fail) = (0u64, 0u64);
    let start = Instant::now();

    arbtest(|u| {
        let range = Range::pick(u)?;
        let dim   = *u.choose(DIMS)?;

        let result = match dim {
            2  => check_matmul::<2>(u, range),
            4  => check_matmul::<4>(u, range),
            8  => check_matmul::<8>(u, range),
            20 => check_matmul::<20>(u, range),
            _  => unreachable!(),
        }?;

        match result {
            None => { pass += 1; }
            Some(msg) => {
                fail += 1;
                writeln!(log, "[iter={}  range={}  dim={dim}]\n{msg}\n",
                    pass + fail, range.label()).ok();
            }
        }

        if (pass + fail) % 500 == 0 {
            let e = start.elapsed().as_secs();
            progress!(e, pass, fail, format!("dim={dim} range={}", range.label()));
        }
        Ok(())
    }).budget_ms(secs * 1000);

    eprintln!();
    assert_eq!(fail, 0, "{fail} failures — see stress_matmul.log");
}

#[test]
#[ignore]
fn stress_inverse() {
    let secs = read_secs();
    let mut log = BufWriter::new(File::create("stress_inverse.log").unwrap());
    let (mut pass, mut fail) = (0u64, 0u64);
    let start = Instant::now();

    arbtest(|u| {

        let range = if u.arbitrary::<bool>()? { Range::Unit } else { Range::Hundred };
        let dim   = *u.choose(DIMS)?;

        let result = match dim {
            2  => check_inverse::<2>(u, range),
            4  => check_inverse::<4>(u, range),
            8  => check_inverse::<8>(u, range),
            20 => check_inverse::<20>(u, range),
            _  => unreachable!(),
        }?;

        match result {
            None => { pass += 1; }
            Some(msg) => {
                fail += 1;
                writeln!(log, "[iter={}  range={}  dim={dim}]\n{msg}\n",
                    pass + fail, range.label()).ok();
            }
        }

        if (pass + fail) % 500 == 0 {
            let e = start.elapsed().as_secs();
            progress!(e, pass, fail, format!("dim={dim} range={}", range.label()));
        }
        Ok(())
    }).budget_ms(secs * 1000);

    eprintln!();
    assert_eq!(fail, 0, "{fail} failures");
}

#[test]
#[ignore]
fn stress_eigen() {
    // Tests both V·D·V⁻¹ ≈ M  AND  tr(M) ≈ Σλ in one pass.
    let secs = read_secs();
    let mut log = BufWriter::new(File::create("stress_eigen.log").unwrap());
    let (mut pass, mut fail) = (0u64, 0u64);
    let start = Instant::now();

    arbtest(|u| {
        let range = if u.arbitrary::<bool>()? { Range::Unit } else { Range::Hundred };
        let dim   = *u.choose(DIMS)?;

        let r1 = match dim {
            2  => check_eigen_reconstruction::<2>(u, range),
            4  => check_eigen_reconstruction::<4>(u, range),
            8  => check_eigen_reconstruction::<8>(u, range),
            20 => check_eigen_reconstruction::<20>(u, range),
            _  => unreachable!(),
        }?;

        let r2 = match dim {
            2  => check_eigenvalue_trace::<2>(u, range),
            4  => check_eigenvalue_trace::<4>(u, range),
            8  => check_eigenvalue_trace::<8>(u, range),
            20 => check_eigenvalue_trace::<20>(u, range),
            _  => unreachable!(),
        }?;

        if r1.is_some() || r2.is_some() {
            fail += 1;
            for msg in [r1, r2].into_iter().flatten() {
                writeln!(log, "[iter={}  range={}  dim={dim}]\n{msg}\n",
                    pass + fail, range.label()).ok();
            }
        } else {
            pass += 1;
        }

        if (pass + fail) % 500 == 0 {
            let e = start.elapsed().as_secs();
            progress!(e, pass, fail, format!("dim={dim} range={}", range.label()));
        }
        Ok(())
    }).budget_ms(secs * 1000);

    eprintln!();
    assert_eq!(fail, 0, "{fail} failures");
}

#[test]
#[ignore]
fn stress_expm_semigroup() {
    let secs = read_secs();
    let mut log = BufWriter::new(File::create("stress_expm.log").unwrap());
    let (mut pass, mut fail) = (0u64, 0u64);
    let start = Instant::now();

    arbtest(|u| {
        let dim = *u.choose(DIMS)?;

        let result = match dim {
            2  => check_expm_semigroup::<2>(u),
            4  => check_expm_semigroup::<4>(u),
            8  => check_expm_semigroup::<8>(u),
            20 => check_expm_semigroup::<20>(u),
            _  => unreachable!(),
        }?;

        match result {
            None => { pass += 1; }
            Some(msg) => {
                fail += 1;
                writeln!(log, "[iter={}  dim={dim}]\n{msg}\n", pass + fail).ok();
            }
        }

        if (pass + fail) % 200 == 0 {
            let e = start.elapsed().as_secs();
            progress!(e, pass, fail, format!("dim={dim}"));
        }
        Ok(())
    }).budget_ms(secs * 1000);

    eprintln!();
    assert_eq!(fail, 0, "{fail} failures");
}




#[test]
fn eigen_convergence_repro() {
    use common::gen_matrix::uniform_matrix;
    use arbtest::arbtest;

    arbtest(|u| {
        let range_pick = u.arbitrary::<bool>()?;
        let range_hi   = if range_pick { 1.0 } else { 100.0 };
        let dim        = *u.choose(&[2usize, 4, 8, 20])?;
        match dim {
            2  => { let m = uniform_matrix::<2,  2>(u, 0.0, range_hi)?; println!("N=2\n{m}"); let _ = m.eigen(); }
            4  => { let m = uniform_matrix::<4,  4>(u, 0.0, range_hi)?; println!("N=4\n{m}"); let _ = m.eigen(); }
            8  => { let m = uniform_matrix::<8,  8>(u, 0.0, range_hi)?; println!("N=8\n{m}"); let _ = m.eigen(); }
            20 => { let m = uniform_matrix::<20,20>(u, 0.0, range_hi)?; println!("N=20\n{m}"); let _ = m.eigen(); }
            _  => unreachable!(),
        }
        Ok(())
    }).seed(0x2235323500000096);
}