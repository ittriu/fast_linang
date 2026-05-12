mod matrix;
pub use matrix::Matrix;
mod arithmetic;
//mod error;
mod lu;
mod eigen;
pub use lu::{LU, LuError};
pub mod expm;


