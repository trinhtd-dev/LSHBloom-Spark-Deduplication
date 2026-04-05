use numpy::{PyReadonlyArrayDyn, ndarray::ArrayViewD};
use pyo3::prelude::*;
#[pymodule]
fn pyhash<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    fn hash(
        arr: ArrayViewD<'_, u64>
        ) -> u64 {
        let total = arr.iter().fold(0u128, |acc, x| { acc + *x as u128});
        let hashed = total % ((1 << 61) - 1);
        (hashed & 0xFFFF_FFFF_FFFF_FFFF).try_into().unwrap()
    }
    #[pyfn(m)]
    #[pyo3(name="hash")]
    fn hash_py<'py>(x: PyReadonlyArrayDyn<'py, u64>) -> PyResult<u64> {
        Ok(hash(x.as_array()))
    }
    Ok(())
}