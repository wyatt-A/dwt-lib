use dft_lib::common::{FftDirection, NormalizationType};
use num_complex::Complex32;

#[cfg(feature = "cuda")]
use dft_lib::cu_fft::{cu_fftn as fftn, cu_fftn_batch as fftn_batched};

#[cfg(feature = "fftw")]
use dft_lib::fftw_fft::{fftw_fftn as fftn, fftw_fftn_batched as fftn_batched};

#[cfg(all(not(feature = "cuda"), not(feature = "fftw")))]
use dft_lib::rs_fft::rs_fftn as fftn;

/// returns the dilation factor for the analysis filter. Level must be greater than 0
pub fn dilation_factor(level: usize) -> usize {
    assert!(level > 0, "level must be greater than 0");
    2usize.pow(level as u32 - 1)
}

// /// prepare the analysis kernel from a filter tap and desired level with some size n.
// /// This returns the filter fourier kernel of length n
// pub fn prep_kernel(tap: &[f32], level: usize, n: usize) -> Vec<Complex32> {
//     // get dilation factor (spacing)
//     let s = dilation_factor(level);
//     // calculate the max size of dilated filter
//     let ld = (tap.len() - 1) * s + 1;
//     assert!(ld <= n, "dilated filter length {ld} exceeds padded array size {n}");
//     // allocate the kernel
//     let mut d = vec![Complex32::ZERO; n];
//     // load the filter values into the buffer with spacing s
//     d.chunks_exact_mut(s).zip(tap.iter().rev()).for_each(|(a, b)| a[0].re = *b);
//     // go to fourier domain and return the kernel
//     fftn(&mut d, &[n], FftDirection::Forward, NormalizationType::Unitary);
//     d
// }

/// prepare the analysis kernel from a filter tap and desired level with size n.
/// This returns the circular Fourier kernel of length n, even when the
/// dilated filter support exceeds n.
pub fn prep_kernel(tap: &[f32], level: usize, n: usize) -> Vec<Complex32> {
    assert!(n > 0, "n must be > 0");
    assert!(!tap.is_empty(), "tap must not be empty");

    // dilation factor (spacing)
    let s = dilation_factor(level);

    // build the circularly wrapped spatial kernel of length n
    let mut d = vec![Complex32::ZERO; n];

    // reverse taps here to preserve your current convolution convention
    for (m, &coeff) in tap.iter().rev().enumerate() {
        let idx = (m * s) % n;
        d[idx].re += coeff;
    }

    // go to Fourier domain and return the kernel
    fftn(&mut d, &[n], FftDirection::Forward, NormalizationType::Unitary);
    d
}

pub fn soft_threshold(z: &mut [Complex32], lambda: f32) {
    z.iter_mut().for_each(|x| *x = soft_thresh_complex(*x, lambda));
}

#[inline]
fn soft_thresh_complex(z: Complex32, lambda: f32) -> Complex32 {
    let mag = z.norm();
    if mag <= lambda {
        Complex32::new(0.0, 0.0)
    } else {
        z * (1.0 - lambda / mag)
    }
}

