use crate::wavelet::{Wavelet, WaveletFilter};
use dft_lib::common::{FftDirection, NormalizationType};
use dft_lib::fftw_fft::fftw_fftn;
use num_complex::Complex32;

struct SWT2Plan {
    /// dimensions of signal (image)
    dims: [usize; 2],
    /// number of decomp/recon levels
    levels: usize,

    /// decomposition LL kernel
    d_ll: Vec<Vec<Complex32>>,
    /// decomposition LH kernel
    d_lh: Vec<Vec<Complex32>>,
    /// decomposition HL kernel
    d_hl: Vec<Vec<Complex32>>,
    /// decomposition HH kernel
    d_hh: Vec<Vec<Complex32>>,

    /// reconstruction LL kernel
    r_ll: Vec<Vec<Complex32>>,
    /// reconstruction LH kernel
    r_lh: Vec<Vec<Complex32>>,
    /// reconstruction HL kernel
    r_hl: Vec<Vec<Complex32>>,
    /// reconstruction HH kernel
    r_hh: Vec<Vec<Complex32>>,

    /// composite transfer function
    h: Vec<Vec<Complex32>>,

    w: Wavelet<f32>,
}

impl SWT2Plan {
    fn forward_level(&self, src: &[Complex32], dst: &mut [Complex32]) -> Vec<Complex32> {
        todo!()
    }

    fn inverse_level(&self, src: &[Complex32], dst: &mut [Complex32]) -> Vec<Complex32> {
        todo!()
    }
    
    pub fn new(nx: usize, ny: usize, levels: usize, w: &Wavelet<f32>) -> SWT2Plan {

        // fourier kernels
        let mut d_ll = vec![];
        let mut d_lh = vec![];
        let mut d_hl = vec![];
        let mut d_hh = vec![];
        let mut r_ll = vec![];
        let mut r_lh = vec![];
        let mut r_hl = vec![];
        let mut r_hh = vec![];
        let mut h = vec![];


        // calculate kernels
        for level in 1..=levels {
            let [d_ll_k, d_lh_k, d_hl_k, d_hh_k] = calc_kernel_2d(nx, ny, w.lo_d(), w.hi_d(), level);
            let [r_ll_k, r_lh_k, r_hl_k, r_hh_k] = calc_kernel_2d(nx, ny, w.lo_r(), w.hi_r(), level);
            let tf = calc_transfer_fn([&d_ll_k, &d_lh_k, &d_hl_k, &d_hh_k, &r_ll_k, &r_lh_k, &r_hl_k, &r_hh_k]);
            h.push(tf);
            d_ll.push(d_ll_k);
            d_lh.push(d_lh_k);
            d_hl.push(d_hl_k);
            d_hh.push(d_hh_k);
            r_ll.push(r_ll_k);
            r_lh.push(r_lh_k);
            r_hl.push(r_hl_k);
            r_hh.push(r_hh_k);
        }

        SWT2Plan {
            dims: [nx, ny],
            levels,
            d_ll,
            d_lh,
            d_hl,
            d_hh,
            r_ll,
            r_lh,
            r_hl,
            r_hh,
            h,
            w: w.clone(),
        }
    }
}

fn calc_transfer_fn(kernels: [&Vec<Complex32>; 8]) -> Vec<Complex32> {
    let n = kernels[0].len();
    let mut h = vec![Complex32::ZERO; n];
    for i in 0..n {
        // LL * LL
        h[i] =
            kernels[0][i] * kernels[4][i] +
                kernels[1][i] * kernels[5][i] +
                kernels[2][i] * kernels[6][i] +
                kernels[3][i] * kernels[7][i];
    }
    h
}


fn calc_kernel_2d(nx: usize, ny: usize, lo: &[f32], hi: &[f32], level: usize) -> [Vec<Complex32>; 4] {
    let klx = prep_kernel(lo, level, nx);
    let khx = prep_kernel(hi, level, nx);
    let kly = prep_kernel(lo, level, ny);
    let khy = prep_kernel(hi, level, ny);

    let mut ll_k = vec![Complex32::ZERO; nx * ny];
    let mut lh_k = vec![Complex32::ZERO; nx * ny];
    let mut hl_k = vec![Complex32::ZERO; nx * ny];
    let mut hh_k = vec![Complex32::ZERO; nx * ny];

    for j in 0..ny {
        for i in 0..nx {
            let idx = nx * j + i;
            ll_k[idx] = klx[i] * kly[j];
            lh_k[idx] = klx[i] * khy[j];
            hl_k[idx] = khx[i] * kly[j];
            hh_k[idx] = khx[i] * khy[j];
        }
    }

    [ll_k, lh_k, hl_k, hh_k]
}


/// returns the dilation factor for the analysis filter. Level must be greater than 0
fn dilation_factor(level: usize) -> usize {
    assert!(level > 0, "level must be greater than 0");
    2usize.pow(level as u32 - 1)
}

/// prepare the analysis kernel from a filter tap and desired level with some size n.
/// This returns the filter fourier kernel of length n
fn prep_kernel(tap: &[f32], level: usize, n: usize) -> Vec<Complex32> {
    // get dilation factor (spacing)
    let s = dilation_factor(level);
    // calculate the max size of dilated filter
    let ld = (tap.len() - 1) * s + 1;
    assert!(ld <= n, "dilated filter length {ld} exceeds padded array size {n}");
    // allocate the kernel
    let mut d = vec![Complex32::ZERO; n];
    // load the filter values into the buffer with spacing s
    d.chunks_exact_mut(s).zip(tap.iter().rev()).for_each(|(a, b)| a[0].re = *b);
    // go to fourier domain and return the kernel
    fftw_fftn(&mut d, &[n], FftDirection::Forward, NormalizationType::Unitary);
    d
}


//    % Dilation factor
//     s = 2^(level - 1);
//
//     % Dilate by inserting zeros between coefficients
//     L = numel(filter);
//     Ld = (L - 1) * s + 1;
//
//     if Ld > n
//         error('Dilated filter length (%d) exceeds n (%d).', Ld, n);
//     end
//
//     h = zeros(1, Ld);
//     h(1:s:end) = filter;
//
//     % Shift so the filter center is at zero lag for FFT convolution
//     %h = circshift(h, -floor(Ld / 2));
//
//     k = zeros(1,n);
//     k(1:Ld) = h;
//
//     h = k;
//     %h = circshift(k, -floor(Ld / 2));
//
//     % Pad/truncate to FFT length and transform
//     f = fft(h);