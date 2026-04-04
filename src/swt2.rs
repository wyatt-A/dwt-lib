use crate::wavelet::{Wavelet, WaveletFilter};
use dft_lib::common::{FftDirection, NormalizationType};
use dft_lib::fftw_fft::{fftw_fftn, fftw_fftn_batched};
use num_complex::Complex32;
use std::cell::RefCell;
use std::ops::Range;
use crate::swt::{prep_kernel, soft_threshold};

pub struct SWT2Plan {
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

    /// wavelet for analysis
    w: Wavelet<f32>,

    /// temp buffers for calculations
    tmp_x: RefCell<Vec<Complex32>>,
    tmp_y: RefCell<Vec<Complex32>>,
}

impl SWT2Plan {
    /// soft threshold only the detail bands, leaving the approx band intact. t_domain is the
    /// entire transform domain.
    pub fn soft_thresh_detail(&self, t_domain: &mut [Complex32], lambda: f32) {
        soft_threshold(&mut t_domain[self.subband_size()..], lambda);
    }

    /// soft threshold all bands, including the approximation band
    pub fn soft_thresh(&self, t_domain: &mut [Complex32], lambda: f32) {
        soft_threshold(t_domain, lambda);
    }

    pub fn reconstruct(&self, src: &[Complex32], dst: &mut [Complex32]) {
        let n = self.subband_size();

        assert_eq!(src.len(), self.t_domain_size());
        assert_eq!(dst.len(), n);

        // current approximation during reconstruction
        let mut approx = src[self.approx_range()].to_vec();

        // scratch for one level input to recon_level: [LL, LH, HL, HH]
        let mut bands = vec![Complex32::ZERO; 4 * n];

        // scratch for reconstructed image at each stage
        let mut tmp = vec![Complex32::ZERO; n];

        for level in (0..self.levels).rev() {
            // fill [LL, LH, HL, HH]
            bands[..n].copy_from_slice(&approx);

            let dr = self.detail_range(level);
            bands[n..4 * n].copy_from_slice(&src[dr]);

            self.recon_level(level, &bands, &mut tmp);

            approx.copy_from_slice(&tmp);
        }

        dst.copy_from_slice(&approx);
    }

    pub fn decompose(&self, src: &[Complex32], dst: &mut [Complex32]) {
        let n = self.subband_size();

        assert_eq!(src.len(), n);
        assert_eq!(dst.len(), self.t_domain_size());

        // current approximation being decomposed
        let mut approx = src.to_vec();

        // scratch for one level: [LL, LH, HL, HH]
        let mut bands = vec![Complex32::ZERO; 4 * n];

        for level in 0..self.levels {
            self.decomp_level(level, &approx, &mut bands);

            // store details for this level
            let dr = self.detail_range(level);
            dst[dr].copy_from_slice(&bands[n..4 * n]);

            // propagate LL to next level
            approx.copy_from_slice(&bands[..n]);
        }

        // store final approximation LL_J
        let ar = self.approx_range();
        dst[ar].copy_from_slice(&approx);
    }

    pub fn approx_range(&self) -> Range<usize> {
        0..self.subband_size()
    }

    /// Range for the 3 detail bands (LH, HL, HH) for a given level.
    /// level = 0 is finest, level = self.levels - 1 is coarsest.
    pub fn detail_range(&self, level: usize) -> Range<usize> {
        assert!(level < self.levels);
        let n = self.subband_size();
        let block = self.levels - 1 - level;
        let start = n + block * 3 * n;
        let stop = start + 3 * n;
        start..stop
    }

    /// returns the size of the transform domain. This is 3 * n_levels + 1 subbands
    pub fn t_domain_size(&self) -> usize {
        self.subband_size() * self.t_bands()
    }

    /// returns the number of subbands for the decomposition
    pub fn t_bands(&self) -> usize {
        3 * self.levels + 1
    }

    /// decomposes src into 4 subbands, storing them in dst with the order (LL, LH, HL, HH).
    /// This implies that dst.len() == 4 * src.len()
    pub fn decomp_level(&self, level: usize, src: &[Complex32], dst: &mut [Complex32]) {
        let mut tmp = self.tmp_x.borrow_mut();
        let x = tmp.as_mut_slice();
        x.copy_from_slice(src);

        fftw_fftn(x, &self.dims, FftDirection::Forward, NormalizationType::Unitary);

        let d_ll = self.d_ll[level].as_slice();
        let d_lh = self.d_lh[level].as_slice();
        let d_hl = self.d_hl[level].as_slice();
        let d_hh = self.d_hh[level].as_slice();
        let kernels = [d_ll, d_lh, d_hl, d_hh];
        let mut bands = dst.chunks_mut(self.subband_size()).collect::<Vec<_>>();
        bands.iter_mut().zip(kernels).for_each(|(band, kern)| {
            band.iter_mut().zip(kern.iter()).zip(x.iter()).for_each(|((b, k), x)| {
                *b = *x * k;
            });
            fftw_fftn(band, &self.dims, FftDirection::Inverse, NormalizationType::Unitary);
        });
    }

    /// reconstructs from 4 subbands in src (LL,LH,HL,HH), storing the result in dst. This implies
    /// that src.len() == 4 * dst.len()
    pub fn recon_level(&self, level: usize, src: &[Complex32], dst: &mut [Complex32]) {
        // small value for numerically stable divisions
        const EPS: f32 = 1e-6;
        assert_eq!(dst.len(), self.subband_size());

        let mut tmp = self.tmp_y.borrow_mut();
        let y = tmp.as_mut_slice();
        y.copy_from_slice(src);

        fftw_fftn_batched(y, &self.dims, 4, FftDirection::Forward, NormalizationType::Unitary);

        let r_ll = self.r_ll[level].as_slice();
        let r_lh = self.r_lh[level].as_slice();
        let r_hl = self.r_hl[level].as_slice();
        let r_hh = self.r_hh[level].as_slice();
        let kernels = [r_ll, r_lh, r_hl, r_hh];
        let h = self.h[level].as_slice();

        let mut bands = y.chunks_exact_mut(self.subband_size()).collect::<Vec<_>>();

        dst.iter_mut().zip(h).enumerate().for_each(|(i, (x, h))| {
            *x = kernels.iter().zip(bands.iter()).map(|(k, b)| {
                k[i] * b[i]
            }).sum::<Complex32>();

            let denom = if h.norm() < EPS {
                Complex32::new(EPS, 0.0)
            } else {
                *h
            };
            *x /= denom;
        });

        fftw_fftn(dst, &self.dims, FftDirection::Inverse, NormalizationType::Unitary);
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
            let tf = calc_transfer_fn_2d([&d_ll_k, &d_lh_k, &d_hl_k, &d_hh_k, &r_ll_k, &r_lh_k, &r_hl_k, &r_hh_k]);
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

        // temp buffer to avoid re-allocations
        let tmp_x = RefCell::new(vec![Complex32::ZERO; nx * ny]);
        let tmp_y = RefCell::new(vec![Complex32::ZERO; nx * ny * 4]);

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
            tmp_x,
            tmp_y,
        }
    }

    pub fn subband_size(&self) -> usize {
        self.dims[0] * self.dims[1]
    }
}

fn calc_transfer_fn_2d(kernels: [&Vec<Complex32>; 8]) -> Vec<Complex32> {
    let n = kernels[0].len();
    let mut h = vec![Complex32::ZERO; n];
    for i in 0..n {
        // LL * LL
        h[i] =
            kernels[0][i] * kernels[4][i] +
                kernels[1][i] * kernels[5][i] +
                kernels[2][i] * kernels[6][i] +
                kernels[3][i] * kernels[7][i]
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

    let scale = ((ny * nx) as f32).sqrt() / 4f32.sqrt();
    for j in 0..ny {
        for i in 0..nx {
            let idx = nx * j + i;
            ll_k[idx] = scale * klx[i] * kly[j];
            lh_k[idx] = scale * klx[i] * khy[j];
            hl_k[idx] = scale * khx[i] * kly[j];
            hh_k[idx] = scale * khx[i] * khy[j];
        }
    }

    [ll_k, lh_k, hl_k, hh_k]
}




