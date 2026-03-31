use crate::wavelet::{Wavelet, WaveletFilter};
use array_lib::ArrayDim;
use dft_lib::common::{FftDirection, NormalizationType};
use dft_lib::fftw_fft::fftw_fftn;
use num_complex::Complex32;
use num_traits::Zero;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wavelet;
    use crate::wavelet::{WaveletFilter, WaveletType};
    use array_lib::ArrayDim;
    use dft_lib::common::{FftDirection, NormalizationType};
    use dft_lib::fftw_fft::fftw_fftn;
    use num_complex::Complex32;
    use std::time::Instant;

    #[test]
    fn dilate() {
        let f = [1, 2, 3, 4];
        let d = dilate_filter(2, &f);
        assert_eq!(&[1, 0, 2, 0, 3, 0, 4, 0], d.as_slice());
    }

    #[test]
    fn swt2_planner() {
        let w = Wavelet::<f32>::new(WaveletType::Daubechies2);

        let m = 600;
        let n = 600;

        let swt = SWT2Planner::new_max_levels(m, n, w);
        println!("running {} levels ...", swt.levels());
        let src = (0..(m * n)).map(|x| Complex32::new(x as f32, 0.0)).collect::<Vec<_>>();
        let mut dst = swt.alloc_t_domain();

        let now = Instant::now();
        swt.forward(&src, &mut dst);
        let src2 = dst;
        let mut dst = vec![Complex32::new(0.0, 0.0); m * n];
        swt.inverse(&src2, &mut dst);
        let dur = now.elapsed();
        println!("took {} ms", dur.as_millis());

        let err = dst.iter().zip(src.iter()).map(|(x, y)| (x - y).norm_sqr()).collect::<Vec<_>>();

        let max_err = err.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        println!("max error: {}", max_err);

        let max_abs_err = dst.iter()
            .zip(src.iter())
            .map(|(x, y)| (*x - *y).norm())
            .fold(0.0f32, f32::max);

        let rmse = (
            dst.iter()
                .zip(src.iter())
                .map(|(x, y)| (*x - *y).norm_sqr())
                .sum::<f32>()
                / src.len() as f32
        ).sqrt();

        let max_src = src.iter()
            .map(|x| x.norm())
            .fold(0.0f32, f32::max);

        println!("max abs err: {}", max_abs_err);
        println!("rmse: {}", rmse);
        println!("rel max err: {}", max_abs_err / max_src);
        println!("rel rmse: {}", rmse / max_src);
    }

    #[test]
    fn forward_inv() {
        let x: Vec<_> = (0..512).map(|x| Complex32::new(x as f32, 0.)).collect();
        let (lo, hi) = swt1_forward(&x);
        let r = swt1_inverse(&lo, &hi);
        println!("{:?}", r);
    }

    fn swt1_inverse(lo: &[Complex32], hi: &[Complex32]) -> Vec<Complex32> {
        assert_eq!(lo.len(), hi.len());

        let level = 1;
        let w = wavelet::Wavelet::<f32>::new(WaveletType::Daubechies2);

        // Use the SAME analysis filters as forward
        let hi_d = w.hi_d();
        let lo_d = w.lo_d();

        let d_hi_d = dilate_filter(level, hi_d);
        let d_lo_d = dilate_filter(level, lo_d);

        let n = lo.len();

        let mut k_hi = vec![0.0f32; n];
        let mut k_lo = vec![0.0f32; n];

        k_hi[0..d_hi_d.len()].copy_from_slice(&d_hi_d);
        k_lo[0..d_lo_d.len()].copy_from_slice(&d_lo_d);

        let k_dim = ArrayDim::from_shape(&[n]);

        let mut k_hi_shift = vec![0.0f32; n];
        let mut k_lo_shift = vec![0.0f32; n];

        let inv = false;
        k_dim.fftshift(&k_hi, &mut k_hi_shift, inv);
        k_dim.fftshift(&k_lo, &mut k_lo_shift, inv);

        let mut h: Vec<Complex32> =
            k_hi_shift.into_iter().map(|x| Complex32::new(x, 0.0)).collect();
        let mut g: Vec<Complex32> =
            k_lo_shift.into_iter().map(|x| Complex32::new(x, 0.0)).collect();

        // FFT of the analysis kernels
        fftw_fftn(&mut h, &[n], FftDirection::Forward, NormalizationType::Unitary);
        fftw_fftn(&mut g, &[n], FftDirection::Forward, NormalizationType::Unitary);

        // FFT of coefficients
        let mut lo_f = lo.to_vec();
        let mut hi_f = hi.to_vec();

        fftw_fftn(&mut lo_f, &[n], FftDirection::Forward, NormalizationType::Unitary);
        fftw_fftn(&mut hi_f, &[n], FftDirection::Forward, NormalizationType::Unitary);

        // Left inverse:
        // X = (conj(G)*Lo + conj(H)*Hi) / (|G|^2 + |H|^2)
        let eps = 1e-6f32;
        let mut x_f = vec![Complex32::new(0.0, 0.0); n];

        for i in 0..n {
            let denom = g[i].norm_sqr() + h[i].norm_sqr();
            let denom = if denom < eps { eps } else { denom };

            x_f[i] = (g[i].conj() * lo_f[i] + h[i].conj() * hi_f[i]) / denom;
        }

        fftw_fftn(&mut x_f, &[n], FftDirection::Inverse, NormalizationType::Unitary);

        x_f
    }


    fn swt1_forward(x: &[Complex32]) -> (Vec<Complex32>, Vec<Complex32>) {
        let mut x = x.to_vec();

        let level = 1;
        let w = wavelet::Wavelet::<f32>::new(WaveletType::Daubechies2);
        let hi_d = w.hi_d();
        let lo_d = w.lo_d();

        let d_hi_d = dilate_filter(level, hi_d);
        let d_lo_d = dilate_filter(level, lo_d);

        let mut k_hi = vec![0.; x.len()];
        let mut k_lo = k_hi.clone();

        let k_dim = ArrayDim::from_shape(&[x.len()]);

        k_hi[0..d_hi_d.len()].copy_from_slice(&d_hi_d);
        k_lo[0..d_lo_d.len()].copy_from_slice(&d_lo_d);

        let n = x.len();

        let mut k_hi_shift = vec![0.; n];
        let mut k_lo_shift = vec![0.; n];

        let inv = false;
        k_dim.fftshift(&k_hi, &mut k_hi_shift, inv);
        k_dim.fftshift(&k_lo, &mut k_lo_shift, inv);

        let mut k_hi: Vec<_> = k_hi_shift.into_iter().map(|x| Complex32::new(x, 0.)).collect();
        let mut k_lo: Vec<_> = k_lo_shift.into_iter().map(|x| Complex32::new(x, 0.)).collect();

        fftw_fftn(&mut k_hi, &[n], FftDirection::Forward, NormalizationType::Unitary);
        fftw_fftn(&mut k_lo, &[n], FftDirection::Forward, NormalizationType::Unitary);

        fftw_fftn(&mut x, &[n], FftDirection::Forward, NormalizationType::Unitary);

        k_hi.iter_mut().zip(&x).for_each(|(a, b)| *a *= *b);
        k_lo.iter_mut().zip(&x).for_each(|(a, b)| *a *= *b);

        fftw_fftn(&mut k_hi, &[n], FftDirection::Inverse, NormalizationType::Unitary);
        fftw_fftn(&mut k_lo, &[n], FftDirection::Inverse, NormalizationType::Unitary);

        let lo = k_lo;
        let hi = k_hi;

        (lo, hi)
    }
}

#[derive(Debug)]
/// stationary wavelet transform planner
pub struct SWT2Planner {
    /// size of each subband image
    image_size: [usize; 2],
    /// number of levels to calculate
    n_levels: usize,
    /// wavelet to perform analysis
    wavelet: Wavelet<f32>,
    /// size of the transform domain
    t_domain_size: usize,
}

impl SWT2Planner {
    /// returns the number of decomp levels for this planner
    pub fn levels(&self) -> usize {
        self.n_levels
    }

    pub fn n_bands(&self) -> usize {
        (2usize.pow(2) - 1) * self.n_levels + 1
    }
    /// create a new planner with specified number of levels
    pub fn new(m: usize, n: usize, w: Wavelet<f32>, n_levels: usize) -> SWT2Planner {
        let t_domain_size = Self::t_domain_size(m, n, n_levels);
        SWT2Planner { image_size: [m, n], n_levels, wavelet: w, t_domain_size }
    }

    /// create a new planner with max number of levels
    pub fn new_max_levels(m: usize, n: usize, w: Wavelet<f32>) -> SWT2Planner {
        let n_levels = Self::n_levels(m, n, &w);
        let t_domain_size = Self::t_domain_size(m, n, n_levels);
        SWT2Planner { image_size: [m, n], n_levels, wavelet: w, t_domain_size }
    }

    pub fn forward(&self, src: &[Complex32], dst: &mut [Complex32]) {
        let mut current = src.to_vec();
        let mut next_ll = vec![Complex32::new(0.0, 0.0); self.subband_size()];

        for level in 0..self.n_levels {
            self.forward_level(level, &current, dst);

            // grab the newly computed LL for the next stage
            next_ll.copy_from_slice(&dst[0..self.subband_size()]);
            std::mem::swap(&mut current, &mut next_ll);
        }
    }

    pub fn inverse(&self, src: &[Complex32], dst: &mut [Complex32]) {
        let mut coeffs = src.to_vec();
        let mut tmp = vec![Complex32::new(0.0, 0.0); self.subband_size()];

        for level in (0..self.n_levels).rev() {
            self.inverse_level(level, &coeffs, &mut tmp);

            // put reconstructed previous-level LL back into the packed buffer
            coeffs[0..self.subband_size()].copy_from_slice(&tmp);
        }

        dst.copy_from_slice(&coeffs[0..self.subband_size()]);
    }


    fn build_fourier_kernels(
        w: &Wavelet<f32>,
        level: usize,
        image_size: [usize; 2],
    ) -> [Vec<Complex32>; 4] {
        let hi_d = w.hi_d();
        let lo_d = w.lo_d();

        let d_hi_d = dilate_filter(level + 1, hi_d);
        let d_lo_d = dilate_filter(level + 1, lo_d);

        let nx = image_size[0];
        let ny = image_size[1];

        // Embed the dilated filters so their center is at index 0
        // in the circular FFT domain.
        let mut kx_hi: Vec<Complex32> = embed_filter_centered(&d_hi_d, nx)
            .into_iter()
            .map(|v| Complex32::new(v, 0.0))
            .collect();

        let mut kx_lo: Vec<Complex32> = embed_filter_centered(&d_lo_d, nx)
            .into_iter()
            .map(|v| Complex32::new(v, 0.0))
            .collect();

        let mut ky_hi: Vec<Complex32> = embed_filter_centered(&d_hi_d, ny)
            .into_iter()
            .map(|v| Complex32::new(v, 0.0))
            .collect();

        let mut ky_lo: Vec<Complex32> = embed_filter_centered(&d_lo_d, ny)
            .into_iter()
            .map(|v| Complex32::new(v, 0.0))
            .collect();

        // FFT the 1-D kernels
        fftw_fftn(&mut kx_hi, &[nx], FftDirection::Forward, NormalizationType::Unitary);
        fftw_fftn(&mut kx_lo, &[nx], FftDirection::Forward, NormalizationType::Unitary);
        fftw_fftn(&mut ky_hi, &[ny], FftDirection::Forward, NormalizationType::Unitary);
        fftw_fftn(&mut ky_lo, &[ny], FftDirection::Forward, NormalizationType::Unitary);

        [kx_hi, kx_lo, ky_hi, ky_lo]
    }

    // fn build_fourier_kernels(w: &Wavelet<f32>, level: usize, image_size: [usize; 2]) -> [Vec<Complex32>; 4] {
    //     let hi_d = w.hi_d();
    //     let lo_d = w.lo_d();
    //     let d_hi_d = dilate_filter(level + 1, hi_d);
    //     let d_lo_d = dilate_filter(level + 1, lo_d);
    //
    //     let nx = image_size[0];
    //     let ny = image_size[1];
    //
    //     // padded kernels for x dimension
    //     let mut kx_hi = vec![0.0f32; nx];
    //     let mut kx_lo = vec![0.0f32; nx];
    //     kx_hi[0..d_hi_d.len()].copy_from_slice(&d_hi_d);
    //     kx_lo[0..d_lo_d.len()].copy_from_slice(&d_lo_d);
    //
    //     // padded kernels for y dimension
    //     let mut ky_hi = vec![0.0f32; ny];
    //     let mut ky_lo = vec![0.0f32; ny];
    //     ky_hi[0..d_hi_d.len()].copy_from_slice(&d_hi_d);
    //     ky_lo[0..d_lo_d.len()].copy_from_slice(&d_lo_d);
    //
    //     // array dims for kernels
    //     let kx_dim = ArrayDim::from_shape(&[nx]);
    //     let ky_dim = ArrayDim::from_shape(&[ny]);
    //
    //     // // buffers for shifted kernels
    //     // let mut kx_hi_shift = vec![0.0f32; nx];
    //     // let mut kx_lo_shift = vec![0.0f32; nx];
    //     // let mut ky_hi_shift = vec![0.0f32; ny];
    //     // let mut ky_lo_shift = vec![0.0f32; ny];
    //     //
    //     // // shift kernels for fft
    //     // let inv = true;
    //     // kx_dim.fftshift(&kx_hi, &mut kx_hi_shift, inv);
    //     // kx_dim.fftshift(&kx_lo, &mut kx_lo_shift, inv);
    //     // ky_dim.fftshift(&ky_hi, &mut ky_hi_shift, inv);
    //     // ky_dim.fftshift(&ky_lo, &mut ky_lo_shift, inv);
    //
    //     // build the complex kernels for fft
    //     let mut kx_hi: Vec<Complex32> = kx_hi.into_iter().map(|v| Complex32::new(v, 0.0)).collect();
    //     let mut kx_lo: Vec<Complex32> = kx_lo.into_iter().map(|v| Complex32::new(v, 0.0)).collect();
    //     let mut ky_hi: Vec<Complex32> = ky_hi.into_iter().map(|v| Complex32::new(v, 0.0)).collect();
    //     let mut ky_lo: Vec<Complex32> = ky_lo.into_iter().map(|v| Complex32::new(v, 0.0)).collect();
    //
    //     // FFT the 1-D kernels
    //     fftw_fftn(&mut kx_hi, &[nx], FftDirection::Forward, NormalizationType::Unitary);
    //     fftw_fftn(&mut kx_lo, &[nx], FftDirection::Forward, NormalizationType::Unitary);
    //     fftw_fftn(&mut ky_hi, &[ny], FftDirection::Forward, NormalizationType::Unitary);
    //     fftw_fftn(&mut ky_lo, &[ny], FftDirection::Forward, NormalizationType::Unitary);
    //
    //     [kx_hi, kx_lo, ky_hi, ky_lo]
    // }

    /// computes a single level of the SWT, starting at level 0. src is the size of the image, and
    /// dst should be large enough to store all subbands at all levels
    pub fn forward_level(&self, level: usize, src: &[Complex32], dst: &mut [Complex32]) {
        let nx = self.image_size[0];
        let ny = self.image_size[1];

        let [kx_hi, kx_lo, ky_hi, ky_lo] = Self::build_fourier_kernels(&self.wavelet, level, [nx, ny]);

        // fft of band
        let mut x_f = src.to_vec();
        fftw_fftn(&mut x_f, &[nx, ny], FftDirection::Forward, NormalizationType::Unitary);

        // Build 2-D outputs in Fourier space
        let mut ll_f = vec![Complex32::new(0.0, 0.0); x_f.len()];
        let mut lh_f = vec![Complex32::new(0.0, 0.0); x_f.len()];
        let mut hl_f = vec![Complex32::new(0.0, 0.0); x_f.len()];
        let mut hh_f = vec![Complex32::new(0.0, 0.0); x_f.len()];

        for yi in 0..ny {
            for xi in 0..nx {
                let idx = yi * nx + xi;

                let lo_x = kx_lo[xi];
                let hi_x = kx_hi[xi];
                let lo_y = ky_lo[yi];
                let hi_y = ky_hi[yi];

                let xv = x_f[idx];

                ll_f[idx] = lo_y * lo_x * xv;
                lh_f[idx] = hi_y * lo_x * xv;
                hl_f[idx] = lo_y * hi_x * xv;
                hh_f[idx] = hi_y * hi_x * xv;
            }
        }

        // transform each band back to image space
        fftw_fftn(&mut ll_f, &[nx, ny], FftDirection::Inverse, NormalizationType::Unitary);
        fftw_fftn(&mut lh_f, &[nx, ny], FftDirection::Inverse, NormalizationType::Unitary);
        fftw_fftn(&mut hl_f, &[nx, ny], FftDirection::Inverse, NormalizationType::Unitary);
        fftw_fftn(&mut hh_f, &[nx, ny], FftDirection::Inverse, NormalizationType::Unitary);

        // pack each band into the correct slot in the dst buffer
        // LL goes first
        dst[0..self.subband_size()].copy_from_slice(&ll_f);

        let addr = self.calc_address(1, level);
        dst[addr..addr + self.subband_size()].copy_from_slice(&lh_f);

        let addr = self.calc_address(2, level);
        dst[addr..addr + self.subband_size()].copy_from_slice(&hl_f);

        let addr = self.calc_address(3, level);
        dst[addr..addr + self.subband_size()].copy_from_slice(&hh_f);
    }

    /// computes a single level of the inverse SWT, starting at level 0. src is the size of all subbands
    /// for all levels, and dst is the original image size
    pub fn inverse_level(&self, level: usize, src: &[Complex32], dst: &mut [Complex32]) {
        let nx = self.image_size[0];
        let ny = self.image_size[1];
        let n = nx * ny;

        let [kx_hi, kx_lo, ky_hi, ky_lo] =
            Self::build_fourier_kernels(&self.wavelet, level, [nx, ny]);

        // read packed subbands
        let mut ll = src[0..self.subband_size()].to_vec();

        let addr = self.calc_address(1, level);
        let mut lh = src[addr..addr + self.subband_size()].to_vec();

        let addr = self.calc_address(2, level);
        let mut hl = src[addr..addr + self.subband_size()].to_vec();

        let addr = self.calc_address(3, level);
        let mut hh = src[addr..addr + self.subband_size()].to_vec();

        // FFT all subbands
        fftw_fftn(&mut ll, &[nx, ny], FftDirection::Forward, NormalizationType::Unitary);
        fftw_fftn(&mut lh, &[nx, ny], FftDirection::Forward, NormalizationType::Unitary);
        fftw_fftn(&mut hl, &[nx, ny], FftDirection::Forward, NormalizationType::Unitary);
        fftw_fftn(&mut hh, &[nx, ny], FftDirection::Forward, NormalizationType::Unitary);

        let eps = 1e-6f32;
        let mut x_f = vec![Complex32::new(0.0, 0.0); n];

        for yi in 0..ny {
            for xi in 0..nx {
                let idx = yi * nx + xi;

                let lo_x = kx_lo[xi];
                let hi_x = kx_hi[xi];
                let lo_y = ky_lo[yi];
                let hi_y = ky_hi[yi];

                let k_ll = lo_y * lo_x;
                let k_lh = hi_y * lo_x;
                let k_hl = lo_y * hi_x;
                let k_hh = hi_y * hi_x;

                let denom = k_ll.norm_sqr()
                    + k_lh.norm_sqr()
                    + k_hl.norm_sqr()
                    + k_hh.norm_sqr();

                let denom = if denom < eps { eps } else { denom };

                x_f[idx] = (
                    k_ll.conj() * ll[idx]
                        + k_lh.conj() * lh[idx]
                        + k_hl.conj() * hl[idx]
                        + k_hh.conj() * hh[idx]
                ) / denom;
            }
        }

        fftw_fftn(&mut x_f, &[nx, ny], FftDirection::Inverse, NormalizationType::Unitary);

        // write reconstructed approximation back into the LL slot
        dst.copy_from_slice(&x_f);
    }


    /// Calculate the address of the first element of a subband.
    /// Subband indices:
    /// 0 = LL
    /// 1 = LH
    /// 2 = HL
    /// 3 = HH
    ///
    /// level_idx is zero-based:
    /// 0 = level 1
    /// 1 = level 2
    /// ...
    pub fn calc_address(&self, subband_idx: usize, level_idx: usize) -> usize {
        assert!(subband_idx < 4, "subband index must be less than 4 (LL, LH, HL, HH)");
        if subband_idx == 0 {
            return 0;
        }
        (1 + 3 * level_idx + (subband_idx - 1)) * self.subband_size()
    }

    pub fn subband_size(&self) -> usize {
        self.image_size[0] * self.image_size[1]
    }

    pub fn alloc_t_domain(&self) -> Vec<Complex32> {
        vec![Complex32::ZERO; self.t_domain_size]
    }

    /// returns the recommended number of levels for the image size and wavelet
    fn n_levels(m: usize, n: usize, w: &Wavelet<f32>) -> usize {
        let s = m.min(n);
        let l = w.filt_len();
        (s as f64 / (l as f64 - 1.)).log2() as usize
    }

    fn t_domain_size(m: usize, n: usize, n_levels: usize) -> usize {
        let d = 2;
        let f = (2usize.pow(d) - 1) * n_levels + 1;
        m * n * f
    }
}


// /// returns a dilated filter based on level. This inserts 0s between filter entries, returning
// /// the dilated filter
// fn dilate_filter<T: Sized + Copy + Zero>(level: usize, filter: &[T]) -> Vec<T> {
//     assert_ne!(level, 0, "level must be greater than 0");
//     let d = 2i32.pow(level as u32 - 1) as usize;
//     let n = filter.len() * d;
//     let mut dilated = vec![T::zero(); n];
//     dilated.chunks_exact_mut(d).zip(filter.iter()).for_each(|(chunk, c)| {
//         chunk[0] = *c;
//     });
//     dilated
// }

fn dilate_filter<T: Copy + Zero>(level: usize, filter: &[T]) -> Vec<T> {
    assert!(level >= 1, "level must be greater than 0");
    assert!(!filter.is_empty(), "filter must not be empty");

    let d = 1usize << (level - 1);
    let n = (filter.len() - 1) * d + 1;

    let mut dilated = vec![T::zero(); n];

    for (i, &c) in filter.iter().enumerate() {
        dilated[i * d] = c;
    }

    dilated
}

fn embed_filter_centered(filter: &[f32], n: usize) -> Vec<f32> {
    assert!(
        filter.len() <= n,
        "dilated filter length {} exceeds signal length {}",
        filter.len(),
        n
    );

    let mut out = vec![0.0f32; n];

    // Start with the usual FIR "center" choice.
    // For even-length wavelet filters, this is the first thing to try.
    let center = filter.len() / 2;

    for (i, &v) in filter.iter().enumerate() {
        let idx = (i + n - center) % n;
        out[idx] = v;
    }

    out
}