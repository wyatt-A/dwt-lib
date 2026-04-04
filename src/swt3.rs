use std::cell::RefCell;
use std::ops::Range;
use dft_lib::common::{FftDirection, NormalizationType};
use dft_lib::fftw_fft::{fftw_fftn, fftw_fftn_batched};
use num_complex::Complex32;
use crate::swt2::SWT2Plan;
use crate::swt::{prep_kernel, soft_threshold};
use crate::wavelet::{Wavelet, WaveletFilter};

#[cfg(test)]
mod tests {
    use std::alloc::alloc;
    use array_lib::ArrayDim;
    use array_lib::io_cfl::read_cfl;
    use array_lib::io_nifti::write_nifti;
    use num_complex::Complex32;
    use crate::swt3::SWT3Plan;
    use crate::wavelet::{Wavelet, WaveletType};

    #[test]
    fn test() {

        // build a test array
        let x_dims = ArrayDim::from_shape(&[10,25,7]);
        let mut x = x_dims.alloc(Complex32::ZERO);
        x.iter_mut().enumerate().for_each(|(e, x)| {
            let [i,j,k,..] = x_dims.calc_idx(e);
            *x = Complex32::new(i as f32 + j as f32, k as f32);
        });

        let w = SWT3Plan::new(x_dims.shape_ns(),2,Wavelet::new(WaveletType::Daubechies2));

        let t_dims = ArrayDim::from_shape(&w.t_domain_shape());
        let mut t = t_dims.alloc(Complex32::ZERO);

        w.decompose(&x, &mut t);

        let xe = x.iter().map(|x|x.norm_sqr() as f64).sum::<f64>();
        let te = t.iter().map(|x|x.norm_sqr() as f64).sum::<f64>();

        println!("x-energy: {}",xe);
        println!("t-energy: {}",te);

        w.reconstruct(t.as_mut_slice(), x.as_mut_slice());

        write_nifti("out.nii",&t.iter().map(|x|x.norm()).collect::<Vec<_>>(),t_dims);

        println!("x-energy: {}",xe);
    }




}

pub struct SWT3Plan {
    /// dimensions of signal (image)
    dims: [usize; 3],

    /// number of decomp/recon levels
    levels: usize,

    // ======================
    // Decomposition kernels
    // ======================

    /// LLL
    d_lll: Vec<Vec<Complex32>>,
    /// LLH
    d_llh: Vec<Vec<Complex32>>,
    /// LHL
    d_lhl: Vec<Vec<Complex32>>,
    /// LHH
    d_lhh: Vec<Vec<Complex32>>,
    /// HLL
    d_hll: Vec<Vec<Complex32>>,
    /// HLH
    d_hlh: Vec<Vec<Complex32>>,
    /// HHL
    d_hhl: Vec<Vec<Complex32>>,
    /// HHH
    d_hhh: Vec<Vec<Complex32>>,

    // ======================
    // Reconstruction kernels
    // ======================

    /// LLL
    r_lll: Vec<Vec<Complex32>>,
    /// LLH
    r_llh: Vec<Vec<Complex32>>,
    /// LHL
    r_lhl: Vec<Vec<Complex32>>,
    /// LHH
    r_lhh: Vec<Vec<Complex32>>,
    /// HLL
    r_hll: Vec<Vec<Complex32>>,
    /// HLH
    r_hlh: Vec<Vec<Complex32>>,
    /// HHL
    r_hhl: Vec<Vec<Complex32>>,
    /// HHH
    r_hhh: Vec<Vec<Complex32>>,

    /// composite transfer function per level
    h: Vec<Vec<Complex32>>,

    /// wavelet for analysis
    w: Wavelet<f32>,

    /// temp buffers for calculations
    tmp_x: RefCell<Vec<Complex32>>, // size = nx * ny * nz
    tmp_y: RefCell<Vec<Complex32>>, // size = nx * ny * nz * 8
}

impl SWT3Plan {
    pub fn new(size:&[usize], levels: usize, w: Wavelet<f32>) -> SWT3Plan {

        assert!(size.len() > 2);

        let nx = size[0];
        let ny = size[1];
        let nz = size[2];

        // fourier kernels
        let mut d_lll = vec![];
        let mut d_llh = vec![];
        let mut d_lhl = vec![];
        let mut d_lhh = vec![];
        let mut d_hll = vec![];
        let mut d_hlh = vec![];
        let mut d_hhl = vec![];
        let mut d_hhh = vec![];

        let mut r_lll = vec![];
        let mut r_llh = vec![];
        let mut r_lhl = vec![];
        let mut r_lhh = vec![];
        let mut r_hll = vec![];
        let mut r_hlh = vec![];
        let mut r_hhl = vec![];
        let mut r_hhh = vec![];

        let mut h = vec![];

        // calculate kernels
        for level in 1..=levels {
            let [
            d_lll_k, d_llh_k, d_lhl_k, d_lhh_k,
            d_hll_k, d_hlh_k, d_hhl_k, d_hhh_k
            ] = calc_kernel_3d(nx, ny, nz, w.lo_d(), w.hi_d(), level);

            let [
            r_lll_k, r_llh_k, r_lhl_k, r_lhh_k,
            r_hll_k, r_hlh_k, r_hhl_k, r_hhh_k
            ] = calc_kernel_3d(nx, ny, nz, w.lo_r(), w.hi_r(), level);

            let tf = calc_transfer_fn_3d([
                &d_lll_k, &d_llh_k, &d_lhl_k, &d_lhh_k,
                &d_hll_k, &d_hlh_k, &d_hhl_k, &d_hhh_k,
                &r_lll_k, &r_llh_k, &r_lhl_k, &r_lhh_k,
                &r_hll_k, &r_hlh_k, &r_hhl_k, &r_hhh_k,
            ]);

            h.push(tf);

            d_lll.push(d_lll_k);
            d_llh.push(d_llh_k);
            d_lhl.push(d_lhl_k);
            d_lhh.push(d_lhh_k);
            d_hll.push(d_hll_k);
            d_hlh.push(d_hlh_k);
            d_hhl.push(d_hhl_k);
            d_hhh.push(d_hhh_k);

            r_lll.push(r_lll_k);
            r_llh.push(r_llh_k);
            r_lhl.push(r_lhl_k);
            r_lhh.push(r_lhh_k);
            r_hll.push(r_hll_k);
            r_hlh.push(r_hlh_k);
            r_hhl.push(r_hhl_k);
            r_hhh.push(r_hhh_k);
        }

        let n = nx * ny * nz;

        // temp buffer to avoid re-allocations
        let tmp_x = RefCell::new(vec![Complex32::ZERO; n]);
        let tmp_y = RefCell::new(vec![Complex32::ZERO; n * 8]);

        SWT3Plan {
            dims: [nx, ny, nz],
            levels,

            d_lll,
            d_llh,
            d_lhl,
            d_lhh,
            d_hll,
            d_hlh,
            d_hhl,
            d_hhh,

            r_lll,
            r_llh,
            r_lhl,
            r_lhh,
            r_hll,
            r_hlh,
            r_hhl,
            r_hhh,

            h,
            w,
            tmp_x,
            tmp_y,
        }
    }

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

        // scratch for one level input to recon_level:
        // [LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH]
        let mut bands = vec![Complex32::ZERO; 8 * n];

        // scratch for reconstructed image at each stage
        let mut tmp = vec![Complex32::ZERO; n];

        for level in (0..self.levels).rev() {
            // fill [LLL, details...]
            bands[..n].copy_from_slice(&approx);

            let dr = self.detail_range(level);
            bands[n..8 * n].copy_from_slice(&src[dr]);

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

        // scratch for one level:
        // [LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH]
        let mut bands = vec![Complex32::ZERO; 8 * n];

        for level in 0..self.levels {
            self.decomp_level(level, &approx, &mut bands);

            // store the 7 detail bands for this level
            let dr = self.detail_range(level);
            dst[dr].copy_from_slice(&bands[n..8 * n]);

            // propagate LLL to next level
            approx.copy_from_slice(&bands[..n]);
        }

        // store final approximation LLL_J
        let ar = self.approx_range();
        dst[ar].copy_from_slice(&approx);
    }

    pub fn decomp_level(&self, level: usize, src: &[Complex32], dst: &mut [Complex32]) {
        let n = self.subband_size();

        assert_eq!(src.len(), n);
        assert_eq!(dst.len(), 8 * n);

        let mut tmp = self.tmp_x.borrow_mut();
        let x = tmp.as_mut_slice();
        x.copy_from_slice(src);


        fftw_fftn(x, &self.dims, FftDirection::Forward, NormalizationType::Unitary);

        let kernels = [
            self.d_lll[level].as_slice(),
            self.d_llh[level].as_slice(),
            self.d_lhl[level].as_slice(),
            self.d_lhh[level].as_slice(),
            self.d_hll[level].as_slice(),
            self.d_hlh[level].as_slice(),
            self.d_hhl[level].as_slice(),
            self.d_hhh[level].as_slice(),
        ];

        for (i,(band, kern)) in dst.chunks_exact_mut(n).zip(kernels).enumerate() {
            println!("decomposing band {}",i+1);
            for ((b, &k), &xf) in band.iter_mut().zip(kern.iter()).zip(x.iter()) {
                *b = xf * k;
            }

            fftw_fftn(band, &self.dims, FftDirection::Inverse, NormalizationType::Unitary);
        }
    }

    pub fn recon_level(&self, level: usize, src: &[Complex32], dst: &mut [Complex32]) {
        const EPS: f32 = 1e-6;

        let n = self.subband_size();

        assert_eq!(dst.len(), n);
        assert_eq!(src.len(), 8 * n);

        let mut tmp = self.tmp_y.borrow_mut();
        let y = tmp.as_mut_slice();
        y.copy_from_slice(src);

        fftw_fftn_batched(
            y,
            &self.dims,
            8,
            FftDirection::Forward,
            NormalizationType::Unitary,
        );

        let kernels = [
            self.r_lll[level].as_slice(),
            self.r_llh[level].as_slice(),
            self.r_lhl[level].as_slice(),
            self.r_lhh[level].as_slice(),
            self.r_hll[level].as_slice(),
            self.r_hlh[level].as_slice(),
            self.r_hhl[level].as_slice(),
            self.r_hhh[level].as_slice(),
        ];

        let h = self.h[level].as_slice();

        for i in 0..n {
            let mut sum = Complex32::ZERO;
            for b in 0..8 {
                sum += kernels[b][i] * y[b * n + i];
            }

            let denom = if h[i].norm() < EPS {
                Complex32::new(EPS, 0.0)
            } else {
                h[i]
            };

            dst[i] = sum / denom;
        }

        fftw_fftn(dst, &self.dims, FftDirection::Inverse, NormalizationType::Unitary);
    }

    pub fn subband_size(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    pub fn approx_range(&self) -> Range<usize> {
        0..self.subband_size()
    }

    /// Range for the 7 detail bands
    /// (LLH, LHL, LHH, HLL, HLH, HHL, HHH)
    /// for a given level.
    /// level = 0 is finest, level = self.levels - 1 is coarsest.
    pub fn detail_range(&self, level: usize) -> Range<usize> {
        assert!(level < self.levels);
        let n = self.subband_size();
        let block = self.levels - 1 - level;
        let start = n + block * 7 * n;
        let stop = start + 7 * n;
        start..stop
    }

    /// returns the size of the transform domain.
    /// This is 7 * n_levels + 1 subbands
    pub fn t_domain_size(&self) -> usize {
        self.subband_size() * self.t_bands()
    }

    /// returns the shape of the transform domain
    pub fn t_domain_shape(&self) -> [usize;4] {
        [self.dims[0],self.dims[1],self.dims[2],self.t_bands()]
    }

    /// returns the number of subbands for the decomposition
    pub fn t_bands(&self) -> usize {
        7 * self.levels + 1
    }
}

fn calc_kernel_3d(
    nx: usize,
    ny: usize,
    nz: usize,
    lo: &[f32],
    hi: &[f32],
    level: usize,
) -> [Vec<Complex32>; 8] {
    let klx = prep_kernel(lo, level, nx);
    let khx = prep_kernel(hi, level, nx);

    let kly = prep_kernel(lo, level, ny);
    let khy = prep_kernel(hi, level, ny);

    let klz = prep_kernel(lo, level, nz);
    let khz = prep_kernel(hi, level, nz);

    let n = nx * ny * nz;

    let mut lll_k = vec![Complex32::ZERO; n];
    let mut llh_k = vec![Complex32::ZERO; n];
    let mut lhl_k = vec![Complex32::ZERO; n];
    let mut lhh_k = vec![Complex32::ZERO; n];
    let mut hll_k = vec![Complex32::ZERO; n];
    let mut hlh_k = vec![Complex32::ZERO; n];
    let mut hhl_k = vec![Complex32::ZERO; n];
    let mut hhh_k = vec![Complex32::ZERO; n];

    // 2-D used sqrt(nx * ny) / 2
    // 3-D analogue: sqrt(nx * ny * nz) / sqrt(8)
    let scale = (n as f32).sqrt() / (8f32).sqrt();

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = nx * ny * k + nx * j + i;

                let lx = klx[i];
                let hx = khx[i];
                let ly = kly[j];
                let hy = khy[j];
                let lz = klz[k];
                let hz = khz[k];

                lll_k[idx] = scale * lx * ly * lz;
                llh_k[idx] = scale * lx * ly * hz;
                lhl_k[idx] = scale * lx * hy * lz;
                lhh_k[idx] = scale * lx * hy * hz;
                hll_k[idx] = scale * hx * ly * lz;
                hlh_k[idx] = scale * hx * ly * hz;
                hhl_k[idx] = scale * hx * hy * lz;
                hhh_k[idx] = scale * hx * hy * hz;
            }
        }
    }

    [lll_k, llh_k, lhl_k, lhh_k, hll_k, hlh_k, hhl_k, hhh_k]
}

fn calc_transfer_fn_3d(kernels: [&Vec<Complex32>; 16]) -> Vec<Complex32> {
    let n = kernels[0].len();
    let mut h = vec![Complex32::ZERO; n];

    for i in 0..n {
        h[i] =
            kernels[0][i] * kernels[8][i] +   // LLL
                kernels[1][i] * kernels[9][i] +   // LLH
                kernels[2][i] * kernels[10][i] +  // LHL
                kernels[3][i] * kernels[11][i] +  // LHH
                kernels[4][i] * kernels[12][i] +  // HLL
                kernels[5][i] * kernels[13][i] +  // HLH
                kernels[6][i] * kernels[14][i] +  // HHL
                kernels[7][i] * kernels[15][i];   // HHH
    }

    h
}