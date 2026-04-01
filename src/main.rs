use array_lib::io_cfl::read_cfl;
use array_lib::io_nifti::write_nifti;
use array_lib::ArrayDim;
use dwt_lib::swt::SWT2Planner;
use dwt_lib::wavelet::{Wavelet, WaveletType};
use num_complex::Complex32;

fn main() {

    // SWT thresholding

    let (slice, slice_dims) = read_cfl("slice");

    let slice_orig = slice.iter().map(|x| x.norm()).collect::<Vec<_>>();
    write_nifti("slice_orig", &slice_orig, slice_dims);

    let shape = slice_dims.shape_squeeze();
    let nx = shape[0];
    let ny = shape[1];

    let swt = SWT2Planner::new(shape[0], shape[1], Wavelet::new(WaveletType::Daubechies2), 1);
    let mut t_dom = swt.alloc_t_domain();
    let t_dims = ArrayDim::from_shape(&[nx, ny, swt.n_bands()]);

    swt.forward(&slice, &mut t_dom);

    let t = t_dom.iter().map(|x| x.norm()).collect::<Vec<_>>();
    write_nifti("dom", &t, t_dims);

    // //** soft threshold t_dom with some lambda
    // swt_soft_threshold(&swt, &mut t_dom, 0.000);
    //
    // let mut slice_out = slice.to_vec();
    // swt.inverse(&t_dom, &mut slice_out);
    //
    // let slice_out = slice_out.iter().map(|x| x.norm()).collect::<Vec<_>>();

    // write_nifti("slice", &slice_out, slice_dims);
    // write_nifti("slice_orig", &slice_orig, slice_dims);
    // write_nifti("slice_t", &t, ArrayDim::from_shape(&[nx, ny, swt.n_bands()]));
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

fn swt_soft_threshold(planner: &SWT2Planner, coeffs: &mut [Complex32], lambda: f32) {
    let subband_size = planner.subband_size();

    // skip LL
    for level in 0..planner.levels() {
        for subband_idx in 1..=3 {
            let addr = planner.calc_address(subband_idx, level);
            let band = &mut coeffs[addr..addr + subband_size];

            for z in band.iter_mut() {
                *z = soft_thresh_complex(*z, lambda);
            }
        }
    }
}