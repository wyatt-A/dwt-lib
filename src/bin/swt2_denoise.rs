use array_lib::io_nifti::{read_nifti, write_nifti};
use array_lib::ArrayDim;
use array_lib::io_cfl::read_cfl;
use dwt_lib::swt2::SWT2Plan;
use dwt_lib::wavelet::{Wavelet, WaveletType};
use num_complex::Complex32;
use dwt_lib::swt3::SWT3Plan;

fn main() {

    println!("loading image");
    let (x,x_dims) = read_cfl("out-0");
    println!("building transform data");
    let w = SWT3Plan::new(x_dims.shape_ns(),2,Wavelet::new(WaveletType::Daubechies2));
    let t_dims = ArrayDim::from_shape(&w.t_domain_shape());
    let mut t = t_dims.alloc(Complex32::ZERO);

    println!("decomposing");
    w.decompose(&x, &mut t);
    println!("finishing");
    write_nifti("t.nii",&t.iter().map(|x|x.norm()).collect::<Vec<_>>(),t_dims);
}





// fn main() {
//     let (slice, slice_dims, ..) = read_nifti::<f32>("slice.nii");
//
//     let slice: Vec<_> = slice.into_iter().map(|x| Complex32::new(x, 0.)).collect();
//
//     let shape = slice_dims.shape_squeeze();
//     let nx = shape[0];
//     let ny = shape[1];
//
//     let w = Wavelet::new(WaveletType::Daubechies2);
//
//     let n_levels = 6;
//     let swt = SWT2Plan::new(nx, ny, n_levels, &w);
//     let mut decomp = vec![Complex32::ZERO; swt.t_domain_size()];
//
//     swt.decompose(&slice, &mut decomp);
//     swt.soft_thresh(&mut decomp, 0.02);
//
//     let out = decomp.iter().map(|x| x.norm()).collect::<Vec<_>>();
//     write_nifti("dec.nii", &out, ArrayDim::from_shape(&[nx, ny, swt.t_bands()]));
//
//     let mut rec = vec![Complex32::ZERO; nx * ny];
//     swt.reconstruct(&decomp, &mut rec);
//
//     let out = rec.iter().map(|x| x.norm()).collect::<Vec<_>>();
//     write_nifti("rec.nii", &out, ArrayDim::from_shape(&[nx, ny]));
// }