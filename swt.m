x = niftiread("slice_orig.nii");
x = mat2gray(x);
niftiwrite(real(x),"slice");
[LL,HL,LH,HH] = swtf(x,1);
r = iswtf(LL,HL,LH,HH,1);
niftiwrite(real(r),"rec_fft");

%% matlab swt
swc = swt2(x,2,"db2");
swc = shrink_complex(swc,0.1);
rec = iswt2(swc,"db2");
niftiwrite(swc,"swc");
niftiwrite(rec,"rec");
%% custom fft swt

[LoD,HiD,~,~] = wfilters('db2');

nx = size(x,1);
ny = size(x,2);

% kx_lo = fft(embed_filter_for_fft(LoD,nx));
% kx_hi = fft(embed_filter_for_fft(HiD,nx));
% ky_lo = fft(embed_filter_for_fft(LoD,ny));
% ky_hi = fft(embed_filter_for_fft(HiD,ny));

% flen = numel(LoD);
% kx_lo = zeros(nx,1);
% kx_lo(1:flen) = flip(LoD(:));
% kx_hi = zeros(nx,1);
% kx_hi(1:flen) = flip(HiD(:));
% ky_lo = zeros(ny,1);
% ky_lo(1:flen) = flip(LoD(:));
% ky_hi = zeros(ny,1);
% ky_hi(1:flen) = flip(HiD(:));
% 
% kx_lo = fft(kx_lo);
% kx_hi = fft(kx_hi);
% ky_lo = fft(ky_lo);
% ky_hi = fft(ky_hi);

level = 1;
kx_lo = filter_kernel(flip(LoD),level,nx)';
kx_hi = filter_kernel(flip(HiD),level,nx)';
ky_lo = filter_kernel(flip(LoD),level,ny)';
ky_hi = filter_kernel(flip(HiD),level,ny)';

KLL = kx_lo .* ky_lo';
KLH = kx_lo .* ky_hi';
KHL = kx_hi .* ky_lo';
KHH = kx_hi .* ky_hi';

X = fftn(x);

LL = ifftn(X.*KLL);
LH = ifftn(X.*KLH);
HL = ifftn(X.*KHL);
HH = ifftn(X.*KHH);

out = zeros(512,256,4);
out(:,:,1) = HL;
out(:,:,2) = LH;
out(:,:,3) = HH;
out(:,:,4) = LL;

niftiwrite(real(out),"swc_fft");




