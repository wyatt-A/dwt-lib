function [LL,HL,LH,HH] = swtf(x,level)

[LoD,HiD,~,~] = wfilters('db2');

nx = size(x,1);
ny = size(x,2);

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
