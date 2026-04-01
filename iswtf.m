function x = iswtf(LL, HL, LH, HH, level)

    [LoD, HiD, LoR, HiR] = wfilters('db2');

    nx = size(LL, 1);
    ny = size(LL, 2);

    % Forward analysis kernels
    ax_lo = filter_kernel(flip(LoD), level, nx)';
    ax_hi = filter_kernel(flip(HiD), level, nx)';
    ay_lo = filter_kernel(flip(LoD), level, ny)';
    ay_hi = filter_kernel(flip(HiD), level, ny)';

    FLL = ax_lo .* ay_lo';
    FLH = ax_lo .* ay_hi';
    FHL = ax_hi .* ay_lo';
    FHH = ax_hi .* ay_hi';

    % Inverse synthesis kernels
    sx_lo = filter_kernel(flip(LoR), level, nx)';
    sx_hi = filter_kernel(flip(HiR), level, nx)';
    sy_lo = filter_kernel(flip(LoR), level, ny)';
    sy_hi = filter_kernel(flip(HiR), level, ny)';

    RLL = sx_lo .* sy_lo';
    RLH = sx_lo .* sy_hi';
    RHL = sx_hi .* sy_lo';
    RHH = sx_hi .* sy_hi';

    % Numerator: synthesis applied to subbands
    Num = ...
        fftn(LL) .* RLL + ...
        fftn(LH) .* RLH + ...
        fftn(HL) .* RHL + ...
        fftn(HH) .* RHH;

    % Exact composite transfer function
    Den = RLL .* FLL + RLH .* FLH + RHL .* FHL + RHH .* FHH;

    x = real(ifftn(Num ./ (Den + 1e-12)));
end
