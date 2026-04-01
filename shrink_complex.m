function z_out = shrink_complex(z, lambda)
%SHRINK_COMPLEX Soft-thresholding for complex values
%
% z_out = shrink_complex(z, lambda)
%
% Inputs:
%   z      : complex array
%   lambda : threshold (scalar or same size as z)
%
% Output:
%   z_out  : thresholded complex array

    mag = abs(z);

    % Avoid divide-by-zero
    scale = max(0, 1 - lambda ./ (mag + eps));

    z_out = scale .* z;
end