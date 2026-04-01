function f = filter_kernel(filter, level, n)

    % Dilation factor
    s = 2^(level - 1);

    % Dilate by inserting zeros between coefficients
    L = numel(filter);
    Ld = (L - 1) * s + 1;

    if Ld > n
        error('Dilated filter length (%d) exceeds n (%d).', Ld, n);
    end

    h = zeros(1, Ld);
    h(1:s:end) = filter;

    % Shift so the filter center is at zero lag for FFT convolution
    %h = circshift(h, -floor(Ld / 2));
    
    k = zeros(1,n);
    k(1:Ld) = h;

    h = circshift(k, -floor(Ld / 2));

    % Pad/truncate to FFT length and transform
    f = fft(h);
end
