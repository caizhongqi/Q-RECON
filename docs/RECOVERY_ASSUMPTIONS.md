# Recovery assumptions and success criteria

## Exact analytic recovery

The analytic method is exact under all of the following conditions:

1. the leaked gradient corresponds to one sample;
2. the first trainable layer is a biased `Linear` layer;
3. that layer consumes the raw input, possibly after flattening;
4. the first-layer bias gradient is non-zero;
5. complete first-layer weight and bias gradients are visible.

It is not a universal result for batch aggregation, convolutional first layers,
secure aggregation, gradient clipping or differential privacy.

## Reported success criteria

- `bitwise_equal_percent`: exact equality of floating-point tensors;
- `within_1e-6_percent`: numerical exactness under float32 roundoff;
- `uint8_equal_percent`: equality after converting image values to 8-bit pixels;
- `max_absolute_error`: worst recovered element;
- `relative_l2_error`: global relative reconstruction error;
- PSNR and correlation: perceptual/structural supporting metrics only.

A low gradient-matching loss is never reported as proof of data recovery by
itself.
