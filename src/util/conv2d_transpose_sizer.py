
h_in = 5#27#58#120
w_in = 5#37#78#160

padding = 1
dilation = 1
stride = 2
kernel = 4

# output_padding assumed = 0.
h_out = (h_in - 1) * stride - 2*padding + dilation * (kernel - 1) + 1
w_out = (w_in - 1) * stride - 2*padding + dilation * (kernel - 1) + 1

print(f'Hout, Wout = {h_out},{w_out}')