
h_in = 15
w_in = 20

padding = 1
dilation = 1
stride = 1
kernel = 4

h_out = (h_in + 2*padding - dilation*(kernel - 1) - 1) / stride
h_out += 1

w_out = (w_in + 2*padding - dilation*(kernel - 1) - 1) / stride
w_out += 1

print(f'Hout, Wout = {h_out},{w_out}')