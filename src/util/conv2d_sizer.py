
h_in = 32
w_in = 32

padding = 0
dilation = 1
stride = 2
kernel = 4

h_out = (h_in + 2*padding - dilation*(kernel - 1) - 1) / stride
h_out += 1

w_out = (w_in + 2*padding - dilation*(kernel - 1) - 1) / stride
w_out += 1

print(f'Hout, Wout = {h_out},{w_out}')