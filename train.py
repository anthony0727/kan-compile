
import torch
# import torch_tensorrt
from utils import timed, generate_data, init_model
import numpy as np

torch.set_float32_matmul_precision('high')
# torch._dynamo.list_backends()
backends = ['cudagraphs', 'inductor'] 
#, 'onnxrt', 'openxla', 'openxla_eval'] not working

device = 'cuda'
N_ITERS = 10


model = init_model()
opt = torch.optim.Adam(model.parameters())

def train(mod, data):
    opt.zero_grad(True)
    pred = mod(data[0])
    loss = torch.nn.CrossEntropyLoss()(pred, data[1])
    loss.backward()
    opt.step()

eager_times = []
for i in range(N_ITERS):
    inp = generate_data(16)
    _, eager_time = timed(lambda: train(model, inp))
    eager_times.append(eager_time)
    print(f"eager train time {i}: {eager_time}")
print("~" * 10)

for backend in backends:
    print('*'*10, backend, '*'*10)
    model = init_model()
    opt = torch.optim.Adam(model.parameters())
    train_opt = torch.compile(train, backend=backend)#mode="reduce-overhead")

    compile_times = []
    for i in range(N_ITERS):
        inp = generate_data(16)
        _, compile_time = timed(lambda: train_opt(model, inp))
        compile_times.append(compile_time)
        print(f"compile train time {i}: {compile_time}")
    print("~" * 10)

    eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    speedup = eager_med / compile_med
    assert(speedup > 1)
    print(f"(train) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
    print("~" * 10)