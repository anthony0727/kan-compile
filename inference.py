
import torch
# import torch_tensorrt
from utils import timed, generate_data, init_model
torch.set_float32_matmul_precision('high')
# torch._dynamo.list_backends()
# backends = ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'openxla_eval']

N_ITERS = 10
infer_only_backends = ['cudagraphs', 'inductor', 'tvm'] 
#, 'openvino', 'tensorrt'] not working

# cpu_backends = ['ipex']

model = init_model()
eager_times = []
for i in range(N_ITERS):
    inp = generate_data(16)[0]
    with torch.no_grad():
        _, eager_time = timed(lambda: model(inp))
    eager_times.append(eager_time)
    print(f"eager eval time {i}: {eager_time}")

print("~" * 10)

for backend in infer_only_backends:
    print('*'*10, backend, '*'*10)
    model = init_model()
    model_opt = torch.compile(model, backend=backend)
    compile_times = []
    for i in range(N_ITERS):
        inp = generate_data(16)[0]
        with torch.no_grad():
            _, compile_time = timed(lambda: model_opt(inp))
        compile_times.append(compile_time)
        print(f"compile eval time {i}: {compile_time}")
    print("~" * 10)

    import numpy as np
    eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    speedup = eager_med / compile_med
    # assert(speedup > 1)
    print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
    print("~" * 10)
