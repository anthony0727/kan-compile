import torch
from kan import KAN

# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
    return (
        torch.randn(b, 1, 28, 28).view(-1, 28 * 28).to(torch.float32).cuda(),
        torch.randint(10, (b,)).cuda(),
    )

def init_model():
    return KAN([28 * 28, 64, 10]).to(torch.float32).cuda()