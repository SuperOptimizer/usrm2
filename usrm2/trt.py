"""TensorRT fp16 engines for the teachers (built by tsm for this GPU: fp32 window in -> fp32 2-class
logits out, at a fixed window). ~2.4x the torch bf16 throughput for the recto teacher."""
import glob

import torch

PLANS = "/vesuvius/tsm/models/trt"


def plan(name, window):
    """The cached engine for `name` ('recto' / 'm7') at `window`, or None."""
    p = sorted(glob.glob(f"{PLANS}/{name}_p{window}_b1_fp16_*.plan"))
    return p[-1] if p else None


class Engine:
    def __init__(self, path, dev="cuda"):
        import tensorrt as trt
        self.dev = torch.device(dev)
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.rt.deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.in_shape, self.out_shape = tuple(self.engine.get_tensor_shape("x")), tuple(self.engine.get_tensor_shape("y"))
        self.out = torch.empty(self.out_shape, dtype=torch.float32, device=self.dev)

    def __call__(self, x):
        assert tuple(x.shape) == self.in_shape, f"engine wants {self.in_shape}, got {tuple(x.shape)}"
        x = x.float().contiguous()
        self.ctx.set_tensor_address("x", x.data_ptr())
        self.ctx.set_tensor_address("y", self.out.data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream(self.dev).cuda_stream)
        return self.out
