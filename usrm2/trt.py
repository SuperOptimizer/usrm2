"""TensorRT fp16 engines for the teachers (built by tsm for this GPU: fp32 window in -> fp32 2-class
logits out, at a fixed window). ~2.4x the torch bf16 throughput for the recto teacher."""
import glob
import os
import time

import torch

PLANS = os.environ.get("USRM2_TRT", "/vesuvius/tsm/models/trt")  # engines (GPU-specific) + the ONNX graphs (not)


def plan(name, window, build_if_missing=True):
    """The engine for `name` ('recto' / 'm7') at `window` on this GPU; built from the ONNX graph when absent."""
    p = sorted(glob.glob(f"{PLANS}/{name}_p{window}_b1_fp16_*.plan"))
    if p:
        return p[-1]
    onnx = f"{PLANS}/{name}_b1_fp16.onnx"
    if build_if_missing and os.path.exists(onnx):
        gpu = torch.cuda.get_device_name(0).replace(" ", "_")
        return build(onnx, f"{PLANS}/{name}_p{window}_b1_fp16_{gpu}.plan", window)
    return None


def build(onnx_path, plan_path, window, batch=1, workspace_gb=8.0):
    """Parse the (shape-agnostic) ONNX graph with its I/O pinned to [batch,C,window^3] and build a fp16 engine
    (ported from tsm.trt.build_engine; strongly typed, so the precision is the graph's own)."""
    import onnx
    import tensorrt as trt
    m = onnx.load(onnx_path, load_external_data=True)
    for vi in list(m.graph.input) + list(m.graph.output):
        d = vi.type.tensor_type.shape.dim
        if len(d) == 5:
            for i, v in enumerate([batch, d[1].dim_value, window, window, window]):
                d[i].dim_value = int(v)
    del m.graph.value_info[:]
    log = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(log)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, log)
    assert parser.parse(m.SerializeToString()), "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    t = time.time()
    ser = b.build_serialized_network(net, cfg)
    assert ser is not None, f"TensorRT build failed for {onnx_path}"
    open(plan_path + ".tmp", "wb").write(bytes(ser))
    os.replace(plan_path + ".tmp", plan_path)
    print(f"built {os.path.basename(plan_path)} ({ser.nbytes >> 20} MiB) in {time.time() - t:.0f}s", flush=True)
    return plan_path


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
