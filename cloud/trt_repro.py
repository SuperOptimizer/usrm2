"""Minimal reproducer: TensorRT engine build fails on Thunder Compute (cuMemHostGetDevicePointer in the autotuner).
pip install torch tensorrt-cu12 onnx ; python trt_repro.py"""
import torch, tensorrt as trt

net = torch.nn.Sequential(torch.nn.Conv3d(1, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv3d(16, 2, 1)).eval()
torch.onnx.export(net, torch.zeros(1, 1, 64, 64, 64), "tiny.onnx", input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
log = trt.Logger(trt.Logger.WARNING)
b = trt.Builder(log)
n = b.create_network(0)
p = trt.OnnxParser(n, log)
assert p.parse_from_file("tiny.onnx"), [str(p.get_error(i)) for i in range(p.num_errors)]
cfg = b.create_builder_config()
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
ser = b.build_serialized_network(n, cfg)
print("TensorRT", trt.__version__, "GPU", torch.cuda.get_device_name(0), "engine built:", ser is not None)
