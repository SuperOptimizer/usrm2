"""Minimal reproducer: TensorRT engine build fails on Thunder Compute with
  [TRT] [E] Error Code: 9: Skipping tactic 0x0 due to exception [autotuner.cpp: create_timing_resources]
  In the autotuner, CUDA error 1 from 'cuMemHostGetDevicePointer' ... Could not find any implementation for node ForeignNode[...]
A conv3d + InstanceNorm3d block in fp16 (what nnU-Net style nets are made of) is enough to trigger it.
pip install torch tensorrt-cu12 onnx ; python trt_repro.py"""
import torch, tensorrt as trt

def blk(i, o):
    return torch.nn.Sequential(torch.nn.Conv3d(i, o, 3, padding=1), torch.nn.InstanceNorm3d(o, affine=True), torch.nn.LeakyReLU(0.01))
net = torch.nn.Sequential(blk(1, 32), blk(32, 32), torch.nn.Conv3d(32, 2, 1)).eval().half()
torch.onnx.export(net, torch.zeros(1, 1, 128, 128, 128, dtype=torch.half), "tiny.onnx", input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
log = trt.Logger(trt.Logger.WARNING)
b = trt.Builder(log)
n = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
p = trt.OnnxParser(n, log)
assert p.parse_from_file("tiny.onnx"), [str(p.get_error(i)) for i in range(p.num_errors)]
cfg = b.create_builder_config()
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
ser = b.build_serialized_network(n, cfg)
print("TensorRT", trt.__version__, "GPU", torch.cuda.get_device_name(0), "engine built:", ser is not None)
