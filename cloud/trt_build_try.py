"""On a GPU whose TensorRT autotuner fails (Thunder): try the cheaper builder settings until an engine builds."""
import os, sys, traceback
import torch
from usrm2 import trt
gpu = torch.cuda.get_device_name(0).replace(" ", "_")
for name, w in (("recto", 256), ("m7", 192)):
    plan = f"{trt.PLANS}/{name}_p{w}_b1_fp16_{gpu}.plan"
    if os.path.exists(plan):
        print(name, "exists", flush=True); continue
    for level, nt in ((0, True), (0, False), (1, True), (2, True)):
        try:
            print(f"{name}: trying level={level} no_timing={nt}", flush=True)
            trt.build(f"{trt.PLANS}/{name}_b1_fp16.onnx", plan, w, level=level, no_timing=nt)
            break
        except Exception as e:
            print(f"  failed: {str(e)[:200]}", flush=True)
print("TRYDONE", flush=True)
