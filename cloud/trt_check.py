"""On an instance: build (if needed) and validate the TensorRT engines against torch on one real window each."""
import time
import numpy as np, torch
from usrm2 import data, teacher, m7, trt
U = "https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0"
ct = data.open_zarr(U)[34432:34432 + 256, 15104:15104 + 256, 18432:18432 + 256]
x = torch.from_numpy(data.zscore(ct)[None, None]).cuda()
net = teacher.load(teacher.CKPT, "cuda")
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    ref = torch.softmax(net(x)["surface"].float(), 1)[0, 1].cpu().numpy()
    t = time.time(); [net(x) for _ in range(10)]; torch.cuda.synchronize(); print("recto torch", round((time.time() - t) / 10, 3), "s/window", flush=True)
del net; torch.cuda.empty_cache()
eng = trt.Engine(trt.plan("recto", 256), "cuda")
out = torch.softmax(eng(x), 1)[0, 1].cpu().numpy()
t = time.time(); [eng(x) for _ in range(10)]; torch.cuda.synchronize(); print("recto trt", round((time.time() - t) / 10, 3), "s/window; corr vs torch", round(float(np.corrcoef(ref.ravel(), out.ravel())[0, 1]), 4), flush=True)
lo = data.open_zarr(U.rsplit("/", 1)[0] + "/2")[8608:8608 + 192, 3776:3776 + 192, 4608:4608 + 192]
n = m7.load(m7.CKPT, "cpu"); mean, std, a, b = n.norm; del n
x = torch.from_numpy(((np.clip(lo.astype(np.float32), a, b) - mean) / std)[None, None]).cuda()
n = m7.load(m7.CKPT, "cuda")
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    ref = torch.softmax(n(x).float(), 1)[0, 1].cpu().numpy()
    t = time.time(); [n(x) for _ in range(10)]; torch.cuda.synchronize(); print("m7 torch", round((time.time() - t) / 10, 3), "s/window", flush=True)
del n; torch.cuda.empty_cache()
eng = trt.Engine(trt.plan("m7", 192), "cuda")
out = torch.softmax(eng(x), 1)[0, 1].cpu().numpy()
t = time.time(); [eng(x) for _ in range(10)]; torch.cuda.synchronize(); print("m7 trt", round((time.time() - t) / 10, 3), "s/window; corr vs torch", round(float(np.corrcoef(ref.ravel(), out.ravel())[0, 1]), 4), flush=True)
print("TRTCHECKDONE", flush=True)
