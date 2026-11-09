from pathlib import Path
import sys
import torch
import triton
import torch._dynamo
import numpy as np
sys.path.insert(0,str(Path.home()/'10weekplan/week6'))
import comparison_projection_fix as m
O=Path.home()/'10weekplan/week6/results/postprocess_diagnosis_20261103'; d=np.load(O/'inference.npz'); frames=sorted((O/'frames').glob('*.png'))
for t in (0,24,49):
 v=d['vertices'][t]; k=torch.from_numpy(d['K'][t]); ids=d['ids'][t]
 left=m.render_one(frames[t],v,d['faces'],k,ids); right=m.render_one(frames[t],v.copy(),d['faces'],k,ids)
 print('frame',t+1,'vertex_equal',np.array_equal(v,v.copy()),'render_equal',np.array_equal(left,right),'max_pixel_error',np.max(np.abs(left.astype(int)-right.astype(int))),flush=True)
