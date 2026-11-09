from pathlib import Path
import sys,os
os.environ['PYOPENGL_PLATFORM']='osmesa';os.environ['LIBGL_ALWAYS_SOFTWARE']='1';os.environ['GALLIUM_DRIVER']='llvmpipe'
import torch
import triton
import torch._dynamo
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path.home()/'10weekplan/week6'))
import comparison_anchor_fix as old
from multihmr2 import init_hmr_session
P=Path.home()/'10weekplan/week6/results/postprocess_diagnosis_20261103'; d=np.load(P/'inference.npz'); a=np.load(P/'candidate_anchor_deltas.npz');v=d['vertices'];pose=d['pose'];shape=d['shape'];K=d['K']; c=int(d['center'])
session=init_hmr_session(str(old.CHECKPOINT),compile_model=False);body=session.model.full_body_decoder.body_model;body.set_skinning_method('lbs');dev=session.device
rot=pose[:,:,:,:3,:3]; sm=pose.copy()
for p in range(2):
 for j in range(rot.shape[2]):
  raw=Rotation.from_matrix(rot[:,p,j]); filtered=Rotation.from_matrix(old.filter_rotation(rot[:,p,j])); relative=(raw.inv()*filtered).as_rotvec(); correction=Rotation.from_rotvec(.25*relative).as_matrix(); sm[:,p,j,:3,:3]=rot[:,p,j]@correction
print('mean/max matrix correction',np.abs(sm-pose).mean(),np.abs(sm-pose).max(),flush=True)
DV=[]
for t in range(50):
 base=old.anny_forward(body,torch.from_numpy(pose[t]),torch.from_numpy(shape[t]),dev,'local-bone')['vertices']; out=old.anny_forward(body,torch.from_numpy(sm[t]),torch.from_numpy(shape[t]),dev,'local-bone')['vertices'];dv=(out-out[:,c:c+1])-(base-base[:,c:c+1]);DV.append(dv.cpu().numpy())
 if t%10==0:print('frame',t+1,flush=True)
DV=np.stack(DV);q=v+a['uv_only'][:,:,None,:]+DV
def proj(x):return np.stack((K[:,None,None,0,0]*x[...,0]/x[...,2]+K[:,None,None,0,2],K[:,None,None,1,1]*x[...,1]/x[...,2]+K[:,None,None,1,2]),-1)
u=proj(v);w=proj(q)
for p in range(2):
 n=lambda x:np.linalg.norm(x,axis=-1)
 print('track',p,'acc original and fixed',n(np.diff(u[:,p],n=2,axis=0)).mean(),n(np.diff(w[:,p],n=2,axis=0)).mean(),'drift mean p95',n(w[:,p]-u[:,p]).mean(),np.percentile(n(w[:,p]-u[:,p]),95),flush=True)
np.savez_compressed(P/'corrected_pose_test.npz',delta=DV,sm_pose=sm)
