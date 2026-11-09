from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
O=Path.home()/'10weekplan/week6/results/postprocess_diagnosis_20261103'
d=np.load(O/'inference.npz'); a=np.load(O/'ablations.npz'); v=d['vertices']; K=d['K']; c=int(d['center']); anchor=v[:,:,c,:]
f=K[:,None,0,0]; cx=K[:,None,0,2]; cy=K[:,None,1,2]
uv=np.stack((f*anchor[:,:,0]/anchor[:,:,2]+cx,f*anchor[:,:,1]/anchor[:,:,2]+cy),axis=-1); invscale=f/anchor[:,:,2]
def low(x):return sosfiltfilt(butter(2,2,fs=10,output='sos'),x,axis=0)
def projection(x):return np.stack((K[:,None,None,0,0]*x[...,0]/x[...,2]+K[:,None,None,0,2],K[:,None,None,1,1]*x[...,1]/x[...,2]+K[:,None,None,1,2]),-1)
def ans(u,z):return np.stack(((u[:,:,0]-cx)*z/f,(u[:,:,1]-cy)*z/f,z),-1)-anchor
variants={'original':np.zeros_like(anchor),'old_xyz':a['anchor_delta'],'xy_only':np.concatenate((low(anchor[:,:,:2])-anchor[:,:,:2],np.zeros((50,2,1))),-1),'depth_only':np.concatenate((np.zeros((50,2,2)),(low(anchor[:,:,2])-anchor[:,:,2])[...,None]),-1),'uv_only':ans(low(uv),anchor[:,:,2]),'scale_only':ans(uv,f/low(invscale)),'uv_and_scale':ans(low(uv),f/low(invscale))}
uv0=projection(v)
print('mode, person, mean_2d_accel, anchor_2d_accel, mean_disp_px, p95_disp_px, p95_disp_mm, mean_3d_accel')
for name,delta in variants.items():
 for r in [0,0.25,0.5] if name in ('uv_only','uv_and_scale') else [0]:
  q=v+delta[:,:,None,:]+r*a['rotation_delta']; u=projection(q); dv=np.linalg.norm(u-uv0,axis=-1)
  for p in range(2):
   vec=lambda x: np.linalg.norm(x,axis=-1).mean()
   print(name+'+rot'+str(r),p,round(vec(np.diff(u[:,p],n=2,axis=0)),3),round(vec(np.diff(u[:,p,c],n=2,axis=0)),3),round(dv[:,p].mean(),3),round(np.percentile(dv[:,p],95),3),round(np.percentile(np.linalg.norm(q[:,p]-v[:,p],axis=-1),95)*1000,1),round(vec(np.diff(q[:,p],n=2,axis=0)),4))
np.savez_compressed(O/'candidate_anchor_deltas.npz',**variants)
