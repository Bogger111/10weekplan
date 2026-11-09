import os
os.environ['PYOPENGL_PLATFORM']='osmesa'
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
os.environ.setdefault('GALLIUM_DRIVER','llvmpipe')
import torch
import triton
import torch._dynamo
from pathlib import Path
import sys, json, pickle
import numpy as np
from scipy.signal import butter, sosfiltfilt
sys.path.insert(0, str(Path.home()/'10weekplan/week6'))
import comparison_anchor_fix as old
from multihmr2 import init_hmr_session, infer_video
P=Path.home()/'10weekplan/week6'
OUT=P/'results/postprocess_diagnosis_20261103'
OUT.mkdir(parents=True,exist_ok=True)
frames=OUT/'frames'; frames.mkdir(exist_ok=True)
def low(x,cut=2): return sosfiltfilt(butter(2,cut,fs=10,output='sos'),x,axis=0).copy()
def main():
 print('Initializing session',flush=True)
 session=init_hmr_session(str(old.CHECKPOINT),compile_model=False)
 print('Running inference',flush=True)
 pred=infer_video(session,str(old.VIDEO),tmp_dir=str(frames))
 print('Frames',len(pred),flush=True)
 if len(pred)!=50 or any(len(p)!=2 for p in pred): raise RuntimeError('Expected 50 frames with two detections; inspect tracking before filtering')
 body=session.model.full_body_decoder.body_model; body.set_skinning_method('lbs'); dev=session.device; ci=int(session.model.full_body_decoder.person_center_idx)
 T=len(pred); verts=[]; poses=[]; shapes=[]; Ks=[]; ids=[]; pelv=[]; err=[]
 for t,p in enumerate(pred):
  order=torch.argsort(p.persons.v3d[:,ci,0],descending=True)
  sh=p.persons.shape[order].detach().cpu().float(); world=p.persons.bone_poses[order].detach().cpu().float(); v=p.persons.v3d[order].detach().cpu().numpy()
  world_out=old.anny_forward(body,world,sh,dev,'world')
  local=body.get_pose_parameterization(world_out,pose_parameterization='local-bone').detach().cpu()
  rec=old.anny_forward(body,local,sh,dev,'local-bone')
  reconstructed=old.align_to_pelvis(rec,p.persons.transl_pelvis[order].reshape(2,3).to(dev)).cpu().numpy()
  err.append(np.linalg.norm(reconstructed-v,axis=-1))
  verts.append(v); poses.append(local.numpy()); shapes.append(sh.numpy()); Ks.append(p.K.detach().cpu().numpy()); ids.append(p.persons.track_id[order].cpu().numpy()); pelv.append(p.persons.transl_pelvis[order].reshape(2,3).cpu().numpy())
  if t%10==0: print('Converted',t+1,flush=True)
 v=np.stack(verts); pose=np.stack(poses); shape=np.stack(shapes); K=np.stack(Ks); ids=np.stack(ids); pelv=np.stack(pelv); err=np.stack(err)
 np.savez_compressed(OUT/'inference.npz',vertices=v,pose=pose,shape=shape,K=K,ids=ids,pelvis=pelv,roundtrip=err,center=ci,faces=body.faces.detach().cpu().numpy() if isinstance(body.faces,torch.Tensor) else np.asarray(body.faces))
 print('Cache written',v.shape,'K',K[:,0,0].min(),K[:,0,0].max(),'ids',np.unique(ids,axis=0),'roundtrip',err.mean(),err.max(),flush=True)
 anchor=v[:,:,ci,:]; sm_anchor=low(anchor); sm_shape=np.clip(low(shape),0,1)
 sm_pose=pose.copy()
 for person in range(2):
  for joint in range(pose.shape[2]): sm_pose[:,person,joint,:3,:3]=old.filter_rotation(pose[:,person,joint,:3,:3])
 delta_rot=[];delta_shape=[]
 for t in range(T):
  base=old.anny_forward(body,torch.from_numpy(pose[t]),torch.from_numpy(shape[t]),dev,'local-bone')['vertices']
  rot=old.anny_forward(body,torch.from_numpy(sm_pose[t]),torch.from_numpy(shape[t]),dev,'local-bone')['vertices']
  shp=old.anny_forward(body,torch.from_numpy(pose[t]),torch.from_numpy(sm_shape[t]),dev,'local-bone')['vertices']
  def rel(x): return x-x[:,ci:ci+1,:]
  delta_rot.append((rel(rot)-rel(base)).cpu().numpy()); delta_shape.append((rel(shp)-rel(base)).cpu().numpy())
  if t%10==0: print('Ablation',t+1,flush=True)
 np.savez_compressed(OUT/'ablations.npz',anchor_delta=sm_anchor-anchor,rotation_delta=np.stack(delta_rot),shape_delta=np.stack(delta_shape),sm_pose=sm_pose,sm_shape=sm_shape)
 print('Ablation cache written',flush=True)
if __name__=='__main__': main()
