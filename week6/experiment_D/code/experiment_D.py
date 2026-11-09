"""Cached Multi-HMR local smoothing experiment; never runs infer_video/infer_image."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
os.environ.setdefault('GALLIUM_DRIVER','llvmpipe')
# Preserve the import order needed by this machine.
import torch
import triton
import torch._dynamo
import ast,csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path.home()/'10weekplan/week6'; SRC=P/'results/sort_comparison_20261103'
OUT=P/'experiment_D'; CODE=OUT/'code'; DATA=OUT/'data'; LOG=OUT/'logs'
sys.path.insert(0,str(P))
import comparison_anchor_fix as old
import multihmr2
NAMES=['A_original','B_smooth3d','C_sort3d','D_local_original_root']
LABELS=['A Original','B 3D Smooth','C SORT + Kalman','D SORT + Local / Original Root']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write_json(path,obj):path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def write_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def project(v,K):
 v=v.astype(np.float64);K=K.astype(np.float64)
 return np.stack((K[:,None,None,0,0]*v[...,0]/v[...,2]+K[:,None,None,0,2],K[:,None,None,1,1]*v[...,1]/v[...,2]+K[:,None,None,1,2]),-1)
def d2(a):return np.linalg.norm(np.diff(a.astype(np.float64),n=2,axis=0),axis=-1)
def stat(a):
 a=a.astype(np.float64);norm=np.linalg.norm(a,axis=-1)
 return {'mean_norm':float(norm.mean()),'max_norm':float(norm.max()),'max_abs_coordinate':float(np.abs(a).max())}

def replay_sort(vertices):
 """Run the exact audited Track/IoU definitions; association only for D."""
 tree=ast.parse((SRC/'experiment.py').read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('Track','iou')]
 env={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SRC/'experiment.py'),'exec'),env)
 Track=env['Track'];iou=env['iou'];tracks=[];nextid=0;ids=np.full((50,2),-1);shifts=np.zeros((50,2,3));trace=[]
 for t in range(50):
  lo=vertices[t].min(axis=1);hi=vertices[t].max(axis=1);boxes=np.concatenate(((lo+hi)/2,hi-lo),-1)
  order=np.array([1,0] if t%2 else [0,1]);det=boxes[order]
  for tr in tracks:tr.predict()
  pairs=[]
  if tracks:
   mat=np.array([[iou(tr.x[:6],box) for box in det] for tr in tracks]);rr,cc=linear_sum_assignment(1-mat);pairs=[(r,c) for r,c in zip(rr,cc) if mat[r,c]>=.01]
  used=set()
  for r,c in pairs:
   tr=tracks[r];tr.update(det[c]);p=int(order[c]);ids[t,p]=tr.id;shifts[t,p]=tr.x[:3]-det[c,:3];used.add(c)
   trace.append({'frame':t+1,'observation_column':p,'sort_id':int(tr.id),'new_track':False,'iou':float(mat[r,c])})
  for c in range(2):
   if c not in used:
    tr=Track(det[c],nextid);nextid+=1;tracks.append(tr);p=int(order[c]);ids[t,p]=tr.id
    trace.append({'frame':t+1,'observation_column':p,'sort_id':int(tr.id),'new_track':True,'iou':None})
  tracks=[tr for tr in tracks if tr.age<=2]
 return ids,shifts,trace

def local_smooth_by_id(pose,shape,ids):
 """B's SO(3) and shape filters grouped by SORT identity and consecutive frames."""
 outpose=pose.copy();outshape=shape.copy();segments=[]
 for tid in sorted(np.unique(ids)):
  obs=[(t,int(np.where(ids[t]==tid)[0][0])) for t in range(50) if np.count_nonzero(ids[t]==tid)==1]
  for segment in old.consecutive_groups(obs):
   ts=np.array([t for t,p in segment]);ps=np.array([p for t,p in segment]);po=pose[ts,ps];sh=shape[ts,ps]
   for joint in range(pose.shape[2]):outpose[ts,ps,joint,:3,:3]=old.filter_rotation(po[:,joint,:3,:3])
   outshape[ts,ps]=np.clip(old.filter_numeric(sh,old.SHAPE_CUTOFF),0,1).astype(np.float32)
   segments.append({'sort_id':int(tid),'start_frame':int(ts[0]+1),'end_frame':int(ts[-1]+1),'length':int(len(ts))})
 return outpose,outshape,segments

def evaluate(groups,K,ids,ci):
 perperson=[];perframe=[];aggregates=[];frame_vectors={}
 for key,v in zip(NAMES,groups):
  uv=project(v,K);root=v[:,:,ci].astype(np.float64);rel=v.astype(np.float64)-root[:,:,None,:]
  values={'center3d_m_frame2':d2(root),'relative_mesh3d_m_frame2':d2(rel).mean(axis=-1),'absolute_mesh3d_m_frame2':d2(v).mean(axis=-1),'projected_mesh2d_px_frame2':d2(uv).mean(axis=-1),'projected_center2d_px_frame2':d2(uv[:,:,ci])}
  frame_vectors[key]=values
  aggregates.append({'method':key,**{k:float(x.mean()) for k,x in values.items()}})
  for p in range(2):perperson.append({'method':key,'sort_id':int(ids[0,p]),**{k:float(x[:,p].mean()) for k,x in values.items()}})
  for t in range(48):
   for p in range(2):perframe.append({'method':key,'center_frame':t+2,'sort_id':int(ids[t+1,p]),**{k:float(x[t,p]) for k,x in values.items()}})
 anomalies=[]
 for key,values in frame_vectors.items():
  for metric,x in values.items():
   for p in range(2):
    arr=x[:,p];median=np.median(arr);mad=np.median(np.abs(arr-median));threshold=float(median+5*max(float(mad),1e-12));top=np.argsort(arr)[-3:][::-1]
    for i in sorted(set(top.tolist()+np.flatnonzero(arr>threshold).tolist())):
     anomalies.append({'method':key,'sort_id':int(ids[0,p]),'metric':metric,'center_frame':int(i+2),'value':float(arr[i]),'median_plus_5MAD_threshold':threshold,'above_threshold':bool(arr[i]>threshold),'top3':bool(i in top)})
 write_csv(OUT/'metrics.csv',aggregates);write_csv(OUT/'metrics_by_track.csv',perperson);write_csv(OUT/'jitter_by_frame.csv',perframe);write_csv(OUT/'anomaly_frames.csv',anomalies)
 pct={k:100*(aggregates[3][k]/aggregates[0][k]-1) for k in values}
 write_json(OUT/'metrics.json',{'aggregate':aggregates,'per_track':perperson,'D_vs_A_percent_change':pct,'anomaly_definition':'per method / track / metric: value > median + 5*MAD; also report top three frames, without claiming ground-truth failures','frames':50,'fps':10,'windows_per_person':48,'center_vertex':ci,'source_image_resolution':[768,432],'no_ground_truth':True})
 fig,axes=plt.subplots(5,2,figsize=(14,16),sharex=True)
 colors=['#507ba2','#d49840','#5aa47d','#bb4e85']
 for row,metric in enumerate(values):
  for p in range(2):
   ax=axes[row,p]
   for key,color in zip(NAMES,colors):ax.plot(np.arange(2,50),frame_vectors[key][metric][:,p],label=key.split('_')[0],color=color,lw=1.3)
   ax.set_title(f'ID {ids[0,p]} - {metric}');ax.grid(alpha=.2)
 for ax in axes[-1]:ax.set_xlabel('Middle source frame at 10 FPS');ax.legend(ncol=4)
 fig.tight_layout();fig.savefig(OUT/'jitter_curves.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,key,unit in zip(axes,['relative_mesh3d_m_frame2','projected_mesh2d_px_frame2'],['m/frame²','px/frame²']):
  bars=ax.bar(['A','B','C','D'],[r[key] for r in aggregates],color=colors);ax.bar_label(bars,fmt='%.4f',padding=3);ax.set_title(key);ax.set_ylabel(unit);ax.set_ylim(0,max(r[key] for r in aggregates)*1.18)
 fig.tight_layout();fig.savefig(OUT/'metric_summary.png',dpi=160);plt.close(fig)
 return aggregates,pct

def render_overlay(groups,K,ids,faces,ci):
 frames=P/'results/postprocess_diagnosis_20261103/frames';dest=OUT/'overlay_frames';dest.mkdir(exist_ok=True)
 fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
 try:font=ImageFont.truetype(fontpath,20);header=ImageFont.truetype(fontpath,22)
 except OSError:font=ImageFont.load_default();header=font
 projections=[project(v,K) for v in groups]
 for t in range(50):
  panels=[]
  for n,g in enumerate(groups):
   img=Image.fromarray(old.render_one(frames/f'{t+1:06d}.png',g[t],faces,torch.from_numpy(K[t]),ids[t].tolist()));draw=ImageDraw.Draw(img)
   for p in range(2):
    uv=projections[n][t,p];x=int(np.clip(np.median(uv[:,0])-22,0,img.width-75));y=int(np.clip(np.percentile(uv[:,1],2)-26,0,img.height-24));color=(40,177,242) if p==0 else (255,148,51)
    draw.rounded_rectangle((x-3,y-1,x+65,y+25),radius=4,fill=(20,24,30));draw.text((x,y),f'ID {ids[t,p]}',font=font,fill=color)
   panels.append(img)
  w,h=panels[0].size;combined=Image.new('RGB',(4*w,h+56),(22,25,31));draw=ImageDraw.Draw(combined)
  for n,(panel,label) in enumerate(zip(panels,LABELS)):
   combined.paste(panel,(n*w,56));draw.text((n*w+12,4),label,font=header,fill='white');draw.text((n*w+12,29),f'Frame {t+1:02d}/50 | 10 FPS | ID 0 blue / ID 1 orange',font=font,fill=(190,200,210))
  combined.save(dest/f'{t+1:06d}.png')
  if t%5==0:print('Overlay rendered',t+1,flush=True)
 subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-framerate','10','-i',str(dest/'%06d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'overlay_ABCD.mp4')],check=True)

def main():
 for p in [OUT,CODE,DATA,LOG]:p.mkdir(parents=True,exist_ok=True)
 protected=[SRC/'experiment.py',SRC/'geometry_3groups.npz',SRC/'human_motion_ABC.blend',P/'comparison_anchor_fix.py',P/'results/postprocess_diagnosis_20261103/inference.npz',P/'results/postprocess_diagnosis_20261103/ablations.npz',P/'results/blender_comparison_20261103/geometry_3groups.npz']
 before={str(p):sha(p) for p in protected};write_json(LOG/'source_hashes_before.json',before)
 inf=np.load(protected[4]);cache=np.load(SRC/'geometry_3groups.npz');abl=np.load(protected[5])
 A=cache['original'];B=cache['smooth3d'];C=cache['sort3d'];K=cache['K'];faces=cache['faces'];ci=int(cache['center']);pose=inf['pose'];shape=inf['shape']
 assert A.shape==(50,2,19158,3) and ci==5054 and np.array_equal(A,inf['vertices']) and np.array_equal(K,inf['K'])
 ids,replay_shifts,trace=replay_sort(A)
 assert np.array_equal(ids,cache['sort_ids']) and np.array_equal(replay_shifts,cache['sort_shifts'])
 assert np.array_equal(ids,cache['ids']) and np.array_equal(ids,np.tile([0,1],(50,1))), 'This cached experiment needs explicit mapping if identities differ'
 write_json(LOG/'sort_association_trace.json',trace)
 # D0 transfers the already smoothed local geometry to the original center.
 D0=A[:,:,ci:ci+1,:]+(B-B[:,:,ci:ci+1,:])
 assert np.array_equal(D0[:,:,ci],A[:,:,ci])
 print('D0 built; rerunning local filters by exact SORT IDs',flush=True)
 smpose,smshape,segments=local_smooth_by_id(pose,shape,ids)
 print('Filtered tracks',segments,flush=True)
 print('Loading existing checkpoint body model; no image/video inference',flush=True)
 session=multihmr2.init_hmr_session(str(old.CHECKPOINT),compile_model=False);body=session.model.full_body_decoder.body_model;body.set_skinning_method('lbs');device=session.device
 D1=np.empty_like(A);Brebuild=np.empty_like(A);delta_all=np.empty_like(A)
 for t in range(50):
  base=old.anny_forward(body,torch.from_numpy(pose[t]),torch.from_numpy(shape[t]),device,'local-bone')['vertices']
  sm=old.anny_forward(body,torch.from_numpy(smpose[t]),torch.from_numpy(smshape[t]),device,'local-bone')['vertices']
  delta=((sm-sm[:,ci:ci+1])-(base-base[:,ci:ci+1])).cpu().numpy()
  # Root rotation follows B's all-163-joint smoother; Root POSITION stays Original.
  D1[t]=A[t]+delta;delta_all[t]=delta
  Brebuild[t]=A[t]+abl['anchor_delta'][t,:,None,:].astype(np.float32)+delta
  if t%10==0:print('Body-model local reconstruction',t+1,flush=True)
 assert np.array_equal(D1[:,:,ci],A[:,:,ci]), 'D1 center must equal A bitwise'
 assert np.isfinite(D1).all() and np.isfinite(D0).all() and (D1[...,2]>0).all()
 assert np.array_equal(ids,cache['ids'])
 diff=stat(D1.astype(np.float64)-D0.astype(np.float64));bdelta=stat(Brebuild.astype(np.float64)-B.astype(np.float64))
 assert diff['max_abs_coordinate']<1e-5 and bdelta['max_abs_coordinate']<1e-5
 localdelta=(D1.astype(np.float64)-D1[:,:,ci:ci+1].astype(np.float64))-(A.astype(np.float64)-A[:,:,ci:ci+1].astype(np.float64))
 assert np.linalg.norm(localdelta,axis=-1).mean()>1e-6
 np.savez_compressed(DATA/'geometry_ABCD.npz',A_original=A,B_smooth3d=B,C_sort3d=C,D_local_original_root=D1,D0_transfer=D0,K=K,faces=faces,ids=ids,official_ids=cache['ids'],center=ci)
 np.savez_compressed(DATA/'D_parameters.npz',original_pose=pose,smoothed_pose=smpose,original_shape=shape,smoothed_shape=smshape,sort_ids=ids)
 groups=[A,B,C,D1];aggs,pct=evaluate(groups,K,ids,ci)
 speed=np.linalg.norm(np.diff(D1,axis=0),axis=-1).mean(axis=-1)
 invariant={'D1_center_bitwise_equals_A':True,'D0_center_bitwise_equals_A':True,'D1_max_center_xyz_difference_m':0.0,'D1_sort_ID_equals_official_ID_all_frames':True,'D1_z_at_center_bitwise_equals_A':True,'D1_no_Kalman_shift_added':True,'all_geometry_finite':all(bool(np.isfinite(v).all()) for v in groups),'D1_min_vertex_z_m':float(D1[...,2].min()),'D1_max_mean_mesh_frame_step_m':float(speed.max()),'D1_D0_difference':diff,'B_rebuild_vs_saved_B':bdelta,'D1_local_mesh_vs_A':stat(localdelta),'D1_pose_vs_B_cached_sm_pose_max_abs':float(np.abs(smpose-abl['sm_pose']).max()),'B_includes_global_root_orientation_smoothing':True,'root_position_definition':'official person-center vertex 5054, not bone-pelvis translation','shape_cutoff_hz':old.SHAPE_CUTOFF,'rotation_alpha':old.ROT_ALPHA,'rotation_joints':163,'segments':segments,'no_inference_or_training':True}
 write_json(OUT/'validation.json',invariant)
 print(json.dumps({'validation':invariant,'metrics':aggs,'D_vs_A_percent':pct},indent=2),flush=True)
 render_overlay(groups,K,ids,faces,ci)
 after={str(p):sha(p) for p in protected};assert before==after;write_json(LOG/'source_preservation.json',{'before':before,'after':after,'unchanged':True})
 print('EXPERIMENT D COMPLETE',flush=True)
if __name__=='__main__':main()
