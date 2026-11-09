from pathlib import Path
import numpy as np,json
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path.home()/'10weekplan/week6/results'; O=P/'comparison_projection_fix_20261103'; D=P/'postprocess_diagnosis_20261103'
d=np.load(O/'geometry_projection.npz'); base=np.load(D/'inference.npz'); v=d['original']; smooth=d['smooth']; K=d['K']; ids=d['track_ids']; center=int(d['center']);
if v.shape[0]!=50 or v.shape!=smooth.shape:raise RuntimeError('unexpected frame count or shape')
if any(set(row.tolist()) != set(ids[0].tolist()) for row in ids) or len(set(ids[0])) != 2:
 raise RuntimeError('missing or changing track IDs')
order=np.argsort(ids,axis=1)
v=np.take_along_axis(v,order[:,:,None,None],axis=1)
smooth=np.take_along_axis(smooth,order[:,:,None,None],axis=1)
ids=np.take_along_axis(ids,order,axis=1)
def proj(x):
 return np.stack((K[:,None,None,0,0]*x[...,0]/x[...,2]+K[:,None,None,0,2],K[:,None,None,1,1]*x[...,1]/x[...,2]+K[:,None,None,1,2]),-1)
def norm(x):return np.linalg.norm(x,axis=-1)
def acc(x):return float(norm(np.diff(x,n=2,axis=0)).mean())
u=proj(v);w=proj(smooth)
anchor_shift=norm((w-u)[:,:,center]); depth_change=np.abs((smooth-v)[:,:,center,2])
if anchor_shift.max()>2.001 or depth_change.max()>1e-5:
 raise RuntimeError(f'anchor constraint failed: pixel shift {anchor_shift.max()}, depth shift {depth_change.max()}')
report={'frames':len(v),'track_ids':ids[0].tolist(),'video_size_before_h264':'768 x 432 per panel','pixel_units':'preprocessed 768 x 432 frame','max_anchor_projection_shift_px':float(anchor_shift.max()),'max_anchor_depth_change_m':float(depth_change.max()),'definition':'mean norm of consecutive 2nd differences across corresponding vertices; includes real motion and is not ground-truth error','people':[]}
fig,axs=plt.subplots(3,2,figsize=(14,10),sharex=True);tt=np.arange(1,51)
for p in range(2):
 ds=norm(w[:,p]-u[:,p]); dm=norm(smooth[:,p]-v[:,p]); dx=norm(np.diff(w[:,p],n=2,axis=0)).mean(axis=1); raw=norm(np.diff(u[:,p],n=2,axis=0)).mean(axis=1)
 report['people'].append({'track_id':int(ids[0,p]),'vertices_2d_acceleration_px_original':acc(u[:,p]),'vertices_2d_acceleration_px_smooth':acc(w[:,p]),'anchor_2d_acceleration_px_original':acc(u[:,p,center]),'anchor_2d_acceleration_px_smooth':acc(w[:,p,center]),'relative_vertices_3d_acceleration_m_original':acc(v[:,p]-v[:,p,center:center+1]),'relative_vertices_3d_acceleration_m_smooth':acc(smooth[:,p]-smooth[:,p,center:center+1]),'mean_projection_change_px':float(ds.mean()),'p95_projection_change_px':float(np.percentile(ds,95)),'max_projection_change_px':float(ds.max()),'mean_vertex_change_mm':float(dm.mean()*1000),'p95_vertex_change_mm':float(np.percentile(dm,95)*1000),'max_vertex_change_mm':float(dm.max()*1000),'frame_mean_projection_change_max_frame':int(np.argmax(ds.mean(axis=1))+1),'frame_mean_projection_change_max_px':float(ds.mean(axis=1).max())})
 axs[0,p].plot(tt,u[:,p,center,0],label='original u');axs[0,p].plot(tt,w[:,p,center,0],label='smooth u',ls='--');axs[0,p].set_ylabel('anchor u (px)')
 axs[1,p].plot(tt[1:-1],raw,label='original');axs[1,p].plot(tt[1:-1],dx,label='smooth');axs[1,p].set_ylabel('vertex 2nd difference (px/frame²)')
 axs[2,p].plot(tt,ds.mean(axis=1),label='mean projected change');axs[2,p].plot(tt,np.percentile(ds,95,axis=1),label='95th percentile');axs[2,p].set_ylabel('change vs original (px)');axs[2,p].set_xlabel('Frame (1-based)');axs[0,p].set_title('Track ID '+str(ids[0,p]))
 for row in axs:row[p].grid(alpha=.3);row[p].legend(fontsize=8)
fig.tight_layout();fig.savefig(O/'final_trajectory_metrics.png',dpi=130)
# Match official-v3d cached baseline by track identity and test reconstruction reproducibility.
ref=base['vertices']; refids=base['ids']; diffs=[]
for t in range(50):
 for p in range(2):
  j=np.flatnonzero(refids[t]==ids[t,p]);
  if len(j)!=1:raise RuntimeError('identity mismatch')
  diffs.append(norm(v[t,p]-ref[t,j[0]]))
diffs=np.stack(diffs);report['official_cache_comparison_m']={'mean':float(diffs.mean()),'max':float(diffs.max())}
report['camera_depth_correlation']=[]
for p in range(2):
 z=v[:,p,center,2]; f=K[:,0,0]; report['camera_depth_correlation'].append({'corr_f_z':float(np.corrcoef(f,z)[0,1]),'corr_diff2_f_z':float(np.corrcoef(np.diff(f,n=2),np.diff(z,n=2))[0,1])})
(O/'final_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,indent=2))
