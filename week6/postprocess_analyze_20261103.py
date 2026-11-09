from pathlib import Path
import json, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path.home()/'10weekplan/week6'; O=P/'results/postprocess_diagnosis_20261103'
d=np.load(O/'inference.npz'); a=np.load(O/'ablations.npz'); v=d['vertices']; K=d['K']; center=int(d['center']); ids=d['ids']; anchor=v[:,:,center]
def project(v):
 return np.stack((K[:,None,None,0,0]*v[...,0]/v[...,2]+K[:,None,None,0,2],K[:,None,None,1,1]*v[...,1]/v[...,2]+K[:,None,None,1,2]),-1)
def mean_norm(x): return float(np.linalg.norm(x,axis=-1).mean())
def pct(x): return [float(y) for y in np.percentile(x,[50,95,100])]
modes={'off':np.zeros_like(v),'anchor':a['anchor_delta'][:,:,None,:]+np.zeros_like(v),'rotation':a['rotation_delta'],'shape':a['shape_delta']}
proj0=project(v); report={'frames':len(v),'track_ids_by_x':ids.tolist(),'focal_px':pct(K[:,0,0]),'roundtrip_m_mean':float(d['roundtrip'].mean()),'roundtrip_m_p95':float(np.percentile(d['roundtrip'],95)),'roundtrip_m_max':float(d['roundtrip'].max()),'modes':{}}
for name,delta in modes.items():
 q=v+delta; uv=project(q); summary=[]
 for p in range(2):
  change3=np.linalg.norm(delta[:,p],axis=-1); change2=np.linalg.norm((uv-proj0)[:,p],axis=-1)
  ac3=np.diff(q[:,p],n=2,axis=0); ac2=np.diff(uv[:,p],n=2,axis=0)
  relative=q[:,p]-q[:,p,center:center+1]
  summary.append({'vertex_second_difference_m':mean_norm(ac3),'relative_vertex_second_difference_m':mean_norm(np.diff(relative,n=2,axis=0)),'projected_vertex_second_difference_px':mean_norm(ac2),'vertex_displacement_m_median_p95_max':pct(change3),'projected_displacement_px_median_p95_max':pct(change2),'projected_displacement_px_mean':float(change2.mean()),'anchor_second_difference_m':mean_norm(np.diff(q[:,p,center],n=2,axis=0)),'anchor_second_difference_px':mean_norm(np.diff(uv[:,p,center],n=2,axis=0))})
 report['modes'][name]=summary
# Existing 50-frame exported parameter sets, compare after stable x sorting raw data, respecting old code mapping.
rdir=P/'results/blender_baseline_20261029_203921/video/anny_params'; sdir=P/'results/anny_smoothed_full'
raw=[]; sm=[]
for f in sorted(rdir.glob('*.pkl')):
 with f.open('rb') as h: x=pickle.load(h)
 with (sdir/f.name).open('rb') as h: y=pickle.load(h)
 order=x['pose_parameters'][:,0,0,3].argsort(descending=True)
 raw.append((x['pose_parameters'][order].numpy(),x['shape'][order].float().numpy()))
 sm.append((y['pose_parameters'][order].numpy(),y['shape'][order].float().numpy()))
rp=np.stack([x[0] for x in raw]); sp=np.stack([x[0] for x in sm]); rs=np.stack([x[1] for x in raw]); ss=np.stack([x[1] for x in sm])
report['saved_parameter_sets']={'frames':len(raw),'root_delta_m_median_p95_max':pct(np.linalg.norm(sp[:,:,0,:3,3]-rp[:,:,0,:3,3],axis=-1)),'shape_abs_delta_mean':float(np.abs(ss-rs).mean()),'rotation_matrix_abs_delta_mean':float(np.abs(sp[:,:,:,:3,:3]-rp[:,:,:,:3,:3]).mean()),'note':'Previously exported bone transforms: world rotations, root translation overwritten. They cannot be treated as local poses.'}
(O/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
fig,axs=plt.subplots(4,2,figsize=(15,12),sharex=True); t=np.arange(1,len(v)+1)
for p in range(2):
 for j in range(3): axs[0,p].plot(t,anchor[:,p,j],label='xyz'[j]); axs[0,p].plot(t,(anchor+a['anchor_delta'])[:,p,j],ls='--',alpha=.7)
 uv0=project(v)[:,p,center]; uv1=project(v+modes['anchor'])[:,p,center]
 for j in range(2): axs[1,p].plot(t,uv0[:,j],label='uv'[j]); axs[1,p].plot(t,uv1[:,j],ls='--',alpha=.7)
 for name in ['anchor','rotation','shape']:
  dv=modes[name][:,p]; axs[2,p].plot(t,np.linalg.norm(dv,axis=-1).mean(axis=-1)*1000,label=name)
  pv=np.linalg.norm((project(v+modes[name])-proj0)[:,p],axis=-1).mean(axis=-1); axs[3,p].plot(t,pv,label=name)
 axs[0,p].set_title(f'Person {p}, ordered by X; solid=original, dashed=anchor')
 axs[0,p].set_ylabel('anchor (m)'); axs[1,p].set_ylabel('anchor projection (px)'); axs[2,p].set_ylabel('mean vertex change (mm)'); axs[3,p].set_ylabel('mean projected change (px)'); axs[3,p].set_xlabel('Frame (1-based)')
 for row in axs: row[p].grid(alpha=.25); row[p].legend(loc='best',fontsize=8)
fig.tight_layout();fig.savefig(O/'trajectories.png',dpi=130)
print(json.dumps(report,indent=2))
