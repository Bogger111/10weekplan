import json,sys,csv
from pathlib import Path
import numpy as np
out=Path(sys.argv[1]);d=np.load(out/'data/geometry_ABCD.npz');ci=int(d['center']);K=d['K'].astype(float)
names=['A_original','B_smooth3d','C_sort3d','D_local_original_root','D0_transfer']
def project(v):
 v=v.astype(float);return np.stack((K[:,None,None,0,0]*v[...,0]/v[...,2]+K[:,None,None,0,2],K[:,None,None,1,1]*v[...,1]/v[...,2]+K[:,None,None,1,2]),-1)
def acc(v):return float(np.linalg.norm(np.diff(v.astype(float),n=2,axis=0),axis=-1).mean())
rows=[];steps=[]
for name in names:
 v=d[name];root=v[:,:,ci].astype(float);rel=v.astype(float)-root[:,:,None,:];uv=project(v)
 rows.append({'method':name,'center3d':acc(root),'relative_mesh3d':acc(rel),'absolute_mesh3d':acc(v),'projected_mesh2d':acc(uv),'projected_center2d':acc(uv[:,:,ci])})
 for p in range(2):
  astep=np.linalg.norm(np.diff(v[:,p].astype(float),axis=0),axis=-1).mean(axis=-1);rstep=np.linalg.norm(np.diff(rel[:,p],axis=0),axis=-1).mean(axis=-1)
  steps.append({'method':name,'track':p,'max_absolute_mean_mesh_step_m':float(astep.max()),'absolute_max_step_frame_pair':[int(astep.argmax()+1),int(astep.argmax()+2)],'max_relative_mean_mesh_step_m':float(rstep.max()),'relative_max_step_frame_pair':[int(rstep.argmax()+1),int(rstep.argmax()+2)]})
D=d['D_local_original_root'];A=d['A_original'];B=d['B_smooth3d'];D0=d['D0_transfer']
delta=(D.astype(float)-D[:,:,ci:ci+1].astype(float))-(B.astype(float)-B[:,:,ci:ci+1].astype(float))
report={'D0_metrics':rows[-1],'all_recomputed_metrics':rows,'D0_D1_max_projected_coordinate_difference_px':float(np.abs(project(D)-project(D0)).max()),'D_local_vs_B_local_max_coordinate_difference_m':float(np.abs(delta).max()),'D_and_A_center_bitwise_equal':bool(np.array_equal(D[:,:,ci],A[:,:,ci])),'D_center_depth_change_from_A_m':float(np.abs(D[:,:,ci,2]-A[:,:,ci,2]).max()),'all_arrays_finite':all(bool(np.isfinite(d[n]).all()) for n in names),'frame_steps':steps,'min_Z_m':{n:float(d[n][...,2].min()) for n in names},'coordinate_system':'same source camera space as original ABC; no axis conversion before metrics','all_checks_pass':True}
assert report['D_and_A_center_bitwise_equal'] and report['all_arrays_finite'] and report['D0_D1_max_projected_coordinate_difference_px']<.001
(out/'qa_additional.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
