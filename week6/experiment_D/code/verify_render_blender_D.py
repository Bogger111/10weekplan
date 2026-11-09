import bpy,sys,json
import numpy as np
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--')+1]).resolve();d=np.load(out/'data/geometry_ABCD.npz');scene=bpy.context.scene
names=['A_original','B_smooth3d','C_sort3d','D_local_original_root'];maxerr=0.;entries=[]
assert (scene.frame_start,scene.frame_end,scene.render.fps)==(1,50,10)
for key in names:
 for p in range(2):
  obj=bpy.data.objects[f'{key}__track_{p}'];assert len(obj.data.shape_keys.key_blocks)==50 and len(obj.data.vertices)==19158
  a=d[key][:,p];expected=np.stack((a[...,0],a[...,2],-a[...,1]),-1);err=0.
  for t in range(50):
   scene.frame_set(t+1);ob=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ob.to_mesh();arr=np.empty(19158*3,np.float32);mesh.vertices.foreach_get('co',arr);ob.to_mesh_clear();err=max(err,float(np.abs(arr.reshape(-1,3)-expected[t]).max()))
  maxerr=max(maxerr,err);entries.append({'object':obj.name,'max_error_m':err})
assert maxerr<1e-5
report={'native_windows_Blender':True,'version':bpy.app.version_string,'reopened_file':bpy.data.filepath,'checked_mesh_frames':400,'max_vertex_error_m':maxerr,'frames':50,'fps':10,'tracking_id_labels':8,'entries':entries}
(out/'blender_verify.json').write_text(json.dumps(report,indent=2))
frames=out/'blender_frames';frames.mkdir(exist_ok=True);scene.render.image_settings.file_format='PNG';scene.render.filepath=str(frames/'frame_')
bpy.ops.render.render(animation=True);assert len(list(frames.glob('frame_*.png')))==50
print('VERIFIED 400 MESH FRAMES AND RENDERED 50 FRAMES',flush=True)
