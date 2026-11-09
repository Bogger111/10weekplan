"""Reuse the established Blender layout conventions for four methods."""
import bpy,sys,json,math
import numpy as np
from pathlib import Path
from mathutils import Vector
out=Path(sys.argv[sys.argv.index('--')+1]).resolve();d=np.load(out/'data/geometry_ABCD.npz')
names=['A_original','B_smooth3d','C_sort3d','D_local_original_root'];labels=['A ORIGINAL','B 3D SMOOTH','C SORT + KALMAN','D LOCAL + ORIGINAL ROOT']
def convert(a):return np.stack((a[...,0],a[...,2],-a[...,1]),-1).astype(np.float32)
vs=[convert(d[n]) for n in names];faces=d['faces'];ids=d['ids'];ci=int(d['center'])
assert all(a.shape==(50,2,19158,3) and np.isfinite(a).all() for a in vs)
assert np.array_equal(d[names[0]][:,:,ci],d[names[3]][:,:,ci]) and np.array_equal(ids,np.tile([0,1],(50,1)))
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=50;scene.render.fps=10;scene.render.fps_base=1
scene.render.engine='BLENDER_WORKBENCH';scene.render.resolution_x=2400;scene.render.resolution_y=500;scene.render.resolution_percentage=100
scene.display.shading.light='STUDIO';scene.display.shading.color_type='MATERIAL';scene.display.shading.show_shadows=True;scene.display.shading.show_cavity=True
scene.display.shading.background_type='WORLD';scene.world.color=(.055,.065,.085)
def material(name,color):
 m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);return m
mats=[material('ID 0 Blue',(.16,.70,.95)),material('ID 1 Orange',(1,.58,.2))];textmat=material('Labels',(.84,.87,.92))
lo=vs[0].min(axis=(0,1,2));hi=vs[0].max(axis=(0,1,2));width=float(hi[0]-lo[0]);spacing=width+1;offsets=np.array([-1.5,-.5,.5,1.5])*spacing
objects=[]
for key,verts,off,label in zip(names,vs,offsets,labels):
 for p in range(2):
  mesh=bpy.data.meshes.new(f'{key}_ID{p}_mesh');mesh.from_pydata(verts[0,p].tolist(),[],faces.tolist());mesh.update()
  ob=bpy.data.objects.new(f'{key}__track_{p}',mesh);scene.collection.objects.link(ob);ob.location.x=float(off);mesh.materials.append(mats[p]);ob['tracking_id']=int(ids[0,p]);ob['method']=key
  for poly in mesh.polygons:poly.use_smooth=True
  ob.shape_key_add(name='Basis')
  for frame in range(2,51):
   sk=ob.shape_key_add(name=f'frame_{frame:03d}',from_mix=False);sk.data.foreach_set('co',verts[frame-1,p].ravel())
   for f,value in [(frame-1,0.),(frame,1.),(frame+1,0.)]:
    if f<=50:sk.value=value;sk.keyframe_insert(data_path='value',frame=f)
   sk.value=0
  action=ob.data.shape_keys.animation_data.action
  try:curves=action.fcurves
  except AttributeError:curves=action.layers[0].strips[0].channelbag(action.slots[0]).fcurves
  for curve in curves:
   for point in curve.keyframe_points:point.interpolation='LINEAR'
  objects.append(ob)
  bpy.ops.object.text_add(rotation=(math.pi/2,0,0));tag=bpy.context.object;tag.name=f'ID_LABEL_{key}_{p}';tag.data.body=f'ID {ids[0,p]}';tag.data.align_x='CENTER';tag.data.size=.14;tag.data.materials.append(mats[p]);bpy.ops.object.convert(target='MESH')
  for t in range(50):
   # Same naming and color mapping; follow the corresponding mesh without altering it.
   tag.location=(float(verts[t,p,ci,0]+off),float(lo[1]-.15),float(verts[t,p,:,2].max()+.07));tag.keyframe_insert(data_path='location',frame=t+1)
  print('CREATED',key,'ID',p,flush=True)
 bpy.ops.object.text_add(location=(float((lo[0]+hi[0])/2+off),float(lo[1]-1),float(hi[2]+.38)),rotation=(math.pi/2,0,0))
 text=bpy.context.object;text.name=f'METHOD_LABEL_{key}';text.data.body=label;text.data.align_x='CENTER';text.data.size=.20;text.data.materials.append(textmat);bpy.ops.object.convert(target='MESH')
cx=float((lo[0]+hi[0])/2);cy=float((lo[1]+hi[1])/2);cz=float((lo[2]+hi[2])/2+.25)
target=Vector((cx,cy,cz));location=Vector((cx,float(lo[1]-12),cz))
def camera(name):
 data=bpy.data.cameras.new(name);obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj);obj.location=location;obj.rotation_euler=(target-location).to_track_quat('-Z','Y').to_euler();data.type='ORTHO';data.ortho_scale=3*spacing+width+1.5;return obj
fixed=camera('CAM_FIXED');scene.camera=fixed
anchors=vs[0][:,:,ci]
for cname,track in [('CAM_FOLLOW_TRACK_0',anchors[:,0]),('CAM_FOLLOW_TRACK_1',anchors[:,1]),('CAM_FOLLOW_BOTH_ROOTS',anchors.mean(axis=1))]:
 cam=camera(cname)
 for t in range(50):cam.location=location+Vector((track[t]-track[0]).tolist());cam.keyframe_insert(data_path='location',frame=t+1)
scene['coordinate_system']='Shared source camera space (x,y,z) rotated to (x,z,-y); columns translated on display X only.'
scene['tracking']='ID 0 blue / ID 1 orange; A/B official IDs and C/D independent SORT IDs exactly correspond throughout 50 frames.'
scene['D_definition']='B local 163-joint SO(3) rotation + shape smoothing; Original center vertex 5054 exactly retained; no Kalman position writes.'
scene['camera']='One common orthographic observer, widened for four columns; same light, shading, topology, timebase. Not source-video intrinsics K.'
scene.frame_set(1)
for area in bpy.context.screen.areas:
 if area.type=='VIEW_3D':
  area.spaces.active.region_3d.view_perspective='CAMERA';area.spaces.active.region_3d.view_camera_zoom=10;area.spaces.active.shading.type='SOLID';area.spaces.active.shading.color_type='MATERIAL';area.spaces.active.overlay.show_overlays=False
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.wm.save_as_mainfile(filepath=str(out/'human_motion_ABCD.blend'),compress=True)
(out/'blender_build.json').write_text(json.dumps({'objects':[o.name for o in objects],'frames':50,'fps':10,'resolution':[2400,500],'column_offsets_m':offsets.tolist(),'cameras':4,'tracking_id_labels':8},indent=2))
print('BLEND SAVED',flush=True)
