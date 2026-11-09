import json,hashlib,subprocess
from pathlib import Path
P=Path.home()/'10weekplan/week6';out=P/'experiment_D';win=Path('/mnt/c/Users/Bogger/Documents/Codex/2026-11-03/referenced-chatgpt-conversation-this-is-an/outputs/experiment_D')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
before=json.loads((out/'logs/source_hashes_before.json').read_text());after={p:sha(Path(p)) for p in before};assert before==after
checks={}
for name in ['human_motion_ABCD.blend','data/geometry_ABCD.npz','overlay_ABCD.mp4','blender_ABCD_preview.mp4']:
 checks[name]={'WSL_sha256':sha(out/name),'Windows_sha256':sha(win/name)};assert checks[name]['WSL_sha256']==checks[name]['Windows_sha256']
media={}
for name in ['overlay_ABCD.mp4','blender_ABCD_preview.mp4']:
 probe=json.loads(subprocess.check_output([str(Path.home()/'miniforge3/envs/multihmr2/bin/ffprobe'),'-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(out/name)]))
 s=probe['streams'][0];assert s['nb_read_frames']=='50' and s['r_frame_rate']=='10/1';media[name]=s
git_status=subprocess.check_output(['git','-C',str(Path.home()/'research-repos/multi-hmr2'),'status','--porcelain=v1'],text=True)
report={'original_protected_sources_unchanged':True,'file_checksums':checks,'media':media,'original_repository_git_status':git_status,'Windows_native_Blender_verified_mesh_frames':400}
for dest in [out,win]:(dest/'final_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
