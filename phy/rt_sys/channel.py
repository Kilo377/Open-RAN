"""Chunked absolute RT CFR cache. Physical geometry is independent of policy."""
from pathlib import Path
from importlib.metadata import version
import json
import os
import numpy as np
from phy.rt_sys.contracts import ChannelSnapshot, digest


def position_at(trajectory, time_s):
    times = np.asarray(trajectory['times_s'], float)
    points = np.asarray(trajectory['positions_m'], float)
    if len(times)<2 or points.shape != (len(times),3) or np.any(np.diff(times)<=0):
        raise ValueError('Trajectory needs increasing times and matching XYZ positions')
    if not np.isfinite(points).all() or time_s<times[0]-1e-9 or time_s>times[-1]+1e-9:
        raise ValueError('Trajectory outside configured duration')
    i = min(max(int(np.searchsorted(times,time_s,side='right'))-1,0),len(times)-2)
    velocity = (points[i+1]-points[i])/(times[i+1]-times[i])
    return points[i] + velocity*(time_s-times[i]), velocity


def combine_complex(value):
    # Current Dr.Jit outputs are real/imag pairs; NumPy/Torch outputs may be complex.
    if isinstance(value, tuple):
        return np.asarray(value[0])+1j*np.asarray(value[1])
    if hasattr(value,'detach'): return value.detach().cpu().numpy()
    return np.asarray(value)


class RtChannelProvider:
    def __init__(self, manifest, cache_root, *, device='cuda', backend=None):
        self.manifest = json.loads(json.dumps(manifest))
        self.device = device
        self.backend = backend or SionnaRayTracer(manifest,device)
        # Backend fingerprint includes actual material properties and scene meshes.
        self.key = digest({'schema':1,'manifest':manifest,'backend':self.backend.fingerprint()})
        self.root = Path(cache_root)/self.key
        self.root.mkdir(parents=True,exist_ok=True)
        self.meta = self.root/'manifest.json'
        text = json.dumps({'cache_key':self.key,'manifest':manifest,'backend':self.backend.fingerprint()},sort_keys=True,indent=2)
        if self.meta.exists() and self.meta.read_text()!=text:
            raise ValueError('Cache manifest mismatch')
        if not self.meta.exists(): self.meta.write_text(text)
        self.loaded_chunk = None
        self.loaded_arrays = None
        self.hits = self.misses = 0
        self.total = int(manifest['total_slots'])
        self.dt = float(manifest['slot_duration_ms'])/1000
        self.chunk_slots = int(manifest.get('chunk_slots',20))
        self.update_slots = max(1,round(float(manifest['path_update_ms'])/manifest['slot_duration_ms']))
        if abs(self.update_slots*manifest['slot_duration_ms']-manifest['path_update_ms'])>1e-6:
            raise ValueError('Path update must be an integral number of simulation slots')
        self.backend.validate_geometry()

    def snapshot(self, slot_id):
        if not 0<=slot_id<=self.total:
            raise ValueError('Requested slot outside RT trace')
        chunk = slot_id//self.chunk_slots
        if chunk != self.loaded_chunk:
            path = self.root/f'chunk_{chunk:06d}.npz'
            if path.exists():
                arrays = dict(np.load(path,allow_pickle=False)); self.hits+=1
            else:
                first = chunk*self.chunk_slots
                last = min(self.total+1,first+self.chunk_slots)
                arrays = self.backend.generate(first,last,self.update_slots)
                tmp = path.with_suffix(f'.{os.getpid()}.tmp')
                with tmp.open('wb') as stream: np.savez_compressed(stream,**arrays)
                os.replace(tmp,path); self.misses+=1
            self.loaded_arrays, self.loaded_chunk = arrays, chunk
        i = slot_id-chunk*self.chunk_slots
        snap = ChannelSnapshot(slot_id,tuple(u['ue_id'] for u in self.manifest['ues']),
            tuple(c['cell_id'] for c in self.manifest['cells']),self.loaded_arrays['cfr'][i],
            self.loaded_arrays['positions'][i],self.key,{
                'normalized':False,'bandwidth_hz':self.manifest['bandwidth_hz'],'source':'sionna_rt','path_update_ms':self.manifest['path_update_ms'],
                'carrier_frequency_hz':self.manifest['carrier_frequency_hz'],
                'subcarrier_spacing_hz':self.manifest['subcarrier_spacing_hz'],
                'cache_hits':self.hits,'cache_misses':self.misses})
        return snap.validate()

    def pregenerate(self):
        for start in range(0,self.total+1,self.chunk_slots): self.snapshot(start)


class SionnaRayTracer:
    def __init__(self,manifest,device):
        import sionna.rt as rt
        import mitsuba as mi
        self.rt,self.mi,self.m = rt,mi,manifest
        if device=='cuda' and not mi.variant().startswith('cuda'):
            raise RuntimeError(f'Sionna RT is not using CUDA: {mi.variant()}')
        self.scene = rt.load_scene(getattr(rt.scene,manifest['scene']))
        self.scene.frequency = manifest['carrier_frequency_hz']
        self.scene.bandwidth = manifest['bandwidth_hz']
        self.scene.tx_array = rt.PlanarArray(num_rows=manifest['tx_rows'],num_cols=manifest['tx_cols'],
            vertical_spacing=.5,horizontal_spacing=.5,pattern='tr38901',polarization='V')
        self.scene.rx_array = rt.PlanarArray(num_rows=1,num_cols=manifest['num_rx_ant'],
            vertical_spacing=.5,horizontal_spacing=.5,pattern='dipole',polarization='V')
        for cell in manifest['cells']:
            self.scene.add(rt.Transmitter(cell['cell_id'],position=cell['position_m'],orientation=cell['orientation_rad']))
        for ue in manifest['ues']:
            pos,vel = position_at(ue['trajectory'],0)
            self.scene.add(rt.Receiver(ue['ue_id'],position=pos.tolist(),velocity=vel.tolist()))
        self.solver = rt.PathSolver(deterministic=True)
        self._last_block=None
        self._last_cfr=None
        self._last_path_arrays=None

    def fingerprint(self):
        import hashlib
        materials = []
        # Unnamed Mitsuba shapes receive process-global counters (no-name-N).
        # Neither names nor dictionary order are stable across scene loads.
        # Bind materials to the actual transformed mesh bytes instead.
        for obj in self.scene.objects.values():
            material = obj.radio_material
            parameters=self.mi.traverse(obj.mi_mesh)
            geometry=hashlib.sha256()
            for field,dtype in (('vertex_positions','<f4'),('faces','<u4')):
                values=np.asarray(parameters[field],dtype=dtype)
                geometry.update(values.tobytes())
            materials.append({'geometry_sha256':geometry.hexdigest(),'material':material.name,
                'relative_permittivity':float(np.asarray(material.relative_permittivity).reshape(-1)[0]),
                'conductivity':float(np.asarray(material.conductivity).reshape(-1)[0])})
        materials.sort(key=lambda row:json.dumps(row,sort_keys=True))
        # Hash mesh inputs, not only scene name. Also hashes packaged XML and meshes.
        from pathlib import Path
        asset = Path(getattr(self.rt.scene,self.m['scene']))
        meshes = {}
        for p in sorted(asset.parent.rglob('*')):
            if p.is_file(): meshes[str(p.relative_to(asset.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
        import platform
        return {'python':platform.python_version(),'torch':version('torch'),'sionna':version('sionna'),'sionna_rt':version('sionna-rt'),
                'mitsuba':version('mitsuba'),'drjit':version('drjit'),
                'materials':materials,'assets':meshes,'rt_variant':self.mi.variant()}

    def validate_geometry(self):
        # Outdoor fixtures: upward ray detects a building/roof above a UE point.
        # Every segment is also tested against the mesh at actual UE height.
        mi = self.mi
        for ue in self.m['ues']:
            points = ue['trajectory']['positions_m']
            for xyz in points:
                ray = mi.Ray3f(mi.Point3f(*xyz),mi.Vector3f(0,0,1))
                if bool(np.asarray(self.scene.mi_scene.ray_test(ray)).reshape(-1)[0]):
                    raise ValueError(f'UE {ue["ue_id"]} point inside/under geometry: {xyz}')
            for left,right in zip(points,points[1:]):
                delta=np.asarray(right)-left; distance=np.linalg.norm(delta)
                if distance<1e-9: continue
                ray = mi.Ray3f(mi.Point3f(*left),mi.Vector3f(*(delta/distance).tolist()))
                ray.maxt=float(distance-1e-5)
                if bool(np.asarray(self.scene.mi_scene.ray_test(ray)).reshape(-1)[0]):
                    raise ValueError(f'UE {ue["ue_id"]} trajectory crosses scene geometry')
        # BS locations may be above roofs; reject placement beneath a roof.
        for cell in self.m['cells']:
            ray=mi.Ray3f(mi.Point3f(*cell['position_m']),mi.Vector3f(0,0,1))
            if bool(np.asarray(self.scene.mi_scene.ray_test(ray)).reshape(-1)[0]):
                raise ValueError(f'BS {cell["cell_id"]} lies beneath scene geometry')

    def generate(self,first,last,update_slots):
        m=self.m; dt=m['slot_duration_ms']/1000; ns=m['num_symbols']; nf=m['fft_size']
        frequencies=(np.arange(nf)-nf//2)*m['subcarrier_spacing_hz']
        out=[]; positions=[]; paths_arrays={}
        # Regenerate geometry only at configured boundaries; cfr() evolves path Doppler.
        for block in range((first//update_slots)*update_slots,last,update_slots):
            length=min(update_slots,m['total_slots']+1-block)
            if block != self._last_block:
                for ue in m['ues']:
                    pos,vel=position_at(ue['trajectory'],block*dt)
                    receiver=self.scene.get(ue['ue_id']); receiver.position=pos.tolist(); receiver.velocity=vel.tolist()
                params=dict(m['ray_solver']); params['seed']=int(m['rng_seed'])+block
                paths=self.solver(self.scene,**params)
                length=min(update_slots,m['total_slots']+1-block)
                cfr=combine_complex(paths.cfr(frequencies=self.mi.Float(frequencies.tolist()),
                    sampling_frequency=ns/dt,num_time_steps=length*ns,
                    normalize_delays=False,normalize=False,out_type='numpy'))
                # Native axes [UE,RX,BS,TX,time,freq] -> per-slot [UE,BS,RX,TX,symbol,freq]
                cfr=np.transpose(cfr,(0,2,1,3,4,5))
                if cfr.shape[:4] != (len(m['ues']),len(m['cells']),m['num_rx_ant'],m['tx_rows']*m['tx_cols']):
                    raise ValueError(f'Unexpected Sionna CFR shape {cfr.shape}')
                block_paths={}
                # Preserve delays and angles for channel provenance and future controls.
                for attr in ('tau','theta_t','phi_t','theta_r','phi_r','doppler'):
                    block_paths[f'block_{block}_{attr}']=np.asarray(getattr(paths,attr))
                block_paths[f'block_{block}_a']=combine_complex(paths.a)
                self._last_block,self._last_cfr,self._last_path_arrays=block,cfr,block_paths
            cfr=self._last_cfr
            paths_arrays.update(self._last_path_arrays)
            for slot in range(max(first,block),min(last,block+length)):
                j=slot-block
                out.append(cfr[...,j*ns:(j+1)*ns,:].astype(np.complex64))
                positions.append([position_at(u['trajectory'],slot*dt)[0] for u in m['ues']])
        return {'cfr':np.asarray(out),'positions':np.asarray(positions),**paths_arrays}
