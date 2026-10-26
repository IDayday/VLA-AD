"""Exact selected-operator replay. Coefficients use CPU FP64, MLP CPU FP32.

The torch cap application is the immutable native cap implementation, not a
second approximate NumPy implementation. Base/student inference is external.
"""
from pathlib import Path
import csv, gzip, hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _npz(ref):
    p = ROOT / ref['path']
    if _sha(p) != ref['sha256']: raise ValueError(f'Frozen asset SHA mismatch: {p}')
    with np.load(p, allow_pickle=False) as z: return {k:np.array(z[k]) for k in z.files}


def pool_base(base, image, text):
    b = np.asarray(base)
    image, text = np.asarray(image), np.asarray(text)
    if b.dtype != np.float32 or b.ndim != 2 or b.shape[1] != 384 or not np.isfinite(b).all():
        raise ValueError('base must be finite FP32 [sequence_length,384] from the frozen native projection')
    if image.dtype != np.bool_ or text.dtype != np.bool_ or image.shape != (len(b),) or text.shape != (len(b),):
        raise ValueError('image/text masks must be Boolean [sequence_length]')
    if np.any(image & text) or not image.any() or not text.any(): raise ValueError('Both nonoverlapping roles are required')
    return np.stack([b[m].astype(np.float64).mean(0) for m in (image,text)]), np.array([image.sum(),text.sum()],np.int64)


class DecisionTable:
    """Saved scene-fixed post-hoc decisions. Never an online selector."""
    def __init__(self, library='D_C8_double', population='train', *, allow_test=False):
        if population not in ('train','test'): raise ValueError(population)
        if population == 'test' and not allow_test: raise ValueError('Test choices are diagnostic only; pass allow_test=True explicitly')
        catalog=json.loads((ROOT/'assets/CATALOG.json').read_text())
        if library not in {x['id'] for x in catalog['libraries']}: raise ValueError(library)
        self.library,self.population=library,population
        folder=ROOT/'decisions'/population/library
        self.manifest=json.loads((folder/'MANIFEST.json').read_text())
        ref=self.manifest['choice_table']; path=ROOT/ref['path']
        if _sha(path)!=ref['sha256']: raise ValueError('Decision-table SHA mismatch')
        with gzip.open(path,'rt') as f: rows=list(csv.DictReader(f))
        self.rows={r['token']:r for r in rows}
        if len(self.rows)!=len(rows) or len(rows)!=self.manifest['scenes']:raise ValueError('Invalid decision coverage')

    def row(self,token):
        if token not in self.rows: raise KeyError(f'{token} is absent from frozen {self.population} choices; no implicit Base fallback')
        return self.rows[token]

    def choice(self,token):return int(self.row(token)['choice'])


class Library:
    def __init__(self, name='D_C8_double'):
        catalog=json.loads((ROOT/'assets/CATALOG.json').read_text())
        entry=next((x for x in catalog['libraries'] if x['id']==name),None)
        if entry is None: raise ValueError(f'Unknown library {name}')
        self.config=json.loads((ROOT/entry['path']).read_text());self.name=name
        sp=_npz(self.config['space']);self.basis=sp['basis'];self.center=sp['center'];self.D0=float(sp['D0'])
        self.parameters=[_npz(s['parameters']) for s in self.config['experts']];self.networks={}

    def _coefficient(self,slot,v):
        s=self.config['experts'][slot];p=self.parameters[slot]
        # The original production predictor computes the coordinates even for
        # a constant centroid; preserve its one-row FP64 arithmetic.
        z=(v-self.center)@self.basis
        if s['method']=='centroid_D0':return p['coefficients'].copy()
        if s['method']=='local':
            distance=np.square((p['local_failed']-z)/p['local_scale']).mean(1)
            ix=np.argsort(distance,kind='stable')[:min(16,len(distance))]
            logits=-distance[ix];w=np.exp(logits-logits.max());w/=w.sum()
            return w@p['local_delta'][ix]
        if s['method']!='mlp':raise ValueError(s['method'])
        import torch
        if slot not in self.networks:
            # fork_rng restores the CPU state; construction must not perturb
            # action-noise sampling in the caller.
            with torch.random.fork_rng(devices=[]):
                h=p['0.weight'].shape[0]
                net=torch.nn.Sequential(torch.nn.Linear(48,h),torch.nn.Tanh(),torch.nn.Linear(h,48))
            net.load_state_dict({k:torch.from_numpy(p[k].copy()) for k in ('0.weight','0.bias','2.weight','2.bias')})
            self.networks[slot]=net.eval().requires_grad_(False)
        with torch.no_grad():
            return self.networks[slot](torch.from_numpy(((z-p['mean'])/p['scale'])[None].astype(np.float32))).numpy()[0].astype(np.float64)*float(p['D0'])

    def requests(self,role_means,counts,choices=None):
        roles=np.asarray(role_means);counts=np.asarray(counts)
        if roles.dtype!=np.float64 or roles.shape!=(2,384) or not np.isfinite(roles).all():raise ValueError('roles: FP64 [2,384]')
        if counts.dtype!=np.int64 or counts.shape!=(2,) or (counts<=0).any():raise ValueError('counts: positive int64 [2]')
        if choices is None:choices=list(range(1,self.config['K']+1))
        choices=list(choices)
        if not choices:raise ValueError('Empty choices')
        if any(not isinstance(c,(int,np.integer)) or c<0 or c>self.config['K'] for c in choices):raise ValueError(choices)
        # Exact native ordinary_view path. A simple sum or float32 mean can
        # change low bits and later change a boundary trajectory/PDMS.
        v=np.einsum('ng,ngd->nd',counts[None]/counts.sum(),roles[None])[0]
        raw=[];caps=[]
        for choice in choices:
            if choice==0:raw.append(np.zeros((2,384),np.float64));caps.append(0.);continue
            s=self.config['experts'][choice-1];q=self._coefficient(choice-1,v)
            # Preserve original decode -> alpha -> factor operation order.
            decoded=q[None]@self.basis.T
            physical=(np.stack([decoded,decoded],axis=-2)*float(s['alpha']))[0]
            raw.append(physical*np.float64(self.config['raw_shift_factor']))
            caps.append(np.float64(s['cap_D0'])*self.D0*np.float64(self.config['cap_factor']))
        raw=np.asarray(raw,np.float64)
        return dict(choices=np.array(choices,np.int64),requested_shifts_float64=raw,
            requested_shifts_float32=raw.astype(np.float32),cap_RMS_float64=np.asarray(caps,np.float64))

    def targets(self,base_interface,image,text,choices):
        """Return native FP32 full-token targets for one scene on its device.

        base_interface must come from the separately frozen RL Base, even if
        the student's own projection is being trained. No quality gating.
        """
        import torch
        from .native_cap import apply
        base=base_interface.detach()
        if base.ndim==2:base=base[None]
        image=torch.as_tensor(image,device=base.device,dtype=torch.bool).reshape(1,-1)
        text=torch.as_tensor(text,device=base.device,dtype=torch.bool).reshape(1,-1)
        roles,counts=pool_base(base[0].float().cpu().numpy(),image[0].cpu().numpy(),text[0].cpu().numpy())
        req=self.requests(roles,counts,choices)
        with torch.no_grad(),torch.autocast(base.device.type,enabled=False):
            result=apply(base,image,text,torch.from_numpy(req['requested_shifts_float32']).to(base.device),
                torch.from_numpy(req['cap_RMS_float64']).to(base.device))
        return dict(**result,**req)

    def selected_target(self,base_interface,image,text,token,decisions):
        if decisions.library!=self.name:raise ValueError('Library and decision table differ')
        return self.targets(base_interface,image,text,[decisions.choice(token)])['interfaces']
