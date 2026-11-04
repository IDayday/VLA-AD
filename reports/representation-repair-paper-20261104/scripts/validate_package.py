"""Validate portable writing material, aggregates, links, and size limits."""
from pathlib import Path
import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
import re
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        for k in (['href'] if tag=='a' else ['src'] if tag in ['img','script','iframe'] else []):
            if k in a:self.links.append(a[k])


def validate(root,write_manifest=False):
    root=root.resolve();files=sorted(p for p in root.rglob('*') if p.is_file())
    forbidden={'.npy','.npz','.pt','.pth','.ckpt','.safetensors','.pkl','.pickle','.tar','.gz','.zip','.bin','.log','.pyc'}
    assert files and all(p.suffix.lower() not in forbidden for p in files)
    assert all(p.stat().st_size<512*1024 for p in files),[(p.name,p.stat().st_size) for p in files if p.stat().st_size>=512*1024]
    total=sum(p.stat().st_size for p in files)
    assert total<5*1024*1024,total
    parsers={}
    for p in files:
        if p.suffix=='.html':
            h=Links();h.feed(p.read_text());assert len(h.ids)==len(set(h.ids)),p
            parsers[p]=h
    link_count=0
    def check(p,href):
        nonlocal link_count
        href=href.strip().split(' "')[0]
        s=urlsplit(href)
        if s.scheme in ['http','https','mailto','data']:return
        assert not s.path.startswith('/'),(p,href)
        target=(p.parent/unquote(s.path)).resolve() if s.path else p
        assert target.is_relative_to(root),(p,href)
        assert target.exists(),(p,href)
        if s.fragment and target.suffix=='.html':assert unquote(s.fragment) in parsers[target].ids,(p,href)
        link_count+=1
    for p,h in parsers.items():
        for href in h.links:check(p,href)
    for p in files:
        if p.suffix=='.md':
            for href in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):check(p,href)
    data=json.loads((root/'tables/figure_data.json').read_text())
    assert len(data['synthetic_curve'])==78
    assert abs(sum(r['contribution_to_total_delta'] for r in data['main_bands'])-1.9280793721007956)<1e-10
    assert len(data['ablations'])==4
    for stem in ['fig01_method_pipeline','fig02_strength_selection','fig03_core_ablations','fig04_main_results','fig05_repair_and_preservation','fig06_deployment_cost','fig07_synthetic_rl_curves']:
        for ext in ['svg','pdf','png']:assert (root/'figures'/f'{stem}.{ext}').is_file()
    manifest_path=root/'PACKAGE_MANIFEST.json'
    expected={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files if p!=manifest_path}
    if write_manifest:manifest_path.write_text(json.dumps(expected,ensure_ascii=False,indent=2))
    else:assert json.loads(manifest_path.read_text())==expected,'Package changed; review and refresh manifest.'
    result=dict(status='PASS_LIGHTWEIGHT_PORTABLE_PAPER_PACKAGE',files=len(expected)+1,
        bytes=sum(v['bytes'] for v in expected.values())+manifest_path.stat().st_size,
        largest_file_bytes=max(v['bytes'] for v in expected.values()),
        relative_links_checked=link_count,figures=7,figure_formats=['svg','pdf','png'],
        no_weights_or_raw_arrays=True,all_package_hashes_match=True)
    print(json.dumps(result,ensure_ascii=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--write-manifest',action='store_true');a=p.parse_args();validate(a.root,a.write_manifest)
