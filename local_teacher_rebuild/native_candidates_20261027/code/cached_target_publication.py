"""Verify controlled replacement of an owned redundant CUDA target producer."""
from pathlib import Path
import json
import hashlib

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def reference(path):return {'path':str(Path(path).resolve()),'sha256':digest(path)}

def read(path):return json.loads(Path(path).read_text())

def checked(ref):
    p=Path(ref['path'])
    if digest(p)!=ref['sha256']:raise ValueError('Controlled target publication reference changed')
    return p

def validate_publication(manifest,results):
    manifest=Path(manifest);v=read(manifest.parent/'ADOPTION.json')
    if (v.get('schema')!='nrkd_cached_actual_target_publication_v1' or v.get('status')!='COMPLETE' or
        v.get('published_manifest')!=reference(manifest) or not v.get('all_previous_writers_exited') or
        not v.get('family_and_attempt_leases_free') or not v.get('old_partial_preserved')):
        raise ValueError('Cached target publication is incomplete')
    producer=read(checked(v['producer_exit']))
    if producer.get('actual_returncode')!=0 and producer.get('actual_child_returncode')!=0:
        raise ValueError('Cached target producer did not actually exit zero')
    status=read(checked(v['superseded_status']))
    if status.get('status')!='EXITED' or status.get('returncode')!=-15:
        raise ValueError('Superseded CUDA child has no actual controlled SIGTERM exit')
    proof=read(checked(v['supersession_proof']))
    if not all(proof.get(k) is True for k in ('exact_pid_start_command_verified','pidfd_exit_observed',
                                              'wrapper_exit_observed','family_and_attempt_leases_free')):
        raise ValueError('CUDA producer ownership/exit proof is incomplete')
    if proof.get('child_pid')!=status.get('child_pid') or proof.get('child_start_ticks')!=status.get('child_start_ticks'):
        raise ValueError('CUDA producer identity changed')
    preserved=Path(v['preserved_partial_directory'])
    if not preserved.is_dir() or preserved.resolve()==manifest.parent.resolve():
        raise ValueError('Superseded partial target data was not preserved')
    checked(v['candidate_acceptance']);checked(v['superseded_spec'])
    return True
