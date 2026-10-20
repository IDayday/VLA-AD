"""Admission to the single locked final evaluation; no model selection here."""


def validate_final_evaluation(lock, *, name, checkpoint_sha256, seeds,
                              screening, limit, population_sha256,
                              protocol_sha256):
    if lock.get('status') != 'LOCKED':
        raise ValueError('Final evaluation configuration is not locked')
    if lock.get('protocol_sha256') != protocol_sha256:
        raise ValueError('Final evaluation protocol identity changed')
    if lock.get('navtest_manifest_sha256') != population_sha256:
        raise ValueError('Final evaluation population changed')
    if limit is not None:
        raise ValueError('Final evaluation requires the complete locked population')
    candidates = [item for item in lock.get('candidates', []) if item['name'] == name]
    if len(candidates) != 1:
        raise ValueError('Evaluation name is not a unique locked final candidate')
    candidate = candidates[0]
    if candidate['checkpoint_sha256'] != checkpoint_sha256:
        raise ValueError('Checkpoint is not the locked final candidate')
    if bool(candidate.get('screening', False)) != screening:
        raise ValueError('Screening and final inference roles cannot be exchanged')
    if list(candidate['seeds']) != list(seeds):
        raise ValueError('Final inference seeds changed')
    return candidate
