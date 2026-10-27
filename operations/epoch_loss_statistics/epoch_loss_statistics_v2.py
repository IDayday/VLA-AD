"""CPU-only, read-only training-log summaries and sparse durable ETA scheduling.

User override V2: retain20261027 only; seed20261028 is cancelled.
No production imports, GPU discovery, scorer, training writes or qualification.
All moments describe logged UPDATE MEANS, not unobserved per-scene variance.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import time
import uuid

ARMS = ('H_full', 'H_gain1', 'H_mu_teacher', 'H_mu_student')
RUNS = tuple(f'{arm}_seed{seed}' for arm in ARMS for seed in (20261027,))
UPDATES, SCENES, BATCH, EPOCHS = 1341, 85813, 64, 40

def utc(value=None):
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Explicit UTC/timezone required')
    return parsed.astimezone(timezone.utc)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with temp.open('w') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected finite numeric metric')
    return float(value)

def moment(pairs):
    pairs = [(number(x), number(w)) for x, w in pairs if w > 0]
    weight = math.fsum(w for _, w in pairs)
    if not weight:
        return {'mean': None, 'min': None, 'max': None, 'std': None, 'weight': 0, 'n': 0}
    mean = math.fsum(x * w for x, w in pairs) / weight
    variance = math.fsum(w * (x - mean) ** 2 for x, w in pairs) / weight
    return {'mean': mean, 'min': min(x for x, _ in pairs), 'max': max(x for x, _ in pairs),
            'std': math.sqrt(max(0., variance)), 'weight': weight, 'n': len(pairs)}

def load_records(path):
    """Read a bounded prefix once; only newline-terminated JSON records count."""
    path = Path(path)
    if not path.exists():
        return {}, {'path': str(path), 'missing': True, 'duplicate_global_steps': {},
                    'invalid_lines': [], 'ignored_trailing_bytes': 0}
    with path.open('rb') as f:
        info = os.fstat(f.fileno())
        data = f.read(info.st_size)
    boundary = data.rfind(b'\n') + 1
    records, counts, invalid = {}, Counter(), []
    for line_no, line in enumerate(data[:boundary].splitlines(), 1):
        try:
            row = json.loads(line)
            step = row['global_step']
            if type(step) is not int or step < 1:
                raise ValueError('Invalid global_step')
            counts[step] += 1
            records[step] = row  # resumed steps: latest complete JSON line wins
        except (ValueError, KeyError, TypeError) as exc:
            invalid.append({'line': line_no, 'error': str(exc)})
    return records, {'path': str(path.resolve()), 'bounded_size': len(data),
                     'source_size_at_open': info.st_size, 'source_mtime_ns': info.st_mtime_ns,
                     'sha256': hashlib.sha256(data).hexdigest(), 'invalid_lines': invalid,
                     'ignored_trailing_bytes': len(data) - boundary,
                     'duplicate_global_steps': {str(k): v for k, v in sorted(counts.items()) if v > 1},
                     'duplicate_records_discarded': sum(v - 1 for v in counts.values())}

def checkpoint_status(folder, epoch, updates=UPDATES, scenes=SCENES):
    checkpoint = Path(folder) / f'epoch_{epoch:03d}.pt'
    marker = Path(str(checkpoint) + '.COMPLETE.json')
    if not marker.exists():
        return {'status': 'CHECKPOINT_PENDING', 'checkpoint': str(checkpoint), 'marker': str(marker)}
    try:
        value = read(marker)
        expected = {'global_step': epoch * updates, 'epoch_index': epoch,
                    'global_batch_cursor': 0, 'real_scenes_seen': epoch * scenes}
        valid = (value.get('status') == 'COMPLETE' and checkpoint.is_file()
                 and value.get('global_step') == epoch * updates
                 and all(value.get('progress', {}).get(k) == v for k, v in expected.items())
                 and value.get('sha256') == sha(checkpoint))
        return {'status': 'PASS' if valid else 'CHECKPOINT_INVALID', 'checkpoint': str(checkpoint),
                'marker': str(marker), 'marker_sha256': sha(marker), 'checkpoint_sha256': value.get('sha256')}
    except (OSError, ValueError, KeyError) as exc:
        return {'status': 'CHECKPOINT_INVALID', 'error': str(exc), 'marker': str(marker)}

def phase(run, step):
    if not run.startswith(('H_mu_teacher_', 'H_mu_student_')):
        return 'H_only'
    update = step - 1
    return 'H_only' if update < 5364 else 'ramp' if update < 8046 else 'joint'

def summarize_records(run, records, checkpoint_dir, *, updates=UPDATES, scenes=SCENES, batch=BATCH):
    epochs = defaultdict(list)
    for step, row in sorted(records.items()):
        epochs[(step - 1) // updates + 1].append(row)
    result = []
    for epoch, rows in sorted(epochs.items()):
        steps = [r['global_step'] for r in rows]
        expected = list(range((epoch - 1) * updates + 1, epoch * updates + 1))
        errors = []
        good = []
        for row in rows:
            step = row['global_step']
            try:
                size = row['real_scenes']
                if type(size) is not int or size <= 0:
                    raise ValueError('invalid real_scenes')
                cursor = (step - 1) % updates + 1
                expected_size = min(batch, scenes - (cursor - 1) * batch)
                if size != expected_size:
                    raise ValueError(f'expected real_scenes={expected_size}, got {size}')
                for key, val in [('epoch', epoch), ('epoch_index', epoch - 1), ('global_batch_cursor', cursor),
                                 ('real_scenes_seen', (epoch - 1) * scenes + min(cursor * batch, scenes))]:
                    if key in row and row[key] != val:
                        raise ValueError(f'{key} disagrees with global_step')
                for key in ('loss', 'h', 'mu', 'zero_h', 'repair_h', 'lr', 'grad_norm', 'grad_clip_ratio',
                            'cap_count', 'actual_rms', 'step_seconds'):
                    number(row[key])
                if row['step_seconds'] < 0 or row['grad_norm'] < 0 or not 0 <= row['grad_clip_ratio'] <= 1:
                    raise ValueError('invalid time/gradient metric')
                for key in ('zero_count', 'repair_count', 'nonfinite_gradients', 'transition_scene_count'):
                    if type(row[key]) is not int or row[key] < 0:
                        raise ValueError('invalid ' + key)
                if row['zero_count'] + row['repair_count'] != size or row['transition_scene_count'] > size:
                    raise ValueError('scene category counts disagree')
                groups = row['source_groups']
                if sum(number(pair[1]) for pair in groups.values()) != size:
                    raise ValueError('source_groups do not cover actual scene count')
                for pair in groups.values():
                    number(pair[0])
                if len(row['mu_by_step']) != 5:
                    raise ValueError('mu_by_step needs five [sum,count] pairs')
                for pair in row['mu_by_step']:
                    number(pair[0]); number(pair[1])
                    if pair[1] < 0:
                        raise ValueError('negative time-step count')
                good.append(row)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                errors.append({'global_step': step, 'error': str(exc)})
        samples = sum(r['real_scenes'] for r in good)
        good_steps = [r['global_step'] for r in good]
        missing = sorted(set(expected) - set(good_steps))
        log_complete = not errors and good_steps == expected and samples == scenes
        cp = checkpoint_status(checkpoint_dir, epoch, updates, scenes) if log_complete else {'status': 'NOT_CHECKED_PARTIAL_EPOCH'}
        status = ('COMPLETE' if log_complete and cp['status'] == 'PASS' else
                  'LOGS_COMPLETE_CHECKPOINT_PENDING' if log_complete and cp['status'] == 'CHECKPOINT_PENDING' else
                  'INVALID_CHECKPOINT' if log_complete else 'PARTIAL')
        by_source = defaultdict(lambda: [0., 0.])
        by_time = [[0., 0.] for _ in range(5)]
        for row in good:
            for source, pair in row['source_groups'].items():
                by_source[source][0] += pair[0]; by_source[source][1] += pair[1]
            for j, pair in enumerate(row['mu_by_step']):
                by_time[j][0] += pair[0]; by_time[j][1] += pair[1]
        category = lambda name: moment([(r[name + '_h'], r[name + '_count']) for r in good])
        metrics = {k: moment([(r[k], r['real_scenes']) for r in good]) for k in ('loss', 'h', 'mu', 'actual_rms', 'cap_count')}
        metrics['weighted_functional_contribution'] = moment([(r['loss']-r['h'],r['real_scenes']) for r in good])
        update_metrics = {k: moment([(r[k], 1) for r in good]) for k in ('lr', 'grad_norm', 'grad_clip_ratio', 'step_seconds')}
        quarter = max(1, len(good) // 4)
        first = moment([(r['loss'], r['real_scenes']) for r in good[:quarter]])['mean']
        last = moment([(r['loss'], r['real_scenes']) for r in good[-quarter:]])['mean']
        result.append({'epoch': epoch, 'status': status, 'observed_updates': len(rows), 'valid_updates': len(good),
                       'expected_updates': updates, 'real_samples': samples, 'expected_real_samples': scenes,
                       'optimizer_step_first': min(steps), 'optimizer_step_last': max(steps),
                       'missing_global_steps': missing, 'record_errors': errors, 'metrics': metrics,
                       'zero_target_h': category('zero'), 'repair_target_h': category('repair'),
                       'update_metrics': update_metrics,
                       'source_groups': {k: {'h_sum': v[0], 'count': v[1], 'h_mean': v[0]/v[1] if v[1] else None} for k, v in sorted(by_source.items())},
                       'mu_by_step': [{'j':j,'mu_sum':s,'scheduled_scene_count':n,'mu_mean':s/n if n else None} for j,(s,n) in enumerate(by_time)],
                       'transition_scene_count': sum(r['transition_scene_count'] for r in good),
                       'phase_update_counts': dict(Counter(phase(run, r['global_step']) for r in good)),
                       'nonfinite_gradients_reported': sum(r['nonfinite_gradients'] for r in good),
                       'clipped_updates': sum(r['grad_clip_ratio'] < 1 for r in good),
                       'step_seconds_sum': math.fsum(r['step_seconds'] for r in good),
                       'first_record_time': good[0].get('time') if good else None,
                       'last_record_time': good[-1].get('time') if good else None,
                       'first_quarter_loss_mean': first, 'last_quarter_loss_mean': last,
                       'quarter_loss_change_percent': 100*(last/first-1) if first not in (0,None) and last is not None else None,
                       'checkpoint': cp})
    return result

def summarize_run(results, out, run, *, log_path=None, updates=UPDATES, scenes=SCENES):
    if run not in RUNS:
        raise ValueError('Only the four retained single-seed Round1 runs are supported')
    folder = Path(results) / run
    records, source = load_records(log_path or folder / 'logs/train_metrics.jsonl')
    epochs = summarize_records(run, records, folder / 'checkpoints', updates=updates, scenes=scenes)
    for epoch in epochs:
        epoch['duplicate_records_discarded'] = sum(count-1 for key,count in source.get('duplicate_global_steps',{}).items()
                                                   if (int(key)-1)//updates+1 == epoch['epoch'])
    value = {'schema':'nrkd_epoch_loss_statistics_v1','run':run,'training_seed':int(run.rsplit('seed',1)[1]),
             'source':source,'epochs':epochs,'complete_epochs':sum(e['status']=='COMPLETE' for e in epochs),
             'statistics_unit':'logged per-update batch means; loss/h/mu/actual_rms/cap_count use actual-scene weights; std is population std of update means, not per-scene std',
             'mu_note':'mu is the normalized functional term before curriculum weight. Actual weighted contribution is mean(loss-h), never final_lambda*mean(mu) during ramp; scheduled step counts may include the zero-weight ramp boundary; transition_scene_count records actually executed supervision',
             'subgroup_note':'zero-target/repair-target H statistics are retrospective diagnostics, not extra Base-keep objectives or training weights',
             'trend_note':'Descriptive training loss only; batch loss need not decrease monotonically, and loss trend is not policy/PDMS acceptance.'}
    dest = Path(out) / run
    atomic_json(dest / 'epoch_metrics.json', value)
    columns = ['run','epoch','status','valid_updates','real_samples','optimizer_step_first','optimizer_step_last',
               'loss_mean','loss_min','loss_max','loss_std','h_mean','mu_mean','weighted_functional_contribution_mean','zero_h_mean','repair_h_mean',
               'lr_mean','grad_norm_mean','grad_norm_max','cap_rate','nonfinite_gradients','step_seconds_sum']
    buf = io.StringIO(newline=''); writer = csv.DictWriter(buf, columns); writer.writeheader()
    for e in epochs:
        row = {k:e[k] for k in ('epoch','status','valid_updates','real_samples','optimizer_step_first','optimizer_step_last')}
        row.update(run=run, **{'loss_'+k:e['metrics']['loss'][k] for k in ('mean','min','max','std')})
        row.update(h_mean=e['metrics']['h']['mean'],mu_mean=e['metrics']['mu']['mean'],weighted_functional_contribution_mean=e['metrics']['weighted_functional_contribution']['mean'],zero_h_mean=e['zero_target_h']['mean'],
                   repair_h_mean=e['repair_target_h']['mean'],lr_mean=e['update_metrics']['lr']['mean'],
                   grad_norm_mean=e['update_metrics']['grad_norm']['mean'],grad_norm_max=e['update_metrics']['grad_norm']['max'],
                   cap_rate=e['metrics']['cap_count']['mean'],nonfinite_gradients=e['nonfinite_gradients_reported'],step_seconds_sum=e['step_seconds_sum'])
        writer.writerow(row)
    atomic_text(dest / 'epoch_metrics.csv', buf.getvalue())
    # Scheduling uses observed median UPDATE duration and explicit last timestamp.
    tail = [r for _,r in sorted(records.items())][-128:]
    durations = []
    functional_durations = []
    for row in tail:
        try:
            value_seconds=number(row['step_seconds'])
            if value_seconds>0:
                durations.append(value_seconds)
                if row.get('transition_scene_count',0)>0:functional_durations.append(value_seconds)
        except (ValueError,TypeError,KeyError):pass
    latest = max(records) if records else 0
    return value, {'last_step':latest,'last_time':records[latest].get('time') if latest else None,
                   'median_recent_step_seconds':statistics.median(durations) if durations else None,
                   'median_recent_functional_step_seconds':statistics.median(functional_durations) if functional_durations else None}

def make_plan(snapshot, results, out, *, buffer_seconds=300, now=None):
    frozen = read(snapshot); current = utc(now); entries = {}
    if set(frozen['runs']) != set(RUNS):
        raise ValueError('Initial frozen snapshot must cover exactly four retained runs')
    for run, row in frozen['runs'].items():
        source = row['log_snapshot']; p = Path(source['path'])
        if sha(p) != source['sha256']:
            raise ValueError('Frozen initial log snapshot SHA mismatch')
        records,_ = load_records(p)
        tail = [v for _,v in sorted(records.items())][-128:]
        seconds = statistics.median(number(v['step_seconds']) for v in tail)
        last = row['last_step']; epoch = (last-1)//UPDATES+1
        predicted = utc(row['latest_time']) + timedelta(seconds=(epoch*UPDATES-last)*seconds)
        entries[run] = {'next_epoch':epoch,'last_step':last,'expected_epoch_completion_utc':predicted.isoformat(),
                        'next_read_utc':max(current+timedelta(seconds=600), predicted+timedelta(seconds=buffer_seconds)).isoformat(),
                        'estimate_source':'frozen actual last128 median step_seconds; excludes future evaluation/phase-change overhead',
                        'median_step_seconds':seconds,'state':'SCHEDULED','last_read_utc':None}
    return {'schema':'nrkd_sparse_epoch_loss_schedule_v1','results_root':str(Path(results).resolve()),
            'out':str(Path(out).resolve()),'initial_snapshot':{'path':str(Path(snapshot).resolve()),'sha256':sha(snapshot)},
            'buffer_seconds':buffer_seconds,'minimum_interval_seconds':600,'runs':entries,
            'terminal_status_paths':{},'status':'SCHEDULED','created_utc':current.isoformat(),
            'phase_rule':'C/D entering epoch5: estimate at least4x recent H-only median plus1800s buffer; reestimate only at scheduled actual epoch window. No40epoch H-only extrapolation.'}

def next_window(entry, observation, summary, now, buffer_seconds):
    last = observation['last_step']; previous = entry.get('last_step',0)
    if (summary['complete_epochs'] == EPOCHS and all(e['status']=='COMPLETE' for e in summary['epochs'])
            and [e['epoch'] for e in summary['epochs']] == list(range(1,EPOCHS+1))):
        entry.update(state='ALL_40_EPOCHS_COMPLETE',next_read_utc=None); return
    if last < 1 or not observation['median_recent_step_seconds']:
        entry.update(state='WAITING_FOR_LOGS', next_read_utc=(now+timedelta(seconds=600)).isoformat()); return
    current_epoch = (last-1)//UPDATES+1
    logs_at_boundary = last % UPDATES == 0
    latest_summary = summary['epochs'][-1] if summary['epochs'] else {}
    if logs_at_boundary and latest_summary.get('status') != 'COMPLETE':
        entry.update(state='WAITING_CHECKPOINT_SEAL',next_read_utc=(now+timedelta(seconds=600)).isoformat()); return
    target = min(EPOCHS, current_epoch + int(logs_at_boundary))
    measured_functional=observation.get('median_recent_functional_step_seconds')
    seconds = measured_functional or observation['median_recent_step_seconds']; extra = buffer_seconds
    entering_functional = (summary['run'].startswith(('H_mu_teacher_','H_mu_student_'))
                           and target == 5 and not measured_functional)
    if entering_functional:
        seconds *= 4; extra = max(extra,1800)
    remaining = max(0,target*UPDATES-last)
    predicted = now + timedelta(seconds=remaining*seconds)
    delay = max(600,remaining*seconds+extra)
    # Full-Val/monitor can hold the endpoint for hours. When no new optimizer
    # step appeared, back off exponentially; do not repeatedly read every10min.
    stale = entry.get('unchanged_windows',0)+1 if last == previous else 0
    if stale:
        delay = max(delay,min(21600,600*(2**min(stale,6))))
    entry.update(state='SCHEDULED',last_step=last,next_epoch=target,
                 median_step_seconds=observation['median_recent_step_seconds'],
                 median_recent_functional_step_seconds=measured_functional,unchanged_windows=stale,
                 expected_epoch_completion_utc=predicted.isoformat(),
                 next_read_utc=(now+timedelta(seconds=delay)).isoformat(),
                 estimate_source='actual last128 median step_seconds; actual functional-row median once observed; per-run remaining epoch steps + buffer; C/D epoch5 conservative4x/1800s only before observed functional rows; stale windows back off, evaluation duration unknown')

def run_due(state_path, *, now=None):
    """A single due-window transaction. Never reads a not-due run's log."""
    state_path=Path(state_path); now=utc(now); state=read(state_path); processed=[]
    for run,entry in state['runs'].items():
        if entry.get('next_read_utc') is None or utc(entry['next_read_utc']) > now:
            continue
        value,observation=summarize_run(state['results_root'],state['out'],run)
        entry['last_read_utc']=now.isoformat()
        entry['last_summary_sha256']=sha(Path(state['out'])/run/'epoch_metrics.json')
        next_window(entry,observation,value,now,state['buffer_seconds'])
        terminal=state.get('terminal_status_paths',{}).get(run)
        if terminal and Path(terminal).exists():
            t=read(terminal)
            if (entry['state'] != 'ALL_40_EPOCHS_COMPLETE' and t.get('status')=='EXITED'
                    and type(t.get('returncode')) is int):
                entry.update(state='ACTUAL_TRAIN_PROCESS_TERMINAL',returncode=t['returncode'],
                             terminal_receipt_sha256=sha(terminal),next_read_utc=None)
        processed.append(run)
    active=[e['next_read_utc'] for e in state['runs'].values() if e.get('next_read_utc')]
    state['next_read_utc']=min(active) if active else None
    state['status']='SCHEDULED' if active else ('ALL_ACTIVE_RUNS_40_EPOCHS_COMPLETE' if all(e['state']=='ALL_40_EPOCHS_COMPLETE' for e in state['runs'].values()) else 'STOPPED_AT_ACTUAL_TERMINAL_EVENT_WITH_PARTIALS')
    if processed:
        state['last_window_utc']=now.isoformat(); atomic_json(state_path,state)
    return {'status':state['status'],'processed_runs':processed,'next_read_utc':state['next_read_utc']}

@contextmanager
def own_lease(state_path):
    with Path(str(state_path)+'.watch.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def watch(state_path):
    """Sleep until durable per-run ETA; no second/minute status/log polling."""
    with own_lease(state_path):
        state_path=Path(state_path)
        ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19])
        atomic_json(str(state_path)+'.WATCH_OWNER.json',{'pid':os.getpid(),'start_ticks':ticks,
                    'source_sha256':sha(__file__),'started_utc':utc().isoformat(),'state':str(state_path.resolve()),
                    'scope':'CPU epoch statistics only; not training or qualification'})
        try:
            while True:
                state=read(state_path)
                windows=[utc(e['next_read_utc']) for e in state['runs'].values() if e.get('next_read_utc')]
                if not windows:
                    value={'status':state['status'],'state':str(state_path),'returncode':0,'ended_utc':utc().isoformat()}
                    atomic_json(str(state_path)+'.WATCH_EXIT.json',value)
                    print(json.dumps(value),flush=True); return
                deadline=min(windows)
                print(json.dumps({'status':'WAITING_EXPLICIT_UTC_WINDOW','next_read_utc':deadline.isoformat(),
                                  'due_runs':[r for r,e in state['runs'].items() if e.get('next_read_utc') and utc(e['next_read_utc'])==deadline]}),flush=True)
                # One long blocking wait is intentional inside the detached helper.
                # The assistant does not repeatedly poll either logs or this process.
                time.sleep(max(0,(deadline-utc()).total_seconds()))
                print(json.dumps(run_due(state_path)),flush=True)
        except BaseException as exc:
            value={'status':'SIDECAR_FAILED','returncode':1,'ended_utc':utc().isoformat(),'error':repr(exc)}
            print(json.dumps(value),flush=True)
            try:atomic_json(str(state_path)+'.WATCH_EXIT.json',value)
            except OSError:pass  # quota failure must still be visible on stdout
            raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    s=sub.add_parser('summarize');s.add_argument('--results-root',type=Path,required=True);s.add_argument('--out',type=Path,required=True);s.add_argument('--runs',nargs='+',choices=RUNS,default=RUNS)
    p=sub.add_parser('plan');p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--results-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--buffer-seconds',type=int,default=300)
    for name in ('run-due','watch'):
        p=sub.add_parser(name);p.add_argument('--state',type=Path,required=True)
    args=parser.parse_args()
    if args.command in ('watch','run-due') and set(read(args.state)['runs']) != set(RUNS):
        raise ValueError('Single-seed V2 watch requires exactly the four retained20261027 runs')
    if args.command=='summarize':
        for run in args.runs:
            value,_=summarize_run(args.results_root,args.out,run)
            print(json.dumps({'run':run,'complete_epochs':value['complete_epochs']}))
    elif args.command=='plan':
        if args.state.exists():raise ValueError('Existing durable schedule must be resumed, not overwritten')
        if args.buffer_seconds<0:raise ValueError('buffer must be nonnegative')
        value=make_plan(args.snapshot,args.results_root,args.out,buffer_seconds=args.buffer_seconds)
        value['next_read_utc']=min(e['next_read_utc'] for e in value['runs'].values())
        atomic_json(args.state,value);print(json.dumps({'status':value['status'],'next_read_utc':value['next_read_utc']}))
    elif args.command=='run-due':
        with own_lease(args.state):print(json.dumps(run_due(args.state)))
    else:watch(args.state)

if __name__=='__main__':main()
