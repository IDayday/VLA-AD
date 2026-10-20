"""Executed DiT invocation and processed-state accounting, no tensor mutation."""
from collections import defaultdict


class ComputeLedger:
    def __init__(self,**heads):
        self.values=defaultdict(lambda:dict(forward_calls=0,trajectory_states=0))
        self.handles=[]
        for name,head in heads.items():
            if head is None:continue
            def count(module,args,kwargs,label=name):
                hidden=kwargs.get('hidden_states',args[0] if args else None)
                if hidden is None:raise ValueError('Unrecognized native DiT input')
                self.values[label]['forward_calls']+=1
                self.values[label]['trajectory_states']+=len(hidden)
            self.handles.append(head.model.register_forward_pre_hook(count,with_kwargs=True))

    def state_dict(self):return {k:dict(v) for k,v in self.values.items()}

    def load_state_dict(self,state):
        for key,value in state.items():self.values[key].update(value)

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]
