"""Opt-in upstream ReCogDrive observation contract; no global baseline changes.

Pinned reference: xiaomi-research/recogdrive@5831ee83c7a6a40bf6919715b96b0e9ba2b48335.
The historical project trimmed system-message newlines and removed fixed padding.
Planner cross-attention consumes all returned tokens, including padding positions.
"""

REFERENCE_SHA='5831ee83c7a6a40bf6919715b96b0e9ba2b48335'
CONTRACT='recogdrive_official_fixed2800_v2'


class _OfficialTokenizer:
    def __init__(self,tokenizer):object.__setattr__(self,'wrapped',tokenizer)
    def __getattr__(self,key):return getattr(self.wrapped,key)
    def __setattr__(self,key,value):setattr(self.wrapped,key,value)
    def __call__(self,*args,**kwargs):
        kwargs.pop('truncation',None)
        kwargs.update(padding='max_length',max_length=2800)
        return self.wrapped(*args,**kwargs)


def use_official_conditioning(backbone):
    from navsim.agents.recogdrive.recogdrive_backbone import NAVSIM_SYSTEM_MESSAGE
    if backbone.model_type!='internvl':raise ValueError('Official2B contract requires InternVL')
    backbone.system_message='\n'+NAVSIM_SYSTEM_MESSAGE+'\n'
    backbone.model.system_message=backbone.system_message
    if not isinstance(backbone.tokenizer,_OfficialTokenizer):backbone.tokenizer=_OfficialTokenizer(backbone.tokenizer)
    backbone._opd_observation_contract=CONTRACT
    return backbone
