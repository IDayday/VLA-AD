from navsim.agents.recogdrive.opd.conditioning import _OfficialTokenizer

def test_official_padding_is_explicit_and_preserves_tokenizer_attributes():
    class Tokenizer:
        padding_side='right'
        def __call__(self,*args,**kwargs):return args,kwargs
    raw=Tokenizer();wrapped=_OfficialTokenizer(raw);wrapped.padding_side='left'
    args,kwargs=wrapped(['real prompt'],padding=True,truncation=False,return_tensors='pt')
    assert raw.padding_side=='left'
    assert args==(['real prompt'],)
    assert kwargs==dict(padding='max_length',max_length=2800,return_tensors='pt')
