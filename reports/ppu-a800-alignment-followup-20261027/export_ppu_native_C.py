"""Export ONE native no-bias convolution tensor from an already loaded PPU model.

Only the caller-selected new diagnostic NPZ is written. No model loading,
parameter/precision/attention changes, teacher search or batch generation.
"""


def export_ppu_native_C(model,pixels,destination):
    from pathlib import Path
    import hashlib
    import json
    import numpy as np
    import torch
    import torch.nn.functional as F
    def fingerprint(t):
        t=t.detach().cpu().contiguous()
        return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert pixels.shape==(9,3,448,448) and pixels.dtype==torch.bfloat16
    assert fingerprint(pixels)=='f0d5c4a3814c428ac6cca1a0ec5293c5d624fe2d5669c926cc9d3c594e7ab4ac','Wrong scene input'
    conv=model.vision_model.embeddings.patch_embedding
    assert fingerprint(conv.weight)=='425967ed2a49b8bd8b855b2e409bc62610668319252b6c19456872cf096cf1b9'
    version=conv.weight._version
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    with torch.no_grad():
        value=F.conv2d(pixels,conv.weight,None,stride=conv.stride,padding=conv.padding,dilation=conv.dilation,groups=conv.groups)
    digest=fingerprint(value)
    assert digest=='c04efdfc37e1d5d2fc0ce8bf618df389103f20a7c183e53e708eaf3dbaf2389b',{'unexpected_native_C_SHA':digest}
    assert conv.weight._version==version and flags==(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    metadata=dict(schema=1,scene='0002267a294e52a7',environment=dict(torch=str(torch.__version__),torch_git=torch.version.git_version,
        tf32_matmul=flags[0],tf32_cudnn=flags[1]),tensors=dict(native_no_bias=dict(shape=list(value.shape),dtype=str(value.dtype),sha256=digest)))
    destination=Path(destination)
    with destination.open('xb') as stream:
        np.savez_compressed(stream,native_no_bias=value.detach().cpu().contiguous().view(torch.uint8).numpy(),
            __metadata_utf8__=np.frombuffer(json.dumps(metadata).encode(),dtype=np.uint8))
    result=dict(path=str(destination),bytes=destination.stat().st_size,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),native_C_sha256=digest)
    print(json.dumps(result,ensure_ascii=False,indent=2));return result
