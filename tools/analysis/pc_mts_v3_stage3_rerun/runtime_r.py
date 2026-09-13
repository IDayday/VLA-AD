"""Record exact execution environment and physical GPU identities."""
from common_r import *
import torch,pytorch_lightning,transformers,subprocess
save(OUT/'manifests'/f'runtime_{socket.gethostname()}.json',dict(identity(),host=socket.gethostname(),python=sys.version,executable=sys.executable,
     torch=torch.__version__,cuda=torch.version.cuda,lightning=pytorch_lightning.__version__,transformers=transformers.__version__,numpy=np.__version__,
     devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).splitlines()))
