"""Export a standalone viewer of the first 24 preselected random scenes."""
import argparse,json
from pathlib import Path
import numpy as np

def main(out,report):
    from analyze import LABELS,ORDER
    tokens=json.load(open(out/'tokens.json'))[:24]
    records={r['token']:r for p in out.glob('features_shard*.json') for r in json.load(p.open())}
    data=[]
    for token in tokens:
        scene=dict(token=token,gt=np.round(np.asarray(records[token]['gt'])[:,:2],4).tolist(),models={})
        for m in ORDER:
            f=np.load(out/'predictions'/m/f'{token}.npz')
            scene['models'][m]={p:np.round(f[p][...,:2],4).tolist() for p in f.files}
        data.append(scene)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>同场景轨迹分布对比</title>
<style>body{font-family:system-ui;margin:24px;background:#f6f8fb;color:#203040}h1{font-size:23px}p{max-width:1000px;line-height:1.6}.controls{display:flex;gap:24px;align-items:center;flex-wrap:wrap;background:white;padding:16px;border-radius:10px}canvas{width:100%;background:white;margin-top:16px;border-radius:10px}select,button{padding:6px}.note{font-size:13px;color:#526270}</style>
<h1>同一场景，五个模型的轨迹分布</h1>
<p>这里展示预先随机选定的前 24 个场景。每个模型使用同一场景、同一组初始噪声；彩色细线为采样轨迹，黑色虚线为 GT。五个面板使用相同物理比例。下方的两两 ADE 越大，场景内轨迹越分散。</p>
<div class="controls"><label>场景 <select id="scene"></select></label><label>采样方式 <select id="protocol"><option value="initial_noise_only">仅初始噪声</option><option value="stochastic_ddim">随机去噪</option></select></label><label>采样条数 <select id="k"><option>16</option><option>32</option><option selected>64</option></select></label><button id="download">保存当前图像</button></div>
<canvas id="plot" width="1700" height="780"></canvas><p class="note">纵轴：向前距离 x（米）；横轴：横向距离 y（米）。轨迹越宽不代表驾驶质量越高。模型名称中的分数来自历史评测，仅用于识别权重。</p>
<script>const DATA=__DATA__, ORDER=__ORDER__, LABELS=__LABELS__, COLORS=['#777777','#2878b5','#cc4b37','#31a354','#8e63b6'];
const scene=document.getElementById('scene'),protocol=document.getElementById('protocol'),count=document.getElementById('k'),canvas=document.getElementById('plot'),ctx=canvas.getContext('2d');
DATA.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=(i+1)+' · '+s.token;scene.appendChild(o)});
function draw(){let s=DATA[+scene.value],p=protocol.value,k=+count.value;ctx.clearRect(0,0,1700,780);ctx.fillStyle='white';ctx.fillRect(0,0,1700,780);
let sets=ORDER.map(m=>s.models[m][p].slice(0,k)),all=sets.flat(2).concat(s.gt).concat([[0,0]]),xs=all.map(v=>v[0]),ys=all.map(v=>v[1]);let xmin=Math.min(...xs)-1,xmax=Math.max(...xs)+1,ymin=Math.min(...ys)-1,ymax=Math.max(...ys)+1;
let scale=Math.min(255/(ymax-ymin),585/(xmax-xmin));
sets.forEach((trajs,i)=>{let left=i*340+45,top=75,cx=left+125,cy=top+292;let point=v=>[cx+(v[1]-(ymin+ymax)/2)*scale,cy- (v[0]-(xmin+xmax)/2)*scale];
ctx.fillStyle='#203040';ctx.font='bold 19px system-ui';ctx.textAlign='center';ctx.fillText(LABELS[ORDER[i]],i*340+170,30);ctx.font='13px system-ui';ctx.fillText(s.token,i*340+170,52);
ctx.strokeStyle='#e0e5ec';ctx.lineWidth=1;let step=Math.max(1,Math.ceil((xmax-xmin)/8));for(let x=Math.ceil(xmin/step)*step;x<=xmax;x+=step){let a=point([x,ymin]),b=point([x,ymax]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.fillStyle='#6c7785';ctx.textAlign='right';ctx.fillText(x.toFixed(0),a[0]-5,a[1]+4)}
function line(tr,color,width,alpha,dash=[]){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.globalAlpha=alpha;ctx.setLineDash(dash);ctx.beginPath();tr.forEach((v,j)=>{let a=point(v);j?ctx.lineTo(...a):ctx.moveTo(...a)});ctx.stroke();ctx.globalAlpha=1;ctx.setLineDash([])}
trajs.forEach(t=>line(t,COLORS[i],1.3,.22));line(s.gt,'#111',2,1,[7,5]);let zero=point([0,0]);ctx.fillStyle='#111';ctx.beginPath();ctx.arc(...zero,3,0,Math.PI*2);ctx.fill();
let sum=0,n=0;for(let a=0;a<k;a++)for(let b=a+1;b<k;b++){let d=0;for(let t=0;t<8;t++)d+=Math.hypot(trajs[a][t][0]-trajs[b][t][0],trajs[a][t][1]-trajs[b][t][1]);sum+=d/8;n++}ctx.fillStyle=COLORS[i];ctx.textAlign='center';ctx.font='bold 17px system-ui';ctx.fillText('两两 ADE = '+(sum/n).toFixed(3)+' m',i*340+170,710);ctx.font='13px system-ui';ctx.fillText('横向范围 '+ymin.toFixed(1)+' ～ '+ymax.toFixed(1)+' m',i*340+170,740);
});}
[scene,protocol,count].forEach(x=>x.addEventListener('change',draw));document.getElementById('download').onclick=()=>{let a=document.createElement('a');a.href=canvas.toDataURL('image/png');a.download=DATA[+scene.value].token+'.png';a.click()};draw();</script></html>'''
    for key,val in [('__DATA__',data),('__ORDER__',ORDER),('__LABELS__',LABELS)]:html=html.replace(key,json.dumps(val,separators=(',',':')))
    (report/'trajectory_viewer.html').write_text(html)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();main(a.out,a.report)
