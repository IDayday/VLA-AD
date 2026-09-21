from common import *

def interval(frame,column,tag='',replicates=None):
 """Scene-equal estimand; resample whole logs, preserving paired differences."""
 x=frame[['token','log',column]].dropna().groupby(['token','log'],as_index=False)[column].mean()
 if len(x)==0:return dict(mean=None,low=None,high=None,scenes=0,logs=0)
 g=x.groupby('log')[column].agg(['sum','count']);rng=np.random.default_rng(seed('bootstrap',cfg()['seed'],tag,column));n=len(g)
 idx=rng.integers(n,size=(replicates or cfg()['bootstrap_replicates'],n));v=g['sum'].to_numpy()[idx].sum(1)/g['count'].to_numpy()[idx].sum(1)
 return dict(mean=float(x[column].mean()),low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975)),scenes=len(x),logs=n)

def summarise(frame,columns,by,tag):
 rows=[]
 for keys,b in frame.groupby(by):
  if not isinstance(keys,tuple):keys=(keys,)
  for col in columns:rows.append(dict(zip(by,keys),metric=col,**interval(b,col,tag+str(keys))))
 return pd.DataFrame(rows)
