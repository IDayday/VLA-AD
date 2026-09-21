"""Supplementary strict winner vs tie distinction, no primary metric edits."""
from shared import *
from metrics import reward,safe
def main():
    rows=[]
    for row in scenes():
        for m in CFG['models']:
            s=arrays(m,row['token'],True)[:16];f=safe(s);r=reward(s);ok=f.any() and (~f).any()
            rows.append(dict(token=row['token'],model=m,strict_unsafe_winner=bool(ok and r[~f].max()>r[f].max()+1e-8),safe_unsafe_tie_at_max=bool(ok and abs(r[~f].max()-r[f].max())<=1e-8)))
    csv('reward_winner_tie_audit.csv',rows)
if __name__=='__main__':main()
