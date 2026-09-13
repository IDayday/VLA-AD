"""Plain Markdown tables without changing the shared Python environment."""
import numbers
import pandas as pd

def markdown_table(frame):
    def cell(value):
        if pd.isna(value):
            return 'NA'
        if isinstance(value,numbers.Real) and not isinstance(value,(bool,numbers.Integral)):
            text=format(value,'.5g')
        else:
            text=str(value)
        return text.replace('|','\\|').replace('\n','<br>')
    rows=[list(frame.columns),['---']*len(frame.columns)]+list(frame.itertuples(index=False,name=None))
    return '\n'.join('| '+' | '.join(cell(value) for value in row)+' |' for row in rows)
