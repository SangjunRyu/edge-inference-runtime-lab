"""Distribution summaries shared by workload runners."""
import numpy as np


def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a): return {'n':0,'p50':None,'p95':None,'max':None,'mean':None}
    return {'n':len(a),'p50':float(np.percentile(a,50)),'p95':float(np.percentile(a,95)),
            'max':float(a.max()),'mean':float(a.mean())}
