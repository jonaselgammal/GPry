"""
C1 counterfactual with progress logging: run the UltraNest acquisition on a
saved d=30 surrogate with the `-1e-300` sentinel replaced by a real -inf
stand-in, writing an eval-count line every 100k surrogate evaluations.

Needed because the fixed run does not terminate in an affordable time, while
the as-run (broken-sentinel) version finishes in seconds -- which is itself the
result.  Reference point at nlive=100 on gauss_d30_S2_ultranest_seed1:
as-run 20,596 evals / 4.4 s / 2.24 e-folds, max logL = -1e-300.

Usage: python diag_c1_cf_progress.py <run_tag> <nlive> <out.jsonl>
"""
import sys, time, json, warnings
import numpy as np
sys.path.insert(0,'/Users/jeg/Documents/GPry-NUTS/GPry/experiments/cluster/run')
import common as C
from gpry.ns_interfaces import InterfaceUltraNest
H='/Users/jeg/Documents/GPry-NUTS/results/paper_runs/h2h/'
tag=sys.argv[1]; nlive=int(sys.argv[2]); out=sys.argv[3]
sur=C.load_gp(H+tag+'_surrogate.pkl')
floor=float(np.min(sur._y[sur._i_regress]))-100.0
st={'n':0,'t0':time.time(),'next':100000}
def logp(X):
    X=np.atleast_2d(X); p=sur.minus_inf_value; sur.minus_inf_value=-1e-300
    y=sur.predict(X,return_std=False,validate=False); sur.minus_inf_value=p
    st['n']+=len(X)
    if st['n']>=st['next']:
        with open(out,'a') as f:
            f.write(json.dumps({"progress_evals":st['n'],"elapsed_s":round(time.time()-st['t0'],1)})+"\n")
        st['next']+=100000
    return np.where(y>-1e-6, floor, y)
ifc=InterfaceUltraNest(np.asarray(sur.bounds), verbosity=1)
ifc.set_precision(nlive=nlive, precision_criterion=0.01, num_repeats=5*sur.d)
t0=time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore"); ifc.run(logp,out_dir=None)
r=ifc.last_ultranest_result
with open(out,'a') as f:
    f.write(json.dumps({"DONE":True,"tag":tag,"nlive":nlive,"sentinel":"fixed",
        "wall_s":round(time.time()-t0,1),"evals":st['n'],"niter":int(r["niter"]),
        "efolds":round(r["niter"]/nlive,2),"max_logl":float(r["maximum_likelihood"]["logl"])})+"\n")
