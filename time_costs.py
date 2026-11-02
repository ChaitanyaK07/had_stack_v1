import os; os.environ["CUDA_VISIBLE_DEVICES"]=""
import time, json, numpy as np, torch
import baseline_model as bm, adversarial_game as ag, had_stack_v2 as hs, eval_extras as ex, run_confirmation as rc
torch.set_num_threads(1); bm.set_seed(42)
T={}
tr, va, te, nf = bm.load_processed_data()
def fresh():
    m=bm.TabularTransformer(nf,d_model=64,n_heads=4,n_layers=2,d_ffn=128,dropout=0.4); m.load_state_dict(torch.load("models/baseline_best.pt",map_location="cpu")); return m
# 1. baseline: one training epoch
m=fresh(); opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-2); crit=bm.FocalLoss(2.0,0.75)
t=time.perf_counter(); bm.train_epoch(m,tr,opt,crit); T["baseline_epoch_s"]=time.perf_counter()-t
# 2. Hamiltonian resolve (H1, H2, H29 complex spectra)
b=hs.HamiltonianBudgetV3(); t=time.perf_counter()
for _ in range(20): b.resolve(-0.5)
T["hamiltonian_resolve_ms"]=1000*(time.perf_counter()-t)/20
# 3. one adversarial training epoch: HAD-Stack v3 and FB-AT (identical trainer, different budget object)
for name,bud in [("v3",hs.HamiltonianBudgetV3()),("fbat",hs.FixedBudget(0.6586,0.6586))]:
    m=fresh(); Tr=ag.StackelbergAdversarialTrainer(m,bud,adapt=(name=="v3"),epochs=30); Tr.adv_weight=bud.derived_lambda()
    t=time.perf_counter(); Tr.train_epoch(tr); T[f"{name}_train_epoch_s"]=time.perf_counter()-t
    sel=ag.HamiltonianAttacker(hs.FixedBudget(0.5,0.5)); t=time.perf_counter(); hs.robust_auc(m,va,sel,0.5); ag.robustness_gap_signal(m,Tr.attacker,va,None)
    T[f"{name}_validation_s"]=time.perf_counter()-t
# 4. inference on the full test set (3,959 rows)
X,y=ex.load_split("test"); m=fresh(); m.eval()
t=time.perf_counter()
for _ in range(5): ex.probs(m,X)
T["inference_test_set_ms"]=1000*(time.perf_counter()-t)/5; T["test_rows"]=len(X)
# 5. worst-case white-box attack on one model, one eps (682 fraud rows): 100 steps x 5 restarts x 2 losses
Xf=torch.tensor(X[y==1]); t=time.perf_counter()
for loss in ("bce","margin"): rc.pgd(m,Xf,1.0,loss,100,5)
T["whitebox_attack_one_eps_s"]=time.perf_counter()-t; T["fraud_rows"]=int(len(Xf))
T["threads"]=torch.get_num_threads(); T["train_rows"]=len(tr.dataset)
json.dump(T,open("logs/timing.json","w"),indent=1); print(json.dumps(T,indent=1))
