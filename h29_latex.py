import re, numpy as np
src=open('final_hamiltonian_solver.py').read()
body=src[src.index('    def construct_full_hamiltonian(self, field):'):src.index('return H_full.real')]
def inner(name):
    i=body.index(name+' = ('); j=body.index('(\n', i)+1; d=1; k=j
    while d: k+=1; d+= {'(':1,')':-1}.get(body[k],0)
    return re.sub(r'\s+',' ',body[j:k]).strip()
B={k:inner(k) for k in ['coupling_E1','coupling_E1_conj','diss_eta','diss_no_eta']}
def to_tex(s):
    s=s.replace('np.eye(16, dtype=complex)','I')
    s=re.sub(r'self\.s(\d)([xyz])', r'\\sigma_{\1}^{\2}', s)
    s=s.replace(' @ ', r'\,').replace(' ','').replace('1j*','i').replace('*','')
    s=re.sub(r'(\d)j', r'\1i', s)
    return s.replace('-','\\,-\\,').replace('+','\\,+\\,')
def from_tex(t):                      # inverse, for verification only
    t=t.replace('\\,+\\,','+').replace('\\,-\\,','-')
    t=re.sub(r'\\sigma_\{(\d)\}\^\{([xyz])\}', r'S.s\1\2', t).replace('\\,','@')
    t=re.sub(r'(\d)i', r'\1j', t); t=re.sub(r'(?<![\w.])i(?=S\.|\()', '1j*', t)
    t=re.sub(r'(?<=[\dj\)])(?=S\.|I\b|\()', '*', t); t=re.sub(r'(?<=[\dj\)])(?=I)', '*', t)
    return t.replace('I', 'np.eye(16)')
ns={}; exec(src.split("def main():")[0], ns); S=ns['Eq29Solver'](dict(ns['PARAMS_EQ29']))
ok=True
for k,s in B.items():
    ref=eval(s.replace('self.','S.').replace('np.eye(16, dtype=complex)','np.eye(16)'),{'S':S,'np':np})
    tex=to_tex(s); back=eval(from_tex(tex),{'S':S,'np':np})
    good=np.allclose(ref,back,atol=1e-12); ok&=good
    print(f"{k:18s} terms={s.count('self.s0')+s.count('self.s1')+s.count('self.s2')+s.count('self.s3')} (sigma factors)  round-trip matches solver: {good}")
# whole operator: energy + both couplings (with field and E1 factors) + both dissipation blocks, vs solver
p=S.params; f=0.37
E=-p['epsilon_0']/2*(S.s0z-S.s2z)-p['epsilon_1']/2*(S.s1z@S.s0z+S.s3z@S.s2z@S.s1z)
ev=lambda k: eval(from_tex(to_tex(B[k])),{'S':S,'np':np})
H=(E+1j*p['g_1']*p['h']*f/4*ev('coupling_E1')*p['E_1']+1j*p['g_1']*p['h']*f/4*ev('coupling_E1_conj')*np.conj(p['E_1'])
   +p['gamma_1']/16*p['eta_r']*ev('diss_eta')+p['gamma_1']/16*ev('diss_no_eta'))
full=np.allclose(H.real, S.construct_full_hamiltonian(f), atol=1e-12); ok&=full
print("full H29 rebuilt from the LaTeX == solver matrix:", full)
open('logs/h29_tex.txt','w').write("\n\n".join(f"%{k}\n"+to_tex(B[k]) for k in B))
print("ALL OK" if ok else "MISMATCH"); print(to_tex(B['coupling_E1']))
