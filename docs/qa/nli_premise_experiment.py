import sys, re; sys.path.insert(0,"src")
from rag_guard.rag.retriever import Retriever
from rag_guard.rag.generator import Generator
from rag_guard.guardrails.grounding_check import GroundingChecker
from rag_guard.guardrails.claim_extractor import extract_claims
r,g,n=Retriever(),Generator(),GroundingChecker()
STOP=set("a an the to of and or in on at for you your our is are be can will with this that it as by from if any have has please".split())
def toks(s): return {w for w in re.findall(r"[a-z']+",s.lower()) if w not in STOP and len(w)>2}
def sents(chunk):
    body=re.sub(r"^Question:.*?\nAnswer:\s*","",chunk,flags=re.S)
    parts=[p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+",body) if len(p.strip())>8]
    wins=list(parts)+[" ".join(parts[i:i+2]) for i in range(len(parts)-1)]
    return wins
def ent(premise,claim):
    p=n._pipe({"text":premise,"text_pair":claim},truncation=True,top_k=3)
    p=p if isinstance(p,list) else [p]
    return max(float(x["score"]) for x in p if n._resolve(x["label"])=="entailment")
def overlap(claim,wins):
    c=toks(claim); 
    return max((len(c&toks(w))/max(1,len(c)) for w in wins),default=0)
import time
INVENT=["A fee of $25 applies to this service.","Our branches are open on Sunday from 9 to 5.","You will receive a cash reward for completing this step.","Only a bank manager can approve this request."]
NEAR=["Click on the \"Logout\" button located at the bottom left corner of the homepage.","You must also send a notarized letter to complete the cancellation.",
 "Your card will be cancelled immediately without any verification.","Visit the nearest branch and pay a cancellation charge.","Avoid using a password longer than 6 characters.",
 "Your username must be your social security number.","Applications are approved instantly without any review.","You must pay the loan amount upfront before applying.",
 "Transfers cannot be cancelled once they are submitted.","The transfer is always free of any fees."]
def top3(claim,wins):
    c=toks(claim); return sorted(wins,key=lambda w:-len(c&toks(w)))[:3]
T={k:0 for k in ["tn","w","f","t3","tw","tf","t3s","nn","nw","nf","n3"]}
for q in ["How can I cancel my credit card?","id like to set pu a password can ya help me","How do I apply for a loan?","How can I make a transfer to someone?","How can I cancel a transfer I just made?"]:
    ch=[c.text for c in r.retrieve(q)]; a=g.generate(q,ch); claims=extract_claims(a)
    wins=[w for c in ch for w in sents(c)]
    t0=time.time(); whole=[max(ent(c_,cl) for c_ in ch) for cl in claims]; T["tw"]+=time.time()-t0
    t0=time.time(); full=[max(ent(x,cl) for x in wins) for cl in claims]; T["tf"]+=time.time()-t0
    t0=time.time(); t3=[max(ent(x,cl) for x in top3(cl,wins)) for cl in claims]; T["t3s"]+=time.time()-t0
    T["tn"]+=len(claims); T["w"]+=sum(x>=.4 for x in whole); T["f"]+=sum(x>=.4 for x in full); T["t3"]+=sum(x>=.4 for x in t3)
    for cl in INVENT+NEAR:
        T["nn"]+=1; T["nw"]+=max(ent(c_,cl) for c_ in ch)>=.4
        T["nf"]+=max(ent(x,cl) for x in wins)>=.4; T["n3"]+=max(ent(x,cl) for x in top3(cl,wins))>=.4
        if max(ent(x,cl) for x in top3(cl,wins))>=.4: print("   FALSE SUPPORT:",cl[:70],"| ctx q:",q[:30])
print({k:(round(v,1) if isinstance(v,float) else v) for k,v in T.items()})
print(f"true claims supported: whole {T['w']}/{T['tn']}  all-windows {T['f']}/{T['tn']}  top3-windows {T['t3']}/{T['tn']}")
print(f"fabricated/near-miss supported: whole {T['nw']}/{T['nn']}  all-windows {T['nf']}/{T['nn']}  top3-windows {T['n3']}/{T['nn']}")
print(f"NLI seconds over 5 answers: whole {T['tw']:.1f}  all-windows {T['tf']:.1f}  top3 {T['t3s']:.1f}")
