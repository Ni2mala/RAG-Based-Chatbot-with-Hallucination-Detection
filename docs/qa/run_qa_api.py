import requests, json, time
B="http://127.0.0.1:8000"
def post(path, **kw):
    try: return requests.post(B+path, timeout=200, **kw)
    except Exception as e: return e
res=[]
def rec(id, desc, ok, got): res.append((id,desc,ok,got)); print(id, "PASS" if ok else "FAIL", got, flush=True)

r=requests.get(B+"/health"); rec("E1","GET /health",r.status_code==200,r.status_code)
r=post("/chat",json={"question":""}); rec("E2","empty question",r.status_code==422,r.status_code)
r=post("/chat",json={"question":"a"*501}); rec("E3","501 chars",r.status_code==422,r.status_code)
r=post("/chat",data="not json",headers={"Content-Type":"application/json"}); rec("E6","invalid JSON",r.status_code==422,r.status_code)
r=post("/chat",json={}); rec("E6b","missing question",r.status_code==422,r.status_code)
r=post("/admin/ingest"); rec("E9","admin no header (401 or 503)",r.status_code in (401,503),r.status_code)
r=post("/admin/ingest",headers={"x-admin-token":"wrong"}); rec("E8","admin wrong token (401 or 503)",r.status_code in (401,503),r.status_code)
t=time.time(); r=post("/chat",json={"question":"How do I block my card?","mode":"none"})
d=r.json(); rec("E5","mode:none ignored",d["verification"]["detector"]!="none",{"detector":d["verification"]["detector"],"status":d["verification"]["status"],"s":round(time.time()-t,1)})
t=time.time(); r=post("/chat",json={"question":"a"*500}); rec("E4","500 chars accepted",r.status_code==200,{"code":r.status_code,"s":round(time.time()-t,1)})
for q in ["¿Cómo bloqueo mi tarjeta?","block my card 🙂","'; DROP TABLE users;--"]:
    r=post("/chat",json={"question":q}); rec("E11",q,r.status_code==200,{"code":r.status_code,"refused":r.json().get("refused") if r.status_code==200 else None})
json.dump(res,open("/private/tmp/claude-501/-Users-manishpun08--claude-spares-spare-u6hOki/32cea381-9aad-41ad-b57a-d7af7cf6cf4e/scratchpad/qa_api.json","w"))
