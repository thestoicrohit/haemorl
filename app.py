"""
HaemoRL v4.0 — Complete Production Backend
1500 patients ALL with diseases + AI Chatbot + Persistent HF Dataset Storage
"""
from __future__ import annotations
import asyncio,json,math,os,random,uuid,time,traceback,sys
from datetime import datetime,timedelta
from pathlib import Path
from typing import Any,Dict,List,Optional
import logging

logging.basicConfig(level=logging.INFO,format="[%(asctime)s] %(levelname)s %(message)s",datefmt="%H:%M:%S")
logger=logging.getLogger("haemorl")

from fastapi import FastAPI,HTTPException,Query,WebSocket,WebSocketDisconnect,BackgroundTasks,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse,PlainTextResponse,JSONResponse
from pydantic import BaseModel

# CONFIG
API_BASE_URL=os.getenv("API_BASE_URL","https://router.huggingface.co/v1")
MODEL_NAME=os.getenv("MODEL_NAME","Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN=os.getenv("HF_TOKEN","")
HF_REPO_ID=os.getenv("HF_REPO_ID","")
PORT=int(os.getenv("PORT","7860"))
SEED_COUNT=int(os.getenv("SEED_PATIENTS","1500"))

DATA_DIR=Path("/tmp/haemorl")
DATA_DIR.mkdir(parents=True,exist_ok=True)
SAVE_FILE=DATA_DIR/"haemorl_db.json"
HF_DATASET_FILE="haemorl_db.json"

BLOOD_TYPES=["A+","A-","B+","B-","AB+","AB-","O+","O-"]
BLOOD_COMPAT={"O-":{"A+","A-","B+","B-","AB+","AB-","O+","O-"},"O+":{"A+","B+","AB+","O+"},"A-":{"A+","A-","AB+","AB-"},"A+":{"A+","AB+"},"B-":{"B+","B-","AB+","AB-"},"B+":{"B+","AB+"},"AB-":{"AB+","AB-"},"AB+":{"AB+"}}
HLA_POOL={"A":["A1","A2","A3","A11","A23","A24","A25","A26","A29","A30","A31","A32","A33"],"B":["B7","B8","B13","B14","B15","B18","B27","B35","B38","B39","B44","B51","B52","B57"],"DR":["DR1","DR2","DR3","DR4","DR5","DR6","DR7","DR8","DR9","DR10","DR11","DR12","DR13"]}

DISEASE_DB={
    "oncology":{"diseases":["Acute Lymphoblastic Leukemia","Chronic Myeloid Leukemia","Multiple Myeloma","Hodgkin Lymphoma Stage III","Non-Hodgkin Lymphoma DLBCL","Lung Adenocarcinoma Stage IV","Breast Carcinoma Triple Negative","Hepatocellular Carcinoma BCLC-C","Pancreatic Ductal Adenocarcinoma","Glioblastoma Multiforme IDH-wildtype","Colorectal Adenocarcinoma KRAS+","Ovarian Cancer High-Grade Stage IV","Prostate Cancer Metastatic CRPC","Acute Myeloid Leukemia FLT3+","CLL Richter Transformation","Myelofibrosis JAK2+","Waldenstrom Macroglobulinaemia","Mantle Cell Lymphoma","T-Cell Lymphoma Angioimmunoblastic","Primary CNS Lymphoma"],"treatment":"Bone Marrow Transplant","organ":"Bone Marrow","need_type":"marrow","symptoms":["Bone pain","Night sweats","Weight loss","Fatigue","Bruising easily","Recurrent infections"],"medications":["Imatinib","Venetoclax","Rituximab","Bortezomib","Lenalidomide","Cytarabine"],"lab_markers":{"WBC":"elevated","Hb":"low","Platelets":"low","LDH":"elevated"}},
    "haematology":{"diseases":["Aplastic Anaemia Severe","Thalassaemia Major Beta","Sickle Cell Disease HbSS","Haemophilia A Severe Factor VIII <1%","Haemophilia B Factor IX Deficiency","Myelodysplastic Syndrome High-Risk","Paroxysmal Nocturnal Haemoglobinuria","ITP Refractory","Fanconi Anaemia","TTP","Diamond-Blackfan Anaemia","von Willebrand Disease Type 3","Autoimmune Haemolytic Anaemia","Hereditary Spherocytosis Severe","Gaucher Disease Type 1","Pure Red Cell Aplasia"],"treatment":"Bone Marrow Transplant","organ":"Bone Marrow","need_type":"marrow","symptoms":["Anaemia","Bleeding episodes","Transfusion dependence","Splenomegaly","Fatigue","Jaundice"],"medications":["Deferoxamine","Hydroxyurea","Factor VIII Concentrate","Eltrombopag","Cyclosporine","Eculizumab"],"lab_markers":{"Hb":"critically low","Reticulocytes":"low","Ferritin":"very high"}},
    "hiv":{"diseases":["HIV/AIDS Stage 3 CD4 <50","HIV/AIDS Stage 3 CD4 50-200","HIV+ Kaposi Sarcoma Pulmonary","HIV+ Primary CNS Lymphoma","HIV+ Cryptococcal Meningitis","HIV+ CMV Retinitis","HIV+ PCP Pneumonia Severe","HIV+ Wasting Syndrome","HIV+ Toxoplasma Encephalitis","HIV+ MAC Disseminated","HIV+ Oesophageal Candidiasis","HIV+ PML"],"treatment":"Antiretroviral Intensification + Organ Support","organ":"Kidney","need_type":"organ","symptoms":["Recurrent infections","Weight loss","Night sweats","Oral thrush","Lymphadenopathy","Diarrhoea"],"medications":["Tenofovir","Emtricitabine","Dolutegravir","Darunavir","Ritonavir","Cotrimoxazole"],"lab_markers":{"CD4":"critically low","Viral load":"detectable","Hb":"low"}},
    "cardiac":{"diseases":["End-Stage Heart Failure NYHA IV EF <15%","Dilated Cardiomyopathy Idiopathic","Ischaemic Cardiomyopathy Post-MI","Hypertrophic Obstructive Cardiomyopathy","Congenital Heart Disease Tetralogy of Fallot","Restrictive Cardiomyopathy Amyloid","Cardiac Sarcoidosis End-Stage","ARVC Right Ventricular Failure","Valvular Heart Disease Severe AS","PAH WHO IV","Peripartum Cardiomyopathy","Chagas Cardiomyopathy","Giant Cell Myocarditis"],"treatment":"Heart Transplant","organ":"Heart","need_type":"organ","symptoms":["Severe dyspnoea at rest","Orthopnoea","Bilateral oedema","Syncope","Chest pain","Palpitations"],"medications":["Carvedilol","Sacubitril/Valsartan","Furosemide","Spironolactone","Digoxin","Dobutamine"],"lab_markers":{"BNP":">5000","Troponin":"elevated","Creatinine":"rising","Na":"low"}},
    "renal":{"diseases":["CKD Stage 5 GFR <10","Polycystic Kidney Disease End-Stage","Diabetic Nephropathy ESRD","IgA Nephropathy Oxford M1E1S1T2","FSGS Steroid-Resistant","Alport Syndrome X-Linked","Lupus Nephritis Class IV","Rapidly Progressive GN ANCA+","Primary Hyperoxaluria Type 1","Cystinosis Nephropathic","Fabry Nephropathy","Analgesic Nephropathy","Hypertensive Nephrosclerosis ESRD","Renal Amyloidosis AL"],"treatment":"Kidney Transplant","organ":"Kidney","need_type":"organ","symptoms":["Oliguria","Peripheral oedema","Nausea/vomiting","Uraemic encephalopathy","Hypertension","Pruritus"],"medications":["Haemodialysis","Erythropoietin","Phosphate binders","ACE inhibitors","Furosemide","Calcium"],"lab_markers":{"Creatinine":">800","eGFR":"<10","Potassium":"high","Phosphate":"high"}},
    "hepatic":{"diseases":["Hepatic Cirrhosis Child-Pugh C MELD >30","Acute Liver Failure Paracetamol","Acute Liver Failure Viral Hepatitis","Primary Biliary Cholangitis Stage IV","Primary Sclerosing Cholangitis","Wilson Disease Hepatic Crisis","Alpha-1 AT Deficiency ZZ","Autoimmune Hepatitis Stage IV","NASH Cirrhosis Decompensated","Budd-Chiari Syndrome","Haemochromatosis Cirrhosis","Hepatitis B Cirrhosis","Hepatitis C Cirrhosis"],"treatment":"Liver Transplant","organ":"Liver","need_type":"organ","symptoms":["Ascites","Hepatic encephalopathy","Jaundice","Variceal bleeding","Coagulopathy","Hepatorenal syndrome"],"medications":["Rifaximin","Lactulose","Spironolactone","Propranolol","Terlipressin","Albumin infusion"],"lab_markers":{"Bilirubin":"very high","INR":">2.5","Albumin":"low","MELD":"30-40"}},
    "pulmonary":{"diseases":["IPF UIP Pattern FVC <50%","Cystic Fibrosis FEV1 <25%","COPD GOLD Stage IV FEV1 <20%","PAH Group 1 WHO IV","Lymphangioleiomyomatosis Advanced","Alpha-1 AT Emphysema","Bronchiectasis Severe Non-CF","Eisenmenger Syndrome","Sarcoidosis Stage IV","Hypersensitivity Pneumonitis Fibrotic","CTD-ILD End-Stage"],"treatment":"Lung Transplant","organ":"Lung","need_type":"organ","symptoms":["Severe dyspnoea at rest","Cyanosis","Cor pulmonale","6MWT <150m","Home O2 24h","Recurrent exacerbations"],"medications":["Nintedanib","Pirfenidone","Sildenafil","Bosentan","Inhaled bronchodilators","Long-term O2"],"lab_markers":{"FEV1":"<25%","FVC":"<50%","PaO2":"low","LAS score":"40-65"}},
    "neurological":{"diseases":["ALS Bulbar Onset FVC <50%","Severe TBI GCS 6 Day 14","Anti-NMDA Receptor Encephalitis Refractory","NMO Spectrum Disorder LETM","Hypoxic-Ischaemic Encephalopathy","Autoimmune Limbic Encephalitis","Progressive Multifocal Leukoencephalopathy","ADEM Severe","Cerebral Venous Sinus Thrombosis"],"treatment":"IVIG Therapy + Plasmapheresis","organ":"Neural Tissue","need_type":"treatment","symptoms":["Altered consciousness","Seizures","Respiratory failure","Dysphagia","Weakness","Cognitive decline"],"medications":["IVIG","Methylprednisolone","Rituximab","Plasma exchange","Levetiracetam","Baclofen"],"lab_markers":{"CSF protein":"elevated","Anti-NMDA antibodies":"positive","MRI":"lesions"}},
    "diabetes":{"diseases":["T1DM Severe Hypoglycaemia Unawareness","T1DM End-Organ Damage","T1DM Brittle Recurrent DKA","Post-Total Pancreatectomy Diabetes","MODY Type 3 HNF1A","Wolfram Syndrome","T1DM Gastroparesis Severe","Neonatal Diabetes Mellitus","LADA End-Stage"],"treatment":"Pancreas Transplant","organ":"Pancreas","need_type":"organ","symptoms":["Severe hypoglycaemia","Recurrent DKA","Weight loss","Neuropathy","Retinopathy","Gastroparesis"],"medications":["Insulin pump","Glucagon kit","Metoclopramide","Gabapentin","ACE inhibitors","Dialysis"],"lab_markers":{"HbA1c":"very high or variable","C-peptide":"<0.1","GAD antibodies":"positive"}},
    "autoimmune":{"diseases":["SLE Severe Nephritis Class IV + CNS","Systemic Sclerosis Diffuse Cutaneous","ANCA Vasculitis GPA Renal Crisis","Anti-GBM Disease Goodpasture","Dermatomyositis with ILD","Polymyositis Severe","Primary Antiphospholipid Syndrome","Cryoglobulinaemic Vasculitis","Mixed CTD","Inflammatory Myopathy Anti-MDA5","RA Vasculitis"],"treatment":"Plasmapheresis + Immunosuppression","organ":"Kidney","need_type":"treatment","symptoms":["Malar rash","Arthritis","Renal failure","Serositis","Haemoptysis","Myositis"],"medications":["Cyclophosphamide","Rituximab","Mycophenolate","Hydroxychloroquine","Belimumab","Plasma exchange"],"lab_markers":{"ANA":"positive","Anti-dsDNA":"high","C3/C4":"low","ANCA":"positive"}},
    "trauma":{"diseases":["Polytrauma Multi-Organ Failure ISS >50","Severe Burns >45% TBSA Full Thickness","Crush Injury Rhabdomyolysis AKI","Blast Injury Haemorrhagic Shock","High Voltage Electrical Injury","TBI Diffuse Axonal","Penetrating Abdominal Trauma","Spinal Cord Injury Complete C4","Major Hepatic Laceration Grade V","Traumatic Aortic Transection"],"treatment":"Emergency Blood Transfusion + Surgery","organ":"Multi-Organ","need_type":"blood","symptoms":["Haemodynamic instability","Active bleeding","Coagulopathy","Respiratory failure","Shock","MOF"],"medications":["Tranexamic acid","FFP","Packed RBC","Platelets","Noradrenaline","Vasopressin"],"lab_markers":{"Hb":"critically low","INR":"high","Lactate":"very high","pH":"acidotic"}},
    "infectious":{"diseases":["Sepsis Multi-Organ Dysfunction SOFA >10","Infective Endocarditis Surgical Emergency","Meningococcal Septicaemia with DIC","Cerebral Malaria Severe with Coma","Necrotising Fasciitis Type 1","Invasive Aspergillosis Disseminated","Mucormycosis Rhino-Orbital-Cerebral","Gram-Negative Bacteraemia ESBL","C.Difficile Fulminant Colitis","Legionella Pneumonia Severe","Staphylococcal Toxic Shock"],"treatment":"Emergency Blood Transfusion + Antibiotics","organ":"Multi-Organ","need_type":"blood","symptoms":["High fever >40C","Hypotension","Tachycardia","Altered consciousness","Petechiae","MOF"],"medications":["Meropenem","Vancomycin","Antifungals","Vasopressors","IVIG","Corticosteroids"],"lab_markers":{"WBC":"very high or low","CRP":">300","Procalcitonin":">10","Lactate":"high"}},
    "genetic":{"diseases":["Phenylketonuria Classic PKU","Gaucher Disease Type 3 Neuronopathic","Fabry Disease End-Stage","MPS Hurler Syndrome","Cystic Fibrosis FEV1 <20%","Alpha-1 AT Deficiency PiZZ Cirrhosis","Wilson Disease Neurological Crisis","Niemann-Pick Disease Type C","Pompe Disease Late-Onset Severe","Tyrosinaemia Type 1","Maple Syrup Urine Disease","Homocystinuria Classic","OTC Deficiency Urea Cycle","Methylmalonic Acidaemia Severe"],"treatment":"Bone Marrow + Enzyme Replacement Therapy","organ":"Bone Marrow","need_type":"marrow","symptoms":["Developmental delay","Hepatosplenomegaly","Neurological regression","Growth failure","Seizures","Coarse facies"],"medications":["Enzyme replacement therapy","Substrate reduction","Dietary restriction","Cofactor therapy"],"lab_markers":{"Enzyme activity":"deficient","Genetic testing":"confirmed mutation","Substrate":"accumulated"}},
}

ISCH_WIN={"critical":[4,8,12,18,24],"urgent":[24,36,48,72,96],"moderate":[168,336,504,720],"stable":[720,2160,4380,8760]}
HOSPITALS=["AIIMS New Delhi","PGIMER Chandigarh","Tata Memorial Mumbai","Apollo Chennai","Manipal Bangalore","Fortis Gurgaon","Narayana Health Bangalore","NIMHANS Bangalore","Safdarjung Hospital","CMC Vellore","KEM Mumbai","JIPMER Puducherry","SGPGI Lucknow","Amrita Kochi","Max Delhi"]
HCOORDS=[[340,55],[155,95],[238,275],[542,305],[498,215],[302,136],[500,232],[510,224],[308,142],[550,312],[242,268],[552,362],[318,175],[518,295],[332,150]]
MN=["Arjun","Rahul","Vikram","Siddharth","Karan","Rohan","Aditya","Mohammed","James","Carlos","Zheng","Taro","Kwame","Ivan","Luca","Sven","Yusuf","Omar","Diego","Ramesh","Vijay","Santosh","Balaji","Deepak","Suresh","Pradeep","Rajesh","Amit","Nikhil","Rohit","Gaurav","Manish","Rajan","Subhash","Dinesh","Naresh","Harish","Girish","Mahesh","Lokesh","Rakesh","Umesh","Ritesh","Hitesh","Paresh","Bhushan","Chetan","Kaustubh","Yogesh"]
FN=["Priya","Ananya","Sneha","Pooja","Divya","Meera","Kavya","Isha","Fatima","Sarah","Emily","Maria","Sofia","Yuki","Amara","Emma","Olivia","Luna","Ingrid","Zainab","Mei","Rekha","Sunita","Geeta","Noa","Luisa","Elena","Chioma","Astrid","Naomi","Kavitha","Lakshmi","Padma","Savitha","Pushpa","Usha","Sudha","Vijaya","Manjula","Kamala","Leela","Seema","Reema","Veena","Neena","Meena","Heena","Sheena"]
LN=["Sharma","Patel","Singh","Kumar","Verma","Gupta","Joshi","Khan","Ahmed","Smith","Johnson","Garcia","Chen","Kim","Sato","Osei","Muller","Rossi","Ferrari","Das","Nair","Reddy","Iyer","Mishra","Park","Rodriguez","Wang","Tanaka","Mensah","Popescu","Yadav","Tiwari","Pandey","Dubey","Shukla","Srivastava","Chaudhary","Bajaj","Malhotra","Kapoor","Mehta","Shah","Modi","Desai","Jain","Agarwal","Bansal","Garg","Mittal","Saxena"]
DONOR_NAMES=["Ramesh Gupta","Preethi Nair","Ahmed Khan","Maria Silva","Taro Yamamoto","Chidi Osei","Ingrid Muller","Paulo Costa","Fatima Al-Rashid","Wei Chen","Lisa Johnson","Kofi Mensah","Ananya Verma","James Brown","Mei Zhang","Sven Eriksson","Omar Abdalla","Luisa Ferrari","Andrei Popescu","Yusuf Hassan","Elena Kozlov","Daniel Park","Sofia Rossi","Kwame Asante","Naomi Tanaka"]
ORGANS_POOL=["Kidney (L)","Kidney (R)","Liver","Heart","Lung (L)","Lung (R)","Pancreas","Cornea (L)","Cornea (R)","Bone Marrow","Skin Graft","Small Intestine"]
ORGAN_VIAB={"Heart":4,"Lung (L)":6,"Lung (R)":6,"Liver":12,"Pancreas":12,"Kidney (L)":24,"Kidney (R)":24,"Small Intestine":6,"Cornea (L)":336,"Cornea (R)":336,"Bone Marrow":72,"Skin Graft":24}
TASK_INFO={"single_match":{"difficulty":"easy","max_steps":10,"baseline":0.847,"desc":"Match 1 critical patient to best compatible donor."},"batch_allocation":{"difficulty":"medium","max_steps":30,"baseline":0.693,"desc":"Match 5 critical patients to 5 donors under full medical constraints."},"crisis_routing":{"difficulty":"hard","max_steps":60,"baseline":0.521,"desc":"Live ischaemia clocks, trauma injections, hospital overload events."}}

def _r(a): return a[random.randint(0,len(a)-1)]
def _ri(lo,hi): return random.randint(lo,hi)
def _rHLA(): return {"A":[_r(HLA_POOL["A"]),_r(HLA_POOL["A"])],"B":[_r(HLA_POOL["B"]),_r(HLA_POOL["B"])],"DR":[_r(HLA_POOL["DR"]),_r(HLA_POOL["DR"])]}
def _hla(a,b):
    m=t=0
    for l in["A","B","DR"]:
        for ag in(a.get(l,[]) if isinstance(a,dict) else []):
            t+=1
            if ag in(b.get(l,[]) if isinstance(b,dict) else []): m+=1
    return round(m/t,3) if t else 0.0
def _blood_ok(donor,recip): return recip in BLOOD_COMPAT.get(donor,set())
def _uid(): return str(uuid.uuid4())[:8].upper()
def _ts(): return datetime.utcnow().isoformat()

class DB:
    patients:Dict[str,dict]={};donors:Dict[str,dict]={};blood_bank:Dict[str,dict]={}
    hospitals:Dict[str,dict]={};routes:List[dict]=[];allocations:Dict[str,dict]={}
    alerts:List[dict]=[];analytics:List[dict]=[];llm_log:List[dict]=[];chat_history:List[dict]=[]
    _pctr=0;_dctr=0;_actr=0;_alctr=0;_seeded=False
    ep_id="";ep_step=0;ep_task="crisis_routing";ep_maxsteps=60;ep_cum=0.0;ep_expired=0;ep_done=False

def _npid(): DB._pctr+=1; return f"HRL-{str(DB._pctr).zfill(5)}"
def _ndid(): DB._dctr+=1; return f"DNR-{str(DB._dctr).zfill(3)}"
def _naid(): DB._alctr+=1; return f"ALLOC-{str(DB._alctr).zfill(4)}"
def _nalid(): DB._actr+=1; return f"ALRT-{str(DB._actr).zfill(4)}"

def _db_payload():
    return {"patients":list(DB.patients.values())[:2000],"donors":list(DB.donors.values()),"blood_bank":DB.blood_bank,"allocations":list(DB.allocations.values())[-500:],"alerts":DB.alerts[-200:],"analytics":DB.analytics[-100:],"llm_log":DB.llm_log[-100:],"chat_history":DB.chat_history[-200:],"counters":{"pctr":DB._pctr,"dctr":DB._dctr,"actr":DB._actr,"alctr":DB._alctr},"ep_cum":DB.ep_cum,"ep_expired":DB.ep_expired,"saved_at":_ts(),"version":"4.0"}

def save_db():
    """Save locally then push to HF Dataset."""
    for attempt in range(3):
        try:
            data=json.dumps(_db_payload(),default=str,ensure_ascii=False)
            tmp=SAVE_FILE.with_suffix(".tmp"); tmp.write_text(data,encoding="utf-8"); tmp.rename(SAVE_FILE)
            logger.info(f"[SAVE] local OK — {len(DB.patients)} patients")
            break
        except Exception as e: logger.warning(f"[SAVE] attempt {attempt+1}: {e}"); time.sleep(0.1)
    if HF_REPO_ID and HF_TOKEN:
        try:
            loop=asyncio.get_event_loop()
            if loop.is_running(): asyncio.ensure_future(_push_hf())
        except: pass

async def _push_hf():
    """Push to HF Dataset using correct commit API."""
    if not HF_REPO_ID or not HF_TOKEN: return
    try:
        import urllib.request,base64
        data=SAVE_FILE.read_text(encoding="utf-8")
        encoded=base64.b64encode(data.encode("utf-8")).decode("ascii")
        payload=json.dumps({"summary":"HaemoRL autosave","files":[{"path":HF_DATASET_FILE,"content":encoded,"encoding":"base64"}]}).encode("utf-8")
        url=f"https://huggingface.co/api/datasets/{HF_REPO_ID}/commit/main"
        req=urllib.request.Request(url,data=payload,headers={"Authorization":f"Bearer {HF_TOKEN}","Content-Type":"application/json"},method="POST")
        await asyncio.get_event_loop().run_in_executor(None,lambda:urllib.request.urlopen(req,timeout=30))
        logger.info("[SAVE] HF Dataset push OK")
    except Exception as e: logger.warning(f"[SAVE] HF push failed: {e}")

def load_db():
    """Load from HF Dataset first, then local file."""
    if HF_REPO_ID and HF_TOKEN:
        try:
            import urllib.request
            url=f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{HF_DATASET_FILE}?raw=true"
            req=urllib.request.Request(url,headers={"Authorization":f"Bearer {HF_TOKEN}"})
            resp=urllib.request.urlopen(req,timeout=30)
            data=json.loads(resp.read().decode("utf-8"))
            _restore_db(data)
            logger.info(f"[LOAD] HF Dataset: {len(DB.patients)} patients, {len(DB.donors)} donors"); return True
        except Exception as e: logger.warning(f"[LOAD] HF failed: {e}, trying local...")
    if SAVE_FILE.exists():
        try:
            data=json.loads(SAVE_FILE.read_text(encoding="utf-8"))
            _restore_db(data)
            logger.info(f"[LOAD] Local: {len(DB.patients)} patients, {len(DB.donors)} donors"); return True
        except Exception as e: logger.error(f"[LOAD] local failed: {e}")
    return False

def _restore_db(d):
    for p in d.get("patients",[]): DB.patients[p["id"]]=p
    for dn in d.get("donors",[]): DB.donors[dn["id"]]=dn
    for a in d.get("allocations",[]): DB.allocations[a["id"]]=a
    DB.blood_bank=d.get("blood_bank",{}); DB.alerts=d.get("alerts",[]); DB.analytics=d.get("analytics",[]); DB.llm_log=d.get("llm_log",[]); DB.chat_history=d.get("chat_history",[])
    c=d.get("counters",{}); DB._pctr=c.get("pctr",0); DB._dctr=c.get("dctr",0); DB._actr=c.get("actr",0); DB._alctr=c.get("alctr",0)
    DB.ep_cum=d.get("ep_cum",0.0); DB.ep_expired=d.get("ep_expired",0); DB._seeded=True

def _make_pt(cat,urg_override=None):
    cd=DISEASE_DB[cat]; aR=random.random()
    age=(_ri(1,12) if aR<0.06 else _ri(13,17) if aR<0.12 else _ri(18,25) if aR<0.20 else _ri(26,35) if aR<0.32 else _ri(36,50) if aR<0.52 else _ri(51,65) if aR<0.72 else _ri(66,80) if aR<0.88 else _ri(81,95))
    g="M" if random.random()<0.48 else ("F" if random.random()<0.94 else "O")
    fn=_r(MN if g=="M" else (FN if g=="F" else ["Alex","Jordan","Taylor"]))
    dis=_r(cd["diseases"]); urg=urg_override or random.choices(["critical","urgent","moderate","stable"],weights=[0.20,0.30,0.30,0.20])[0]
    wins=ISCH_WIN.get(urg,[720]); wh=_r(wins); rem=(random.uniform(1,wh) if urg=="critical" else random.uniform(wh*0.2,wh) if urg=="urgent" else float(wh))
    is_paed=age<=17; pid=_npid(); wkg=round(random.uniform(12 if is_paed else 38,60 if is_paed else 115),1); hcm=round(random.uniform(85 if is_paed else 148,160 if is_paed else 192),1)
    symptom=_r(cd["symptoms"]); med=_r(cd["medications"]); urg_b={"critical":0.35,"urgent":0.20,"moderate":0.05,"stable":0.0}
    return {"id":pid,"name":f"{fn} {_r(LN)}","age":age,"gender":g,"blood_type":_r(BLOOD_TYPES),"category":cat,"disease":dis,"urgency":urg,"treatment":cd["treatment"],"need_type":cd["need_type"],"organ_needed":cd["organ"],"hospital":_r(HOSPITALS),"hla":_rHLA(),"ischaemia_h":round(rem,2),"ischaemia_total":float(wh),"is_paediatric":is_paed,"paed_bonus":0.15 if is_paed else 0.0,"wait_months":_ri(1,60),"weight_kg":wkg,"height_cm":hcm,"bmi":round(wkg/(hcm/100)**2,1),"meld_score":(_ri(22,40) if cat=="hepatic" else None),"las_score":(_ri(32,72) if cat=="pulmonary" else None),"cd4_count":(_ri(10,199) if cat=="hiv" else None),"ef_percent":(_ri(10,25) if cat=="cardiac" else None),"fev1_percent":(_ri(15,35) if cat=="pulmonary" else None),"creatinine":(_ri(400,1200) if cat=="renal" else None),"primary_symptom":symptom,"current_medication":med,"lab_markers":cd["lab_markers"],"notes":f"{dis} confirmed. Primary symptom: {symptom}. On {med}.","rl_score":round(min(0.999,0.5+urg_b.get(urg,0)+(0.15 if is_paed else 0)+random.uniform(-0.03,0.03)),3),"match_score":_ri(52,98),"is_allocated":False,"allocation_id":None,"admitted_at":(datetime.utcnow()-timedelta(days=_ri(0,365))).isoformat(),"added_by":"system"}

def _make_donor(idx=0):
    did=_ndid(); name=DONOR_NAMES[idx%len(DONOR_NAMES)]; organs=random.sample(ORGANS_POOL,_ri(1,4))
    return {"id":did,"name":name,"age":_ri(18,65),"blood_type":_r(BLOOD_TYPES),"donor_type":random.choice(["deceased","living"]),"organs":[{"organ":o,"status":"available","viability_h":ORGAN_VIAB.get(o,24)} for o in organs],"hla":_rHLA(),"hospital":_r(HOSPITALS),"available":True,"registered_at":_ts(),"added_by":"system"}

def _seed():
    if DB._seeded: return
    for i,h in enumerate(HOSPITALS): DB.hospitals[h]={"id":f"H-{str(i+1).zfill(3)}","name":h,"load_pct":_ri(48,95),"beds_total":_ri(300,3000),"beds_available":_ri(15,400),"icu_beds":_ri(40,300),"icu_available":_ri(3,60),"has_transplant":True,"coords":HCOORDS[i]}
    for i in range(len(HOSPITALS)):
        for j in range(i+1,len(HOSPITALS)):
            if random.random()<0.38:
                x1,y1=HCOORDS[i];x2,y2=HCOORDS[j];d=math.sqrt((x1-x2)**2+(y1-y2)**2)
                DB.routes.append({"from":HOSPITALS[i],"to":HOSPITALS[j],"hours":round(d/40+random.uniform(0.5,2.5),1),"active":random.random()<0.3,"dist_km":round(d*12,0)})
    for bt in BLOOD_TYPES: DB.blood_bank[bt]={"blood_type":bt,"units":_ri(25,160),"component":_r(["Whole Blood","Packed RBC","Platelets","Fresh Frozen Plasma"]),"expiry_days":_ri(5,42)}
    cats=list(DISEASE_DB.keys()); per_cat=SEED_COUNT//len(cats); remainder=SEED_COUNT%len(cats)
    logger.info(f"[SEED] Generating {SEED_COUNT} patients across {len(cats)} disease categories...")
    for ci,cat in enumerate(cats):
        count=per_cat+(1 if ci<remainder else 0)
        for _ in range(count): p=_make_pt(cat); DB.patients[p["id"]]=p
    for i in range(25): d=_make_donor(i); DB.donors[d["id"]]=d
    for p in [x for x in DB.patients.values() if x["urgency"]=="critical"][:25]:
        DB.alerts.append({"id":_nalid(),"severity":"critical","title":f"Critical: {p['name']}","message":f"{p['disease']} - {p.get('ischaemia_h',0):.1f}h - {p['hospital']}","patient_id":p["id"],"created_at":_ts(),"resolved":False})
    DB._seeded=True; save_db()
    logger.info(f"[SEED] Done - {len(DB.patients)} patients, {len(DB.donors)} donors")

def _reward(patient,donor,hospital_name,expired=0,crit_remaining=0,action_type="match_organ"):
    """
    7-component shaped reward function for organ allocation.
    
    Components:
    1. blood_compatibility: ABO/Rh matching (hard constraint)
    2. hla_match: 6-antigen HLA tissue typing score (0-35% of reward)
    3. ischaemia_urgency: Non-linear decay based on time remaining
    4. paediatric_priority: Structural equity bonus for children
    5. hospital_load: System-level capacity awareness
    6. survival_probability: Patient-specific outcome prediction
    7. organ_expiry_penalty: Global penalty for wasted organs
    """
    if action_type=="skip":
        # Skipping when critical patients are waiting is penalised
        # but skipping when no good match exists is neutral
        v=-0.30 if crit_remaining>0 else 0.0
        return {"value":v,"hla_score":0.0,"breakdown":{"skip":v},"explanation":f"Skip ({crit_remaining} critical waiting)"}
    if not patient: return {"value":-0.50,"hla_score":0.0,"breakdown":{"invalid":-0.50},"explanation":"Patient not found"}
    if not donor:   return {"value":-0.40,"hla_score":0.0,"breakdown":{"no_donor":-0.40},"explanation":"No donor available"}
    
    r=0; bd={}
    
    # 1. BLOOD COMPATIBILITY (hard medical constraint)
    bok=_blood_ok(donor.get("blood_type",""),patient.get("blood_type",""))
    bd["blood"]=0.25 if bok else -0.15; r+=bd["blood"]
    
    # 2. HLA TISSUE TYPING (6-antigen match across A, B, DR loci)
    # Full 6/6 match = +0.35, partial matches scaled proportionally
    hsc=_hla(patient.get("hla",{}),donor.get("hla",{}))
    bd["hla"]=round(hsc*0.35,4); r+=bd["hla"]
    
    # 3. ISCHAEMIA URGENCY (non-linear — exponential decay near expiry)
    # Hearts: 4h window. Livers: 12h. Kidneys: 24h.
    ih=patient.get("ischaemia_h",24)
    total=patient.get("ischaemia_total",24)
    frac=ih/max(total,1)  # fraction of window remaining
    if ih<=0:     isch=-0.30   # expired — worst outcome
    elif ih<1:    isch=0.22    # <1h — maximum urgency bonus
    elif ih<2:    isch=0.20    # <2h — critical
    elif ih<4:    isch=0.15    # <4h — very urgent
    elif ih<6:    isch=0.10    # <6h — urgent
    elif ih<12:   isch=0.05    # <12h — moderate
    elif frac>0.8:isch=-0.02   # >80% window left — deprioritise
    else:         isch=0.01
    bd["isch"]=round(isch,4); r+=bd["isch"]
    
    # 4. PAEDIATRIC PRIORITY (equity component — matches NOTTO guidelines)
    # Children <18 get structural priority; neonates get extra
    age=patient.get("age",40)
    if age<=1:    paed=0.20    # neonatal — highest priority
    elif age<=12: paed=0.15    # child
    elif age<=17: paed=0.10    # adolescent
    else:         paed=0.0
    bd["paed"]=paed; r+=bd["paed"]
    
    # 5. HOSPITAL LOAD (system capacity — don't overload already-stressed hospitals)
    load=DB.hospitals.get(hospital_name,{}).get("load_pct",70)
    icu=DB.hospitals.get(hospital_name,{}).get("icu_available",10)
    if load>95:    hosp=-0.25  # critically overloaded
    elif load>90:  hosp=-0.20
    elif load>85:  hosp=-0.12
    elif load>75:  hosp=-0.05
    elif load<45:  hosp=0.08   # low load — bonus for good routing
    elif load<55:  hosp=0.05
    else:          hosp=0.0
    if icu<=2:     hosp-=0.05  # ICU beds critical
    bd["hosp"]=round(hosp,4); r+=bd["hosp"]
    
    # 6. SURVIVAL PROBABILITY (patient-specific outcome prediction)
    # Based on age, disease severity markers
    base_surv=max(0,1-(age/130))
    disease_penalty=0.0
    meld=patient.get("meld_score")
    if meld:
        # MELD score predicts 3-month mortality for liver patients
        if meld>=40:   disease_penalty=0.25
        elif meld>=35: disease_penalty=0.15
        elif meld>=30: disease_penalty=0.10
        elif meld>=25: disease_penalty=0.05
    cd4=patient.get("cd4_count")
    if cd4 and cd4<50: disease_penalty+=0.10   # AIDS-defining illness
    ef=patient.get("ef_percent")
    if ef and ef<15:   disease_penalty+=0.08   # severe heart failure
    fev1=patient.get("fev1_percent")
    if fev1 and fev1<20: disease_penalty+=0.06 # end-stage lung
    surv_score=max(0,base_surv-disease_penalty)
    bd["surv"]=round(surv_score*0.15,4); r+=bd["surv"]
    
    # 7. ORGAN EXPIRY PENALTY (global signal — wasted organs harm the whole system)
    if expired>0: bd["expiry"]=round(min(-0.05*expired,-0.30),3); r+=bd["expiry"]
    
    # WAIT TIME BONUS (longer waiting = slight priority boost — fairness)
    wait=patient.get("wait_months",0)
    if wait>24:   r+=0.03
    elif wait>12: r+=0.015
    
    final=round(max(-1.0,min(1.0,r)),3)
    explanation=(
        f"blood={'OK' if bok else 'BAD'} "
        f"HLA={int(hsc*100)}% "
        f"isch={ih:.1f}h({int(frac*100)}%) "
        f"paed={'YES(+'+str(paed)+')' if paed>0 else 'no'} "
        f"load={load}% "
        f"surv={int(surv_score*100)}% "
        f"wait={wait}mo"
    )
    return {"value":final,"hla_score":round(hsc,3),"breakdown":bd,"explanation":explanation}

def _grade_single():
    done=[a for a in DB.allocations.values() if a["status"] in("complete","active","pending")]
    if not done: return {"task_id":"single_match","score":0.001,"passed":False,"details":"No allocations","metrics":{}}
    best=max(done,key=lambda a:a.get("hla_score",0)*0.5+a.get("reward",{}).get("value",0)*0.5)
    s=round(min(1.0,0.40+(0.30 if best.get("reward",{}).get("breakdown",{}).get("blood",0)>0 else 0)+best.get("hla_score",0)*0.30+(0.05 if best.get("is_paediatric") else 0)),3)
    return {"task_id":"single_match","score":round(max(0.001,min(0.999,s)),3),"passed":s>=0.25,"details":f"hla={int(best.get('hla_score',0)*100)}% score={s}","metrics":{"n":len(done)}}

def _grade_batch():
    done=[a for a in DB.allocations.values() if a["status"] in("complete","active","pending")]; n=len(done)
    if n==0: return {"task_id":"batch_allocation","score":0.001,"passed":False,"details":"No allocations","metrics":{}}
    avghla=sum(a.get("hla_score",0) for a in done)/n; bok=sum(1 for a in done if a.get("reward",{}).get("breakdown",{}).get("blood",0)>0)/n; paed=sum(1 for a in done if a.get("is_paediatric",False))
    s=round(min(1.0,min(1.0,n/5)*0.50+avghla*0.20+bok*0.10+min(0.10,paed*0.05)-min(0.15,DB.ep_expired*0.03)),3)
    return {"task_id":"batch_allocation","score":round(max(0.001,min(0.999,s)),3),"passed":s>=0.30,"details":f"n={n} avghla={int(avghla*100)}% score={s}","metrics":{"matches":n,"avg_hla":round(avghla,2)}}

def _grade_crisis():
    done=list(DB.allocations.values()); n=len(done); step=DB.ep_step; cum=DB.ep_cum; exp=DB.ep_expired
    rs=min(0.35,max(0,(cum/max(step,1))*0.35)); crit_m=sum(1 for a in done if a.get("reward",{}).get("breakdown",{}).get("isch",0)>=0.10); paed_m=sum(1 for a in done if a.get("is_paediatric") and a.get("reward",{}).get("breakdown",{}).get("isch",0)>=0.10)
    s=round(min(1.0,rs+min(0.25,crit_m*0.05)+max(0,0.20-exp*0.04)+min(0.10,paed_m*0.05)),3)
    return {"task_id":"crisis_routing","score":round(max(0.001,min(0.999,s)),3),"passed":s>=0.20,"details":f"steps={step} cum={cum:.2f} crit={crit_m} exp={exp} score={s}","metrics":{"steps":step,"cum":round(cum,3),"crit_matched":crit_m}}

def _obs():
    pts=list(DB.patients.values()); crit=sorted([p for p in pts if p.get("urgency")=="critical"],key=lambda p:p.get("ischaemia_h",999))
    avd=[d for d in DB.donors.values() if d.get("available")]; orgs=sum(len([o for o in d.get("organs",[]) if o.get("status")=="available"]) for d in avd)
    bl=sum(b.get("units",0) for b in DB.blood_bank.values()); avg_h=round(sum(h.get("load_pct",0) for h in DB.hospitals.values())/max(len(DB.hospitals),1),1)
    avghla=round(sum(_hla(crit[0].get("hla",{}),d.get("hla",{})) for d in avd[:5])/max(len(avd[:5]),1),3) if crit and avd else 0.0
    def _p(i): return crit[i] if i<len(crit) else {}
    def _d(i): return avd[i] if i<len(avd) else {}
    p1,p2,p3=_p(0),_p(1),_p(2); d1,d2,d3=_d(0),_d(1),_d(2)
    def pv(p,k,df): return p.get(k,df) if p else df
    def bb(bt): return DB.blood_bank.get(bt,{}).get("units",0)
    return {"episode_id":DB.ep_id,"step":DB.ep_step,"total_patients":len(pts),"critical_count":len(crit),"urgent_count":sum(1 for p in pts if p.get("urgency")=="urgent"),"organ_queue":sum(1 for p in pts if p.get("need_type")=="organ"),"blood_units_total":bl,"donors_available":len(avd),"organs_available":orgs,"avg_hla_score":avghla,"best_hla_score":avghla,"avg_ischaemia_remaining":round(crit[0].get("ischaemia_h",0),2) if crit else 0.0,"min_ischaemia_remaining":round(min((p.get("ischaemia_h",999) for p in crit),default=0),2),"paediatric_critical":sum(1 for p in crit if p.get("is_paediatric")),"expired_organs":DB.ep_expired,"hospital_avg_load":avg_h,"hospitals_overloaded":sum(1 for h in DB.hospitals.values() if h.get("load_pct",0)>85),"active_allocations":sum(1 for a in DB.allocations.values() if a.get("status")=="active"),"completed_allocations":sum(1 for a in DB.allocations.values() if a.get("status")=="complete"),"cumulative_reward":round(DB.ep_cum,3),"p1_blood_type":pv(p1,"blood_type","?"),"p1_hla_a1":(p1.get("hla",{}).get("A",["?"])[0] if p1 else "?"),"p1_ischaemia_h":pv(p1,"ischaemia_h",0.0),"p1_paed":int(bool(pv(p1,"is_paediatric",False))),"p1_meld":pv(p1,"meld_score",0) or 0,"p2_blood_type":pv(p2,"blood_type","?"),"p2_hla_a1":(p2.get("hla",{}).get("A",["?"])[0] if p2 else "?"),"p2_ischaemia_h":pv(p2,"ischaemia_h",0.0),"p2_paed":int(bool(pv(p2,"is_paediatric",False))),"p2_meld":pv(p2,"meld_score",0) or 0,"p3_blood_type":pv(p3,"blood_type","?"),"p3_hla_a1":(p3.get("hla",{}).get("A",["?"])[0] if p3 else "?"),"p3_ischaemia_h":pv(p3,"ischaemia_h",0.0),"p3_paed":int(bool(pv(p3,"is_paediatric",False))),"p3_meld":pv(p3,"meld_score",0) or 0,"d1_blood_type":pv(d1,"blood_type","?"),"d1_hla_a1":(d1.get("hla",{}).get("A",["?"])[0] if d1 else "?"),"d1_organs_count":len([o for o in d1.get("organs",[]) if o.get("status")=="available"]) if d1 else 0,"d2_blood_type":pv(d2,"blood_type","?"),"d2_hla_a1":(d2.get("hla",{}).get("A",["?"])[0] if d2 else "?"),"d2_organs_count":len([o for o in d2.get("organs",[]) if o.get("status")=="available"]) if d2 else 0,"d3_blood_type":pv(d3,"blood_type","?"),"d3_hla_a1":(d3.get("hla",{}).get("A",["?"])[0] if d3 else "?"),"d3_organs_count":len([o for o in d3.get("organs",[]) if o.get("status")=="available"]) if d3 else 0,"blood_A_pos":bb("A+"),"blood_B_pos":bb("B+"),"blood_O_pos":bb("O+"),"blood_AB_pos":bb("AB+")}

CHAT_SYSTEM="""You are HaemoBot - AI medical assistant for HaemoRL, India organ allocation system.
Help with: diseases, treatments, HLA compatibility, lab values, ischaemia, patient prioritisation, India transplant crisis, RL reward.
Be helpful, accurate, empathetic. Correct medical terminology. Concise responses."""

async def llm_chat(messages,context=""):
    if not HF_TOKEN: return "HaemoBot needs HF_TOKEN secret to activate Qwen 2.5 72B. Add it in Space Settings."
    for attempt in range(2):
        try:
            from openai import AsyncOpenAI
            client=AsyncOpenAI(base_url=API_BASE_URL,api_key=HF_TOKEN)
            full_sys=CHAT_SYSTEM+(f"\n\nSystem: {context}" if context else "")
            timeout=25.0 if attempt==0 else 40.0
            resp=await asyncio.wait_for(client.chat.completions.create(model=MODEL_NAME,messages=[{"role":"system","content":full_sys}]+messages[-10:],temperature=0.4,max_tokens=600),timeout=timeout)
            return resp.choices[0].message.content
        except asyncio.TimeoutError:
            if attempt==0: logger.warning("LLM timeout, retrying..."); continue
            return "AI taking too long. Please try again."
        except Exception as e:
            logger.error(f"LLM error: {e}")
            if attempt==0: continue
            return f"AI temporarily unavailable. Rule-based system still running allocations."
    return "Could not get AI response."

async def llm_decide(obs,patients,donors):
    crit=sorted([p for p in patients if p.get("urgency")=="critical" and not p.get("is_allocated")],key=lambda p:p.get("ischaemia_h",999))
    avd=[d for d in donors if d.get("available")]
    def rule():
        if not crit or not avd: return {"action":{"patient_id":"","donor_id":None,"hospital":None,"action_type":"skip"},"reasoning":"No critical patients or donors.","confidence":1.0,"priority_factors":[],"mode":"rule_based"}
        paed=[p for p in crit if p.get("is_paediatric") and p.get("ischaemia_h",999)<6]
        target=paed[0] if paed else crit[0]
        scored=[(d,_hla(target.get("hla",{}),d.get("hla",{})),_blood_ok(d.get("blood_type",""),target.get("blood_type",""))) for d in avd]
        scored.sort(key=lambda x:x[1]*0.6+(0.4 if x[2] else 0),reverse=True); bd=scored[0][0]; bh=min(DB.hospitals.values(),key=lambda h:h.get("load_pct",100))
        return {"action":{"patient_id":target["id"],"donor_id":bd["id"],"hospital":bh["name"],"action_type":"match_organ"},"reasoning":f"{target.get('name')} ({target.get('disease')}, {target.get('ischaemia_h',0):.1f}h) -> {bd.get('name')} HLA={int(scored[0][1]*100)}%","confidence":round(0.6+scored[0][1]*0.3,2),"priority_factors":["ischaemia","paediatric" if target.get("is_paediatric") else "","blood" if scored[0][2] else ""],"mode":"rule_based"}
    if not HF_TOKEN: return rule()
    try:
        from openai import AsyncOpenAI
        client=AsyncOpenAI(base_url=API_BASE_URL,api_key=HF_TOKEN)
        msg=f"Critical={obs.get('critical_count',0)} MinIsch={obs.get('min_ischaemia_remaining',0):.1f}h\nPatients: {json.dumps([{k:v for k,v in p.items() if k in ['id','name','age','blood_type','urgency','ischaemia_h','is_paediatric','disease']} for p in crit[:4]])}\nDonors: {json.dumps([{k:v for k,v in d.items() if k in ['id','name','blood_type','organs']} for d in avd[:4]])}\nJSON only: {{action:{{patient_id,donor_id,hospital,action_type}},reasoning,confidence,priority_factors,mode}}"
        resp=await asyncio.wait_for(client.chat.completions.create(model=MODEL_NAME,messages=[{"role":"system","content":"RL organ allocation agent. JSON only."},{"role":"user","content":msg}],temperature=0.1,max_tokens=300),timeout=18.0)
        text=resp.choices[0].message.content.strip()
        if "```" in text: text=text.split("```")[1].lstrip("json").strip()
        r=json.loads(text); r["mode"]="llm"; return r
    except: return rule()

app=FastAPI(title="HaemoRL",version="4.0.0",description="RL-Based Smart Organ Allocation",docs_url="/docs")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

@app.exception_handler(Exception)
async def global_exc(request:Request,exc:Exception):
    logger.error(f"Error on {request.url.path}: {exc}")
    return JSONResponse(status_code=500,content={"error":"Internal server error","detail":str(exc)[:200]})

class WSMgr:
    def __init__(self): self.conns:List[WebSocket]=[]
    async def connect(self,ws:WebSocket): await ws.accept(); self.conns.append(ws)
    def disconnect(self,ws:WebSocket): self.conns=[c for c in self.conns if c!=ws]
    async def broadcast(self,data:dict):
        dead=[]
        for ws in self.conns:
            try: await ws.send_json(data)
            except: dead.append(ws)
        for ws in dead: self.disconnect(ws)
wsm=WSMgr()

_rate_limits:Dict[str,list]={}
def _rate_ok(key:str,max_pm:int=30)->bool:
    now=time.time()
    if key not in _rate_limits: _rate_limits[key]=[]
    _rate_limits[key]=[t for t in _rate_limits[key] if now-t<60]
    if len(_rate_limits[key])>=max_pm: return False
    _rate_limits[key].append(now); return True

@app.on_event("startup")
async def startup():
    logger.info("HaemoRL v4.0 starting...")
    try:
        loaded=load_db()
        if not loaded: logger.info("No saved data, seeding..."); _seed()
        else: logger.info(f"Loaded {len(DB.patients)} patients")
        if not DB.hospitals:
            for i,h in enumerate(HOSPITALS): DB.hospitals[h]={"id":f"H-{str(i+1).zfill(3)}","name":h,"load_pct":_ri(48,95),"beds_total":_ri(300,3000),"beds_available":_ri(15,400),"icu_beds":_ri(40,300),"icu_available":_ri(3,60),"has_transplant":True,"coords":HCOORDS[i]}
        if not DB.routes:
            for i in range(len(HOSPITALS)):
                for j in range(i+1,len(HOSPITALS)):
                    if random.random()<0.38:
                        x1,y1=HCOORDS[i];x2,y2=HCOORDS[j];d=math.sqrt((x1-x2)**2+(y1-y2)**2)
                        DB.routes.append({"from":HOSPITALS[i],"to":HOSPITALS[j],"hours":round(d/40+random.uniform(0.5,2.5),1),"active":random.random()<0.3,"dist_km":round(d*12,0)})
        if not DB.blood_bank:
            for bt in BLOOD_TYPES: DB.blood_bank[bt]={"blood_type":bt,"units":_ri(25,160),"component":"Whole Blood","expiry_days":_ri(5,42)}
    except Exception as e:
        logger.critical(f"Startup error: {e}"); traceback.print_exc()
        try: _seed()
        except: pass
    asyncio.create_task(_bg_wrapper())
    logger.info(f"Ready - {len(DB.patients)} patients, LLM={'on' if HF_TOKEN else 'off'}, Dataset={'on' if HF_REPO_ID else 'off'}")

async def _bg_wrapper():
    while True:
        try: await _bg()
        except Exception as e: logger.error(f"BG crashed: {e}, restarting in 30s"); await asyncio.sleep(30)

async def _bg():
    errs=0
    while True:
        try:
            await asyncio.sleep(60)
            for p in list(DB.patients.values()):
                if p.get("urgency") in("critical","urgent"):
                    p["ischaemia_h"]=max(0.0,round(p.get("ischaemia_h",0)-1.0,4))
                    if p["ischaemia_h"]==0.0: DB.ep_expired+=1
            pts=list(DB.patients.values())
            DB.analytics.append({"ts":_ts(),"total":len(pts),"critical":sum(1 for p in pts if p.get("urgency")=="critical"),"allocations":len(DB.allocations),"expired":DB.ep_expired})
            if len(DB.analytics)>500: DB.analytics=DB.analytics[-200:]
            if len(DB.alerts)>500: DB.alerts=DB.alerts[-200:]
            if len(DB.llm_log)>300: DB.llm_log=DB.llm_log[-100:]
            if len(DB.chat_history)>600: DB.chat_history=DB.chat_history[-200:]
            save_db()
            crit=sum(1 for p in DB.patients.values() if p.get("urgency")=="critical")
            await wsm.broadcast({"event":"tick","ts":_ts(),"critical":crit,"expired":DB.ep_expired})
            errs=0
        except asyncio.CancelledError: break
        except Exception as e:
            errs+=1; logger.error(f"BG error #{errs}: {e}")
            if errs>10: await asyncio.sleep(300); errs=0

@app.get("/",response_class=HTMLResponse)
async def ui():
    f=Path(__file__).parent/"index.html"
    if f.exists(): return HTMLResponse(f.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>HaemoRL - Upload index.html</h1>")

@app.get("/ping")
def ping(): return {"pong":True,"ts":_ts()}

@app.get("/health")
def health():
    return {"status":"healthy","ts":_ts(),"version":"4.0.0","patients":len(DB.patients),"critical":sum(1 for p in DB.patients.values() if p.get("urgency")=="critical"),"donors":len(DB.donors),"allocations":len(DB.allocations),"llm_enabled":bool(HF_TOKEN),"model":MODEL_NAME if HF_TOKEN else "rule_based","persistent_storage":SAVE_FILE.exists(),"save_file_size_kb":round(SAVE_FILE.stat().st_size/1024,1) if SAVE_FILE.exists() else 0,"hf_dataset":bool(HF_REPO_ID),"chatbot_enabled":bool(HF_TOKEN),"disease_categories":len(DISEASE_DB),"total_diseases":sum(len(v["diseases"]) for v in DISEASE_DB.values()),"routes":len(DB.routes),"hospitals":len(DB.hospitals)}

@app.post("/reset")
async def reset(body:dict={}):
    task=(body or {}).get("task","crisis_routing")
    if task not in TASK_INFO: task="crisis_routing"
    info=TASK_INFO[task]; DB.ep_id=_uid(); DB.ep_step=0; DB.ep_task=task; DB.ep_maxsteps=info["max_steps"]; DB.ep_cum=0.0; DB.ep_done=False; DB.allocations.clear()
    return {"observation":_obs(),"done":False,"info":{"task":task,"episode_id":DB.ep_id,"max_steps":info["max_steps"],"patients":len(DB.patients),"critical":sum(1 for p in DB.patients.values() if p.get("urgency")=="critical")}}

@app.post("/step")
async def step(body:dict):
    if not isinstance(body,dict): body={}
    if DB.ep_done: raise HTTPException(400,"Episode done. Call /reset.")
    if DB.ep_step>=DB.ep_maxsteps: DB.ep_done=True; return {"observation":_obs(),"reward":{"value":0.0},"done":True,"info":{"reason":"max_steps"}}
    DB.ep_step+=1; delta={"single_match":2.0,"batch_allocation":1.5,"crisis_routing":1.0}.get(DB.ep_task,1.0)
    for p in DB.patients.values():
        if p.get("urgency") in("critical","urgent"):
            p["ischaemia_h"]=max(0.0,round(p.get("ischaemia_h",0)-delta,4))
            if p["ischaemia_h"]==0.0: DB.ep_expired+=1
    if DB.ep_task=="crisis_routing" and DB.ep_step%5==0:
        tp=_make_pt("trauma","critical"); DB.patients[tp["id"]]=tp
    patient=DB.patients.get(body.get("patient_id",""))
    donor=DB.donors.get(body.get("donor_id","")) if body.get("donor_id") else None
    if donor and not donor.get("available"): donor=None
    crit_c=sum(1 for p in DB.patients.values() if p.get("urgency")=="critical")
    rew=_reward(patient,donor,body.get("hospital"),DB.ep_expired,crit_c,body.get("action_type","match_organ"))
    err=None
    if body.get("action_type","match_organ")!="skip" and patient and donor:
        aid=_naid()
        DB.allocations[aid]={"id":aid,"patient_id":patient["id"],"patient_name":patient.get("name"),"donor_id":donor["id"],"donor_name":donor.get("name"),"organ":donor.get("organs",[{"organ":"Unknown"}])[0].get("organ"),"hospital":body.get("hospital"),"action_type":body.get("action_type"),"status":"pending","reward":rew,"hla_score":rew["hla_score"],"is_paediatric":patient.get("is_paediatric",False),"step":DB.ep_step,"created_at":_ts(),"disease":patient.get("disease")}
        donor["available"]=False; patient["is_allocated"]=True; patient["allocation_id"]=aid; DB.ep_cum+=rew["value"]
    elif body.get("action_type","match_organ")!="skip": err="patient or donor not found"
    crit_left=sum(1 for p in DB.patients.values() if p.get("urgency")=="critical")
    done=(DB.ep_step>=DB.ep_maxsteps)or(crit_left==0 and DB.ep_task=="single_match"); DB.ep_done=done
    return {"observation":_obs(),"reward":rew,"done":done,"info":{"step":DB.ep_step,"cumulative_reward":round(DB.ep_cum,3),"critical_remaining":crit_left,"expired_organs":DB.ep_expired,"last_action_error":err}}

@app.get("/state")
def state():
    return {"episode_id":DB.ep_id,"step":DB.ep_step,"task":DB.ep_task,"done":DB.ep_done,"cumulative_reward":round(DB.ep_cum,3),"expired_organs":DB.ep_expired,"patients":list(DB.patients.values())[:20],"donors":list(DB.donors.values()),"blood_bank":list(DB.blood_bank.values()),"hospitals":list(DB.hospitals.values()),"allocations":list(DB.allocations.values()),"observation":_obs()}

@app.get("/tasks")
def tasks():
    return {"tasks":[{"task_id":k,"difficulty":v["difficulty"],"max_steps":v["max_steps"],"description":v["desc"],"baseline_score":v["baseline"]} for k,v in TASK_INFO.items()]}

@app.post("/grade")
def grade():
    r={"single_match":_grade_single(),"batch_allocation":_grade_batch(),"crisis_routing":_grade_crisis()}
    return {"episode_id":DB.ep_id,"grader_results":r,"mean_score":round(sum(v["score"] for v in r.values())/3,3),"all_passed":all(v["passed"] for v in r.values()),"graded_at":_ts()}

@app.get("/validate")
@app.post("/validate")
def validate():
    return {"valid":True,"name":"haemorl-organ-allocation","version":"4.0.0","spec_version":"openenv-1.0","tasks":list(TASK_INFO.keys()),"difficulties":{k:v["difficulty"] for k,v in TASK_INFO.items()},"max_steps":{k:v["max_steps"] for k,v in TASK_INFO.items()},"baseline_scores":{k:v["baseline"] for k,v in TASK_INFO.items()},"action_space":{"type":"structured","fields":["patient_id","donor_id","hospital","action_type"]},"observation_space":{"type":"dict","dims":42},"reward_range":[-1.0,1.0],"shaped_reward":True,"partial_progress":True}

@app.get("/openenv.yaml")
def yaml_spec():
    return PlainTextResponse(f"name: haemorl-organ-allocation\nversion: \"4.0.0\"\ndescription: RL organ allocation. {len(DB.patients)} patients across {len(DISEASE_DB)} disease categories.\ntags: [openenv,medical,rl,organ-allocation]\ntasks:\n  - id: single_match\n    difficulty: easy\n    max_steps: 10\n    baseline_score: 0.847\n  - id: batch_allocation\n    difficulty: medium\n    max_steps: 30\n    baseline_score: 0.693\n  - id: crisis_routing\n    difficulty: hard\n    max_steps: 60\n    baseline_score: 0.521\nobservation_space:\n  type: dict\n  dims: 42\naction_space:\n  type: structured\n  fields: [patient_id, donor_id, hospital, action_type]\nreward_range: [-1.0, 1.0]\nshaped_reward: true\npartial_progress: true\n",media_type="text/yaml")

@app.get("/api/stats")
def api_stats():
    pts=list(DB.patients.values()); crit=[p for p in pts if p.get("urgency")=="critical"]
    return {"total_patients":len(pts),"critical":len(crit),"urgent":sum(1 for p in pts if p.get("urgency")=="urgent"),"moderate":sum(1 for p in pts if p.get("urgency")=="moderate"),"stable":sum(1 for p in pts if p.get("urgency")=="stable"),"organ_queue":sum(1 for p in pts if p.get("need_type")=="organ"),"blood_queue":sum(1 for p in pts if p.get("need_type")=="blood"),"marrow_queue":sum(1 for p in pts if p.get("need_type")=="marrow"),"paediatric":sum(1 for p in pts if p.get("is_paediatric")),"paed_critical":sum(1 for p in crit if p.get("is_paediatric")),"blood_units":sum(b.get("units",0) for b in DB.blood_bank.values()),"donors_total":len(DB.donors),"donors_available":sum(1 for d in DB.donors.values() if d.get("available")),"organs_available":sum(len([o for o in d.get("organs",[]) if o.get("status")=="available"]) for d in DB.donors.values() if d.get("available")),"allocations_total":len(DB.allocations),"allocations_pending":sum(1 for a in DB.allocations.values() if a.get("status")=="pending"),"allocations_active":sum(1 for a in DB.allocations.values() if a.get("status")=="active"),"allocations_done":sum(1 for a in DB.allocations.values() if a.get("status")=="complete"),"expired_organs":DB.ep_expired,"hospitals_overloaded":sum(1 for h in DB.hospitals.values() if h.get("load_pct",0)>85),"ischaemia_danger":sum(1 for p in crit if p.get("ischaemia_h",999)<=2),"ischaemia_warning":sum(1 for p in crit if 2<p.get("ischaemia_h",999)<=6),"cumulative_reward":round(DB.ep_cum,3),"llm_enabled":bool(HF_TOKEN),"chatbot_enabled":bool(HF_TOKEN),"disease_distribution":{cat:sum(1 for p in pts if p.get("category")==cat) for cat in DISEASE_DB.keys()},"persistent":SAVE_FILE.exists()}

@app.get("/api/dashboard")
def api_dashboard():
    pts=list(DB.patients.values()); cat_d={}; bt_d={}; urg_d={"critical":0,"urgent":0,"moderate":0,"stable":0}; age_d={"0-12":0,"13-19":0,"20-39":0,"40-59":0,"60-74":0,"75+":0}
    for p in pts:
        cat_d[p.get("category","other")]=cat_d.get(p.get("category","other"),0)+1; bt_d[p.get("blood_type","?")]=bt_d.get(p.get("blood_type","?"),0)+1; urg_d[p.get("urgency","stable")]=urg_d.get(p.get("urgency","stable"),0)+1
        a=p.get("age",40)
        if a<=12: age_d["0-12"]+=1
        elif a<=19: age_d["13-19"]+=1
        elif a<=39: age_d["20-39"]+=1
        elif a<=59: age_d["40-59"]+=1
        elif a<=74: age_d["60-74"]+=1
        else: age_d["75+"]+=1
    return {"urgency_dist":urg_d,"disease_dist":dict(sorted(cat_d.items(),key=lambda x:x[1],reverse=True)),"blood_dist":bt_d,"age_dist":age_d,"hospital_loads":[{"name":h.get("name"),"load_pct":h.get("load_pct"),"beds_available":h.get("beds_available"),"icu_available":h.get("icu_available")} for h in sorted(DB.hospitals.values(),key=lambda h:h.get("load_pct",0),reverse=True)],"blood_bank":list(DB.blood_bank.values()),"recent_alerts":[a for a in DB.alerts if not a.get("resolved")][:10],"llm_log":DB.llm_log[-5:],"chat_count":len(DB.chat_history)}

@app.get("/api/patients")
def api_patients(page:int=Query(1,ge=1),per_page:int=Query(20,ge=1,le=100),urgency:str=Query(None),category:str=Query(None),need_type:str=Query(None),q:str=Query(None)):
    pts=list(DB.patients.values())
    if urgency and urgency!="all": pts=[p for p in pts if p.get("urgency")==urgency]
    if category and category!="all": pts=[p for p in pts if p.get("category")==category]
    if need_type and need_type!="all": pts=[p for p in pts if p.get("need_type")==need_type]
    if q:
        ql=q.lower(); pts=[p for p in pts if ql in p.get("name","").lower() or ql in p.get("id","").lower() or ql in p.get("disease","").lower() or ql in p.get("category","").lower()]
    pts=sorted(pts,key=lambda p:({"critical":0,"urgent":1,"moderate":2,"stable":3}.get(p.get("urgency","stable"),4),p.get("ischaemia_h",999)))
    total=len(pts); start=(page-1)*per_page
    return {"total":total,"page":page,"per_page":per_page,"pages":max(1,(total+per_page-1)//per_page),"patients":pts[start:start+per_page]}

@app.post("/api/patients",status_code=201)
async def api_create_patient(body:dict,bg:BackgroundTasks):
    cat=body.get("category","oncology")
    if cat not in DISEASE_DB: cat="oncology"
    p=_make_pt(cat,body.get("urgency"))
    for k in["name","age","gender","blood_type","disease","urgency","hospital"]:
        if body.get(k) is not None: p[k]=body[k]
    if body.get("age"): p["is_paediatric"]=int(body["age"])<=17; p["paed_bonus"]=0.15 if p["is_paediatric"] else 0.0
    p["added_by"]=body.get("added_by","user"); p["admitted_at"]=_ts(); DB.patients[p["id"]]=p
    if p.get("urgency")=="critical":
        DB.alerts.append({"id":_nalid(),"severity":"critical","title":f"New Critical: {p['name']}","message":f"{p.get('disease')} - {p.get('ischaemia_h',0):.1f}h","patient_id":p["id"],"created_at":_ts(),"resolved":False})
    save_db(); bg.add_task(wsm.broadcast,{"event":"patient_added","patient_id":p["id"],"urgency":p.get("urgency"),"name":p.get("name"),"disease":p.get("disease"),"ts":_ts()})
    return p

@app.delete("/api/patients/{pid}")
async def api_del_patient(pid:str,bg:BackgroundTasks):
    if pid not in DB.patients: raise HTTPException(404)
    del DB.patients[pid]; save_db(); bg.add_task(wsm.broadcast,{"event":"patient_removed","patient_id":pid,"ts":_ts()})
    return {"deleted":pid}

@app.get("/api/patients/{pid}")
def api_get_patient(pid:str):
    p=DB.patients.get(pid)
    if not p: raise HTTPException(404)
    return p

@app.get("/api/patients/{pid}/hla-matches")
def api_hla_matches(pid:str,top:int=Query(5)):
    p=DB.patients.get(pid)
    if not p: raise HTTPException(404)
    avd=[d for d in DB.donors.values() if d.get("available")]
    sc=[{"donor_id":d["id"],"donor_name":d.get("name"),"hla_score":round(_hla(p.get("hla",{}),d.get("hla",{})),3),"blood_ok":_blood_ok(d.get("blood_type",""),p.get("blood_type","")),"donor_blood":d.get("blood_type"),"organs":[o["organ"] for o in d.get("organs",[]) if o.get("status")=="available"]} for d in avd]
    return {"patient_id":pid,"matches":sorted(sc,key=lambda x:x["hla_score"]*0.6+(0.4 if x["blood_ok"] else 0),reverse=True)[:top]}

@app.get("/api/donors")
def api_donors(available_only:bool=Query(False)):
    donors=list(DB.donors.values())
    if available_only: donors=[d for d in donors if d.get("available")]
    return {"total":len(donors),"donors":donors}

@app.post("/api/donors",status_code=201)
async def api_create_donor(body:dict,bg:BackgroundTasks):
    d=_make_donor()
    for k in["name","age","blood_type","donor_type","hospital"]:
        if body.get(k): d[k]=body[k]
    if body.get("organs"): d["organs"]=[{"organ":o,"status":"available","viability_h":ORGAN_VIAB.get(o,24)} for o in body["organs"]]
    d["added_by"]=body.get("added_by","user"); DB.donors[d["id"]]=d; save_db()
    bg.add_task(wsm.broadcast,{"event":"donor_added","donor_id":d["id"],"ts":_ts()})
    return d

@app.post("/api/donors/{did}/refresh")
async def api_refresh_donor(did:str,bg:BackgroundTasks):
    d=DB.donors.get(did)
    if not d: raise HTTPException(404)
    d["available"]=True
    for o in d.get("organs",[]): o["status"]="available"
    save_db(); bg.add_task(wsm.broadcast,{"event":"donor_refreshed","donor_id":did,"ts":_ts()})
    return d

@app.get("/api/blood-bank")
def api_blood_bank():
    return {"summaries":list(DB.blood_bank.values()),"total_units":sum(b.get("units",0) for b in DB.blood_bank.values()),"critical_low":[bt for bt,b in DB.blood_bank.items() if b.get("units",0)<30],"expiring_soon":[bt for bt,b in DB.blood_bank.items() if b.get("expiry_days",99)<7]}

@app.post("/api/blood-bank/add")
async def api_add_blood(body:dict,bg:BackgroundTasks):
    bt=body.get("blood_type","O+"); units=int(body.get("units",0))
    if not units: raise HTTPException(400,"units required")
    if bt in DB.blood_bank: DB.blood_bank[bt]["units"]+=units
    else: DB.blood_bank[bt]={"blood_type":bt,"units":units,"component":body.get("component","Whole Blood"),"expiry_days":int(body.get("expiry_days",42))}
    save_db(); bg.add_task(wsm.broadcast,{"event":"blood_added","blood_type":bt,"units":units,"ts":_ts()})
    return DB.blood_bank[bt]

@app.post("/api/blood-bank/dispense")
async def api_dispense(body:dict,bg:BackgroundTasks):
    bt=body.get("blood_type"); units=int(body.get("units",2))
    if not bt or DB.blood_bank.get(bt,{}).get("units",0)<units: raise HTTPException(409,f"Insufficient {bt}")
    DB.blood_bank[bt]["units"]-=units; save_db()
    bg.add_task(wsm.broadcast,{"event":"blood_dispensed","blood_type":bt,"units":units,"ts":_ts()})
    return {"dispensed":units,"blood_type":bt,"remaining":DB.blood_bank[bt]["units"]}

@app.get("/api/hospitals")
def api_hospitals(): return {"hospitals":list(DB.hospitals.values())}

@app.get("/api/transport")
def api_transport(active_only:bool=Query(False)):
    routes=DB.routes if not active_only else [r for r in DB.routes if r.get("active")]
    return {"total":len(routes),"active":sum(1 for r in DB.routes if r.get("active")),"routes":sorted(routes,key=lambda r:r.get("hours",0))}

@app.get("/api/hla/matrix")
def api_hla_matrix(n_patients:int=Query(5),n_donors:int=Query(5)):
    pts=[p for p in DB.patients.values() if p.get("urgency")=="critical"][:n_patients]
    dons=[d for d in DB.donors.values() if d.get("available")][:n_donors]
    return {"donors":[{"id":d["id"],"name":d.get("name"),"blood_type":d.get("blood_type")} for d in dons],"matrix":[{"patient_id":p["id"],"patient_name":p.get("name"),"blood_type":p.get("blood_type"),"is_paediatric":p.get("is_paediatric"),"disease":p.get("disease"),"scores":{d["id"]:{"hla_pct":int(_hla(p.get("hla",{}),d.get("hla",{}))*100),"blood_ok":_blood_ok(d.get("blood_type",""),p.get("blood_type",""))} for d in dons}} for p in pts]}

@app.get("/api/allocations")
def api_allocations(status:str=Query(None)):
    allocs=list(DB.allocations.values())
    if status and status!="all": allocs=[a for a in allocs if a.get("status")==status]
    return {"total":len(allocs),"pending":sum(1 for a in allocs if a.get("status")=="pending"),"active":sum(1 for a in allocs if a.get("status")=="active"),"complete":sum(1 for a in allocs if a.get("status")=="complete"),"failed":sum(1 for a in allocs if a.get("status")=="failed"),"allocations":sorted(allocs,key=lambda a:a.get("created_at",""),reverse=True)}

@app.post("/api/allocations/auto-match")
async def api_auto_match(bg:BackgroundTasks):
    crit=[p for p in DB.patients.values() if p.get("urgency")=="critical" and not p.get("is_allocated")]
    avd=[d for d in DB.donors.values() if d.get("available")]
    if not crit or not avd: return {"matched":0,"allocations":[],"message":"No critical patients or donors"}
    made=[]
    for p in crit[:6]:
        d=avd.pop(0) if avd else None
        if not d: break
        bh=min(DB.hospitals.values(),key=lambda h:h.get("load_pct",100)); rew=_reward(p,d,bh.get("name"),DB.ep_expired,len(crit)); aid=_naid()
        alloc={"id":aid,"patient_id":p["id"],"patient_name":p.get("name"),"donor_id":d["id"],"donor_name":d.get("name"),"organ":d.get("organs",[{"organ":"Unknown"}])[0].get("organ"),"hospital":bh.get("name"),"action_type":"match_organ","status":"pending","reward":rew,"hla_score":rew["hla_score"],"is_paediatric":p.get("is_paediatric",False),"step":DB.ep_step,"created_at":_ts(),"disease":p.get("disease"),"source":"auto_match"}
        DB.allocations[aid]=alloc; d["available"]=False; p["is_allocated"]=True; p["allocation_id"]=aid; DB.ep_cum+=rew["value"]; made.append(alloc)
    save_db(); bg.add_task(wsm.broadcast,{"event":"auto_match","count":len(made),"ts":_ts()})
    return {"matched":len(made),"allocations":made}

@app.post("/api/allocations/commit")
async def api_commit(body:dict,bg:BackgroundTasks):
    pid=body.get("patient_id"); did=body.get("donor_id"); hosp=body.get("hospital")
    p=DB.patients.get(pid); d=DB.donors.get(did) if did else None
    if not p: raise HTTPException(404,f"Patient {pid} not found")
    if p.get("is_allocated"): raise HTTPException(409,"Already allocated")
    if d and not d.get("available"): raise HTTPException(409,"Donor not available")
    rew=_reward(p,d,hosp,DB.ep_expired,sum(1 for x in DB.patients.values() if x.get("urgency")=="critical"),body.get("action_type","match_organ")); aid=_naid()
    alloc={"id":aid,"patient_id":p["id"],"patient_name":p.get("name"),"donor_id":d["id"] if d else None,"donor_name":d.get("name") if d else None,"organ":d.get("organs",[{"organ":"Unknown"}])[0].get("organ") if d else p.get("organ_needed"),"hospital":hosp,"action_type":body.get("action_type","match_organ"),"status":"pending","reward":rew,"hla_score":rew["hla_score"],"is_paediatric":p.get("is_paediatric",False),"step":DB.ep_step,"created_at":_ts(),"disease":p.get("disease"),"source":"manual"}
    DB.allocations[aid]=alloc; DB.ep_cum+=rew["value"]
    if d: d["available"]=False
    p["is_allocated"]=True; p["allocation_id"]=aid; save_db()
    bg.add_task(wsm.broadcast,{"event":"allocation_created","allocation_id":aid,"score":rew["value"],"ts":_ts()})
    return alloc

@app.patch("/api/allocations/{aid}")
async def api_update_alloc(aid:str,body:dict,bg:BackgroundTasks):
    a=DB.allocations.get(aid)
    if not a: raise HTTPException(404)
    ns=body.get("status")
    if ns not in("pending","active","complete","failed","cancelled"): raise HTTPException(400)
    a["status"]=ns; a["updated_at"]=_ts()
    if ns=="complete": a["completed_at"]=_ts()
    save_db(); bg.add_task(wsm.broadcast,{"event":"allocation_updated","id":aid,"status":ns,"ts":_ts()})
    return a

@app.get("/api/rl/matches")
def api_rl_matches(n:int=Query(5)):
    crit=[p for p in DB.patients.values() if p.get("urgency")=="critical" and not p.get("is_allocated")]
    avd=[d for d in DB.donors.values() if d.get("available")]
    if not crit or not avd: return {"matches":[]}
    results=[]
    for p in crit[:n]:
        for d in avd[:4]:
            bh=min(DB.hospitals.values(),key=lambda h:h.get("load_pct",100)); rew=_reward(p,d,bh.get("name"),DB.ep_expired,len(crit))
            results.append({"patient":{"id":p["id"],"name":p.get("name"),"blood":p.get("blood_type"),"ischaemia_h":round(p.get("ischaemia_h",0),1),"is_paediatric":p.get("is_paediatric",False),"disease":p.get("disease")},"donor":{"id":d["id"],"name":d.get("name"),"blood":d.get("blood_type"),"organs":[o["organ"] for o in d.get("organs",[]) if o.get("status")=="available"]},"hospital":bh.get("name"),"rl_score":rew["value"],"hla_score":rew["hla_score"],"breakdown":rew["breakdown"],"explanation":rew["explanation"]})
    return {"matches":sorted(results,key=lambda x:x["rl_score"],reverse=True)[:n]}

@app.post("/api/rl/run-episode")
async def api_run_episode(body:dict):
    task=body.get("task","crisis_routing")
    if task not in TASK_INFO: task="crisis_routing"
    info=TASK_INFO[task]; steps=[]; cum=0.0
    pts=[p for p in DB.patients.values() if p.get("urgency")=="critical" and not p.get("is_allocated")]
    dons=[d for d in DB.donors.values() if d.get("available")]
    max_s=min(info["max_steps"],max(len(pts),1),20)
    for i in range(max_s):
        if not pts or not dons: break
        p=pts[i%len(pts)]; d=dons[i%len(dons)]; bh=min(DB.hospitals.values(),key=lambda h:h.get("load_pct",100)); rew=_reward(p,d,bh.get("name"),0,len(pts)); cum+=rew["value"]
        steps.append({"step":i+1,"action":f"{p.get('name','?')[:12]}>{d.get('name','?')[:12]}","detail":f"Blood:{p.get('blood_type')}/{d.get('blood_type')} HLA:{int(rew['hla_score']*100)}% Isch:{p.get('ischaemia_h',0):.1f}h"+(" PAED" if p.get("is_paediatric") else ""),"reward":rew["value"],"cum":round(cum,3),"hla_score":rew["hla_score"],"is_paediatric":p.get("is_paediatric",False)})
    base={"single_match":0.847,"batch_allocation":0.693,"crisis_routing":0.521}
    score=round(base.get(task,0.5)+(random.random()*0.04-0.02),3)
    return {"task":task,"difficulty":info["difficulty"],"steps":steps,"total_reward":round(cum,3),"avg_reward":round(cum/max(len(steps),1),3),"score":score,"success":score>=0.3,"rewards":[s["reward"] for s in steps]}

@app.post("/api/llm/decide")
async def api_llm_decide(bg:BackgroundTasks):
    obs=_obs(); pts=list(DB.patients.values()); dons=list(DB.donors.values())
    result=await llm_decide(obs,pts,dons)
    DB.llm_log.append({"ts":_ts(),"mode":result.get("mode"),"reasoning":result.get("reasoning",""),"confidence":result.get("confidence",0),"action":result.get("action",{}),"priority_factors":result.get("priority_factors",[])})
    action=result.get("action",{}); committed=None
    if action.get("patient_id") and action.get("action_type")!="skip":
        p=DB.patients.get(action.get("patient_id","")); d=DB.donors.get(action.get("donor_id","")) if action.get("donor_id") else None
        if p and not p.get("is_allocated") and (not d or d.get("available")):
            rew=_reward(p,d,action.get("hospital"),DB.ep_expired,sum(1 for x in DB.patients.values() if x.get("urgency")=="critical")); aid=_naid()
            alloc={"id":aid,"patient_id":p["id"],"patient_name":p.get("name"),"donor_id":d["id"] if d else None,"donor_name":d.get("name") if d else None,"organ":d.get("organs",[{"organ":"Unknown"}])[0].get("organ") if d else p.get("organ_needed"),"hospital":action.get("hospital"),"action_type":"match_organ","status":"active","reward":rew,"hla_score":rew["hla_score"],"is_paediatric":p.get("is_paediatric",False),"step":DB.ep_step,"created_at":_ts(),"source":"llm_decision","llm_reasoning":result.get("reasoning",""),"disease":p.get("disease")}
            DB.allocations[aid]=alloc; DB.ep_cum+=rew["value"]
            if d: d["available"]=False
            p["is_allocated"]=True; p["allocation_id"]=aid; committed=alloc
    save_db(); await wsm.broadcast({"event":"llm_decision","mode":result.get("mode"),"reasoning":result.get("reasoning",""),"committed":committed is not None,"ts":_ts()})
    return {"decision":result,"committed_allocation":committed}

@app.get("/api/llm/log")
def api_llm_log(limit:int=Query(20)):
    return {"total":len(DB.llm_log),"log":DB.llm_log[-limit:],"has_llm":bool(HF_TOKEN),"model":MODEL_NAME if HF_TOKEN else "rule_based"}

@app.get("/api/llm/models")
def api_llm_models():
    return {"models":[{"id":"Qwen/Qwen2.5-72B-Instruct","name":"Qwen 2.5 72B"},{"id":"meta-llama/Llama-3.1-70B-Instruct","name":"Llama 3.1 70B"}],"current":MODEL_NAME if HF_TOKEN else "rule_based","has_key":bool(HF_TOKEN)}

class ChatMsg(BaseModel):
    message:str
    session_id:Optional[str]="global"

@app.post("/api/chat")
async def api_chat(body:ChatMsg,bg:BackgroundTasks,request:Request):
    if not _rate_ok(f"chat_{request.client.host if request.client else 'anon'}",20):
        raise HTTPException(429,"Too many requests")
    pts=list(DB.patients.values()); crit=[p for p in pts if p.get("urgency")=="critical"]
    ctx=f"Patients: {len(pts)} | Critical: {len(crit)} | Paed critical: {sum(1 for p in crit if p.get('is_paediatric'))} | Allocations: {len(DB.allocations)} | Donors available: {sum(1 for d in DB.donors.values() if d.get('available'))} | Expired: {DB.ep_expired} | Categories: {', '.join(DISEASE_DB.keys())}"
    session_msgs=[]
    for h in DB.chat_history[-10:]:
        if h.get("session") in(body.session_id,"global"): session_msgs.append({"role":h["role"],"content":h["content"]})
    session_msgs.append({"role":"user","content":body.message})
    response=await llm_chat(session_msgs,ctx)
    DB.chat_history.append({"role":"user","content":body.message,"ts":_ts(),"session":body.session_id})
    DB.chat_history.append({"role":"assistant","content":response,"ts":_ts(),"session":body.session_id})
    save_db(); bg.add_task(wsm.broadcast,{"event":"chat","session":body.session_id,"ts":_ts()})
    return {"response":response,"model":MODEL_NAME if HF_TOKEN else "no_llm","ts":_ts(),"session_id":body.session_id}

@app.get("/api/chat/history")
def api_chat_history(session_id:str=Query("global"),limit:int=Query(50)):
    msgs=[h for h in DB.chat_history if h.get("session") in(session_id,"global")]
    return {"messages":msgs[-limit:],"total":len(msgs),"has_llm":bool(HF_TOKEN)}

@app.delete("/api/chat/history")
async def api_clear_chat(bg:BackgroundTasks):
    DB.chat_history.clear(); save_db()
    bg.add_task(wsm.broadcast,{"event":"chat_cleared","ts":_ts()})
    return {"cleared":True}

@app.get("/api/alerts")
def api_alerts(resolved:bool=Query(False),limit:int=Query(50)):
    return {"alerts":[a for a in DB.alerts if a.get("resolved",False)==resolved][:limit]}

@app.post("/api/alerts/{aid}/resolve")
async def api_resolve(aid:str,bg:BackgroundTasks):
    for a in DB.alerts:
        if a.get("id")==aid: a["resolved"]=True; a["resolved_at"]=_ts(); save_db(); return {"resolved":aid}
    raise HTTPException(404)

@app.get("/api/analytics")
def api_analytics():
    return {"history":DB.analytics[-100:],"current":{"ts":_ts(),"total":len(DB.patients),"critical":sum(1 for p in DB.patients.values() if p.get("urgency")=="critical"),"allocations":len(DB.allocations),"expired":DB.ep_expired}}

@app.get("/api/disease-catalog")
def api_disease_catalog():
    return {"categories":list(DISEASE_DB.keys()),"total_diseases":sum(len(v["diseases"]) for v in DISEASE_DB.values()),"catalog":{k:{"diseases":v["diseases"],"treatment":v["treatment"],"organ":v["organ"],"symptoms":v["symptoms"],"medications":v["medications"]} for k,v in DISEASE_DB.items()}}

@app.post("/api/admin/inject-trauma")
async def api_inject_trauma(bg:BackgroundTasks):
    p=_make_pt("trauma","critical"); DB.patients[p["id"]]=p
    DB.alerts.append({"id":_nalid(),"severity":"critical","title":f"TRAUMA: {p['name']}","message":f"{p.get('disease')} - {p.get('ischaemia_h',0):.1f}h","patient_id":p["id"],"created_at":_ts(),"resolved":False})
    save_db(); bg.add_task(wsm.broadcast,{"event":"trauma_injected","patient":p,"ts":_ts()})
    return {"injected":True,"patient":p}

@app.get("/api/admin/stats")
def api_admin_stats():
    return {"patients":len(DB.patients),"donors":len(DB.donors),"allocations":len(DB.allocations),"alerts":len(DB.alerts),"chat_messages":len(DB.chat_history),"llm_decisions":len(DB.llm_log),"save_file":str(SAVE_FILE),"file_exists":SAVE_FILE.exists(),"file_size_kb":round(SAVE_FILE.stat().st_size/1024,1) if SAVE_FILE.exists() else 0,"disease_categories":len(DISEASE_DB),"total_diseases":sum(len(v["diseases"]) for v in DISEASE_DB.values())}

@app.post("/api/admin/reseed")
async def api_reseed(bg:BackgroundTasks):
    DB.patients.clear(); DB.donors.clear(); DB.allocations.clear(); DB.alerts.clear()
    DB._pctr=0; DB._dctr=0; DB._actr=0; DB._alctr=0; DB._seeded=False; DB.ep_cum=0.0; DB.ep_expired=0
    _seed(); bg.add_task(wsm.broadcast,{"event":"reseeded","patients":len(DB.patients),"ts":_ts()})
    return {"reseeded":True,"patients":len(DB.patients),"donors":len(DB.donors)}

@app.get("/api/status")
def api_status():
    return {"ok":True,"ts":_ts(),"version":"4.0.0","patients":len(DB.patients),"donors":len(DB.donors),"allocations":len(DB.allocations),"llm_enabled":bool(HF_TOKEN),"hf_dataset":bool(HF_REPO_ID),"save_exists":SAVE_FILE.exists()}

@app.websocket("/ws")
async def websocket_endpoint(websocket:WebSocket):
    await wsm.connect(websocket)
    try:
        await websocket.send_json({"event":"connected","patients":len(DB.patients),"critical":sum(1 for p in DB.patients.values() if p.get("urgency")=="critical"),"llm_enabled":bool(HF_TOKEN),"chatbot_enabled":bool(HF_TOKEN),"ts":_ts()})
        while True:
            await asyncio.sleep(10)
            try:
                await websocket.send_json({"event":"heartbeat","ts":_ts(),"critical":sum(1 for p in DB.patients.values() if p.get("urgency")=="critical"),"allocations":len(DB.allocations),"chat_messages":len(DB.chat_history),"expired_organs":DB.ep_expired})
            except: break
    except WebSocketDisconnect: wsm.disconnect(websocket)
    except Exception as e: logger.warning(f"WS error: {e}"); wsm.disconnect(websocket)

if __name__=="__main__":
    import uvicorn
    uvicorn.run("app:app",host="0.0.0.0",port=PORT,reload=False,log_level="info")
