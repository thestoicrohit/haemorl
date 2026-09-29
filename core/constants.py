"""
HaemoRL — Core Constants
All static data: blood types, HLA pool, organs, hospitals, diseases, names.
"""
from __future__ import annotations

BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# donor_blood_type -> set of patient_blood_types it can donate to
BLOOD_COMPAT: dict[str, set[str]] = {
    "O-":  {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"},
    "O+":  {"A+", "B+", "AB+", "O+"},
    "A-":  {"A+", "A-", "AB+", "AB-"},
    "A+":  {"A+", "AB+"},
    "B-":  {"B+", "B-", "AB+", "AB-"},
    "B+":  {"B+", "AB+"},
    "AB-": {"AB+", "AB-"},
    "AB+": {"AB+"},
}

HLA_POOL: dict[str, list[str]] = {
    "A":  ["A1","A2","A3","A11","A23","A24","A25","A26","A29","A30","A31","A32","A33"],
    "B":  ["B7","B8","B13","B14","B15","B18","B27","B35","B38","B39","B44","B51","B52","B57"],
    "DR": ["DR1","DR2","DR3","DR4","DR5","DR6","DR7","DR8","DR9","DR10","DR11","DR12","DR13"],
}

# Max cold ischaemia time in hours per organ
ORGAN_VIABILITY: dict[str, int] = {
    "Heart":      4,
    "Lung":       6,
    "Liver":      12,
    "Kidney":     36,
    "Pancreas":   12,
    "Bone Marrow": 72,
    "Multi-Organ": 6,
}

ORGAN_LIST = ["Heart", "Liver", "Kidney", "Lung", "Pancreas", "Bone Marrow"]

HOSPITALS = [
    "AIIMS New Delhi",       "PGI Chandigarh",       "SGPGI Lucknow",
    "Medanta Gurugram",      "Fortis Delhi",          "AIIMS Mumbai",
    "KEM Hospital Mumbai",   "Tata Memorial Mumbai",  "AIIMS Bhubaneswar",
    "SCB Cuttack",           "AIIMS Bengaluru",       "Manipal Hospital Bengaluru",
    "NIMHANS Bengaluru",     "JIPMER Puducherry",     "Fortis Chennai",
    "KGMU Lucknow",          "Narayana Health Kolkata","SSKM Kolkata",
    "GMCH Guwahati",         "AIIMS Jodhpur",
]

# Map coordinates (SVG space) for each hospital
HOSPITAL_COORDS: list[tuple[int, int]] = [
    (250,120),(230,95),(260,140),(248,125),(252,118),
    (205,190),(200,192),(208,188),(310,185),(315,188),
    (215,250),(218,252),(216,249),(260,265),(262,260),
    (258,138),(330,160),(328,162),(345,135),(195,170),
]

STATES = [
    "Delhi","Punjab","UP","Haryana","Rajasthan",
    "Maharashtra","Odisha","Karnataka","Tamil Nadu",
    "West Bengal","Assam","Gujarat",
]

CITIES = [
    "Mumbai","Delhi","Bengaluru","Hyderabad","Ahmedabad","Chennai","Kolkata",
    "Surat","Pune","Jaipur","Lucknow","Kanpur","Nagpur","Indore","Bhopal",
    "Visakhapatnam","Pimpri","Patna","Vadodara","Ludhiana",
]

# The state each city belongs to (so a patient in Bhopal is in Madhya Pradesh)
CITY_STATE: dict[str, str] = {
    "Mumbai": "Maharashtra", "Delhi": "Delhi", "Bengaluru": "Karnataka", "Hyderabad": "Telangana",
    "Ahmedabad": "Gujarat", "Chennai": "Tamil Nadu", "Kolkata": "West Bengal", "Surat": "Gujarat",
    "Pune": "Maharashtra", "Jaipur": "Rajasthan", "Lucknow": "Uttar Pradesh", "Kanpur": "Uttar Pradesh",
    "Nagpur": "Maharashtra", "Indore": "Madhya Pradesh", "Bhopal": "Madhya Pradesh",
    "Visakhapatnam": "Andhra Pradesh", "Pimpri": "Maharashtra", "Patna": "Bihar",
    "Vadodara": "Gujarat", "Ludhiana": "Punjab",
}

# State for each entry in HOSPITALS (same order)
HOSPITAL_STATES: list[str] = [
    "Delhi", "Chandigarh", "Uttar Pradesh", "Haryana", "Delhi",
    "Maharashtra", "Maharashtra", "Maharashtra", "Odisha", "Odisha",
    "Karnataka", "Karnataka", "Karnataka", "Puducherry", "Tamil Nadu",
    "Uttar Pradesh", "West Bengal", "West Bengal", "Assam", "Rajasthan",
]

FIRST_NAMES = [
    "Arjun","Priya","Rahul","Sunita","Vikram","Meera","Anil","Kavita","Suresh","Neha",
    "Ramesh","Pooja","Deepak","Anjali","Sanjay","Ritu","Amit","Geeta","Vivek","Usha",
    "Rohit","Leela","Kiran","Poonam","Arun","Lata","Prakash","Smita","Rajesh","Divya",
    "Mohan","Shanti","Nitin","Rekha","Vinod","Anita","Manish","Sarla","Ajay","Preeti",
    "Santosh","Radha","Sunil","Madhuri","Ashok","Nirmala","Pankaj","Sudha","Hemant","Savita",
]

LAST_NAMES = [
    "Sharma","Patel","Singh","Kumar","Gupta","Joshi","Mehta","Verma","Reddy","Nair",
    "Iyer","Pillai","Shah","Rao","Mishra","Tiwari","Pandey","Banerjee","Das","Chatterjee",
    "Bose","Mukherjee","Kapoor","Malhotra","Khanna","Bhatia","Saxena","Agarwal","Srivastava","Dubey",
]

OCCUPATIONS = [
    "Farmer","Teacher","Shopkeeper","Driver","Software Engineer","Housewife",
    "Student","Doctor","Labourer","Engineer","Retired","Nurse","Businessman",
]

DISEASE_DB: dict[str, dict] = {
    "oncology": {
        "diseases": [
            "Acute Lymphoblastic Leukemia","Chronic Myeloid Leukemia","Multiple Myeloma",
            "Hodgkin Lymphoma Stage III","Non-Hodgkin Lymphoma DLBCL","Lung Adenocarcinoma Stage IV",
            "Breast Carcinoma Triple Negative","Hepatocellular Carcinoma BCLC-C",
            "Pancreatic Ductal Adenocarcinoma","Glioblastoma Multiforme IDH-wildtype",
            "Colorectal Adenocarcinoma KRAS+","Ovarian Cancer High-Grade Stage IV",
            "Prostate Cancer Metastatic CRPC","Acute Myeloid Leukemia FLT3+",
            "CLL Richter Transformation","Myelofibrosis JAK2+","Mantle Cell Lymphoma",
            "T-Cell Lymphoma Angioimmunoblastic","Primary CNS Lymphoma","Waldenstrom Macroglobulinaemia",
        ],
        "treatment": "Bone Marrow Transplant", "organ": "Bone Marrow", "need_type": "marrow",
        "symptoms": ["Bone pain","Night sweats","Weight loss","Fatigue","Bruising easily","Recurrent infections"],
        "medications": ["Imatinib","Venetoclax","Rituximab","Bortezomib","Lenalidomide","Cytarabine"],
        "lab_markers": {"WBC": "elevated", "Hb": "low", "Platelets": "low", "LDH": "elevated"},
    },
    "haematology": {
        "diseases": [
            "Aplastic Anaemia Severe","Thalassaemia Major Beta","Sickle Cell Disease HbSS",
            "Haemophilia A Severe Factor VIII <1%","Haemophilia B Factor IX Deficiency",
            "Myelodysplastic Syndrome High-Risk","Paroxysmal Nocturnal Haemoglobinuria",
            "ITP Refractory","Fanconi Anaemia","TTP","Diamond-Blackfan Anaemia",
            "von Willebrand Disease Type 3","Autoimmune Haemolytic Anaemia",
            "Hereditary Spherocytosis Severe","Gaucher Disease Type 1","Pure Red Cell Aplasia",
        ],
        "treatment": "Bone Marrow Transplant", "organ": "Bone Marrow", "need_type": "marrow",
        "symptoms": ["Anaemia","Bleeding episodes","Transfusion dependence","Splenomegaly","Fatigue","Jaundice"],
        "medications": ["Deferoxamine","Hydroxyurea","Factor VIII Concentrate","Eltrombopag","Cyclosporine","Eculizumab"],
        "lab_markers": {"Hb": "critically low", "Reticulocytes": "low", "Ferritin": "very high"},
    },
    "cardiac": {
        "diseases": [
            "End-Stage Heart Failure NYHA IV EF <15%","Dilated Cardiomyopathy Idiopathic",
            "Ischaemic Cardiomyopathy Post-MI","Hypertrophic Obstructive Cardiomyopathy",
            "Congenital Heart Disease Tetralogy of Fallot","Restrictive Cardiomyopathy Amyloid",
            "Cardiac Sarcoidosis End-Stage","ARVC Right Ventricular Failure",
            "Valvular Heart Disease Severe AS","PAH WHO IV","Peripartum Cardiomyopathy",
            "Chagas Cardiomyopathy","Giant Cell Myocarditis",
        ],
        "treatment": "Heart Transplant", "organ": "Heart", "need_type": "organ",
        "symptoms": ["Severe dyspnoea at rest","Orthopnoea","Bilateral oedema","Syncope","Chest pain","Palpitations"],
        "medications": ["Carvedilol","Sacubitril/Valsartan","Furosemide","Spironolactone","Digoxin","Dobutamine"],
        "lab_markers": {"BNP": ">5000", "Troponin": "elevated", "Creatinine": "rising", "Na": "low"},
    },
    "renal": {
        "diseases": [
            "CKD Stage 5 GFR <10","Polycystic Kidney Disease End-Stage",
            "Diabetic Nephropathy ESRD","IgA Nephropathy Oxford M1E1S1T2",
            "FSGS Steroid-Resistant","Alport Syndrome X-Linked",
            "Lupus Nephritis Class IV","Rapidly Progressive GN ANCA+",
            "Primary Hyperoxaluria Type 1","Cystinosis Nephropathic",
            "Fabry Nephropathy","Analgesic Nephropathy",
            "Hypertensive Nephrosclerosis ESRD","Renal Amyloidosis AL",
        ],
        "treatment": "Kidney Transplant", "organ": "Kidney", "need_type": "organ",
        "symptoms": ["Oliguria","Peripheral oedema","Nausea/vomiting","Uraemic encephalopathy","Hypertension","Pruritus"],
        "medications": ["Haemodialysis","Erythropoietin","Phosphate binders","ACE inhibitors","Furosemide","Calcium"],
        "lab_markers": {"Creatinine": ">800", "eGFR": "<10", "Potassium": "high", "Phosphate": "high"},
    },
    "hepatic": {
        "diseases": [
            "Hepatic Cirrhosis Child-Pugh C MELD >30","Acute Liver Failure Paracetamol",
            "Acute Liver Failure Viral Hepatitis","Primary Biliary Cholangitis Stage IV",
            "Primary Sclerosing Cholangitis","Wilson Disease Hepatic Crisis",
            "Alpha-1 AT Deficiency ZZ","Autoimmune Hepatitis Stage IV",
            "NASH Cirrhosis Decompensated","Budd-Chiari Syndrome",
            "Haemochromatosis Cirrhosis","Hepatitis B Cirrhosis","Hepatitis C Cirrhosis",
        ],
        "treatment": "Liver Transplant", "organ": "Liver", "need_type": "organ",
        "symptoms": ["Ascites","Hepatic encephalopathy","Jaundice","Variceal bleeding","Coagulopathy","Hepatorenal syndrome"],
        "medications": ["Rifaximin","Lactulose","Spironolactone","Propranolol","Terlipressin","Albumin infusion"],
        "lab_markers": {"Bilirubin": "very high", "INR": ">2.5", "Albumin": "low", "MELD": "30-40"},
    },
    "pulmonary": {
        "diseases": [
            "IPF UIP Pattern FVC <50%","Cystic Fibrosis FEV1 <25%","COPD GOLD Stage IV FEV1 <20%",
            "PAH Group 1 WHO IV","Lymphangioleiomyomatosis Advanced","Alpha-1 AT Emphysema",
            "Bronchiectasis Severe Non-CF","Eisenmenger Syndrome",
            "Sarcoidosis Stage IV","Hypersensitivity Pneumonitis Fibrotic","CTD-ILD End-Stage",
        ],
        "treatment": "Lung Transplant", "organ": "Lung", "need_type": "organ",
        "symptoms": ["Severe dyspnoea at rest","Cyanosis","Cor pulmonale","6MWT <150m","Home O2 24h","Recurrent exacerbations"],
        "medications": ["Nintedanib","Pirfenidone","Sildenafil","Bosentan","Inhaled bronchodilators","Long-term O2"],
        "lab_markers": {"FEV1": "<25%", "FVC": "<50%", "PaO2": "low", "LAS score": "40-65"},
    },
    "diabetes": {
        "diseases": [
            "T1DM Severe Hypoglycaemia Unawareness","T1DM End-Organ Damage",
            "T1DM Brittle Recurrent DKA","Post-Total Pancreatectomy Diabetes",
            "MODY Type 3 HNF1A","Wolfram Syndrome",
            "T1DM Gastroparesis Severe","Neonatal Diabetes Mellitus","LADA End-Stage",
        ],
        "treatment": "Pancreas Transplant", "organ": "Pancreas", "need_type": "organ",
        "symptoms": ["Severe hypoglycaemia","Recurrent DKA","Weight loss","Neuropathy","Retinopathy","Gastroparesis"],
        "medications": ["Insulin pump","Glucagon kit","Metoclopramide","Gabapentin","ACE inhibitors","Dialysis"],
        "lab_markers": {"HbA1c": "very high or variable", "C-peptide": "<0.1", "GAD antibodies": "positive"},
    },
    "trauma": {
        "diseases": [
            "Polytrauma Multi-Organ Failure ISS >50","Severe Burns >45% TBSA Full Thickness",
            "Crush Injury Rhabdomyolysis AKI","Blast Injury Haemorrhagic Shock",
            "High Voltage Electrical Injury","TBI Diffuse Axonal",
            "Penetrating Abdominal Trauma","Spinal Cord Injury Complete C4",
            "Major Hepatic Laceration Grade V","Traumatic Aortic Transection",
        ],
        "treatment": "Emergency Blood Transfusion + Surgery", "organ": "Multi-Organ", "need_type": "blood",
        "symptoms": ["Haemodynamic instability","Active bleeding","Coagulopathy","Respiratory failure","Shock","MOF"],
        "medications": ["Tranexamic acid","FFP","Packed RBC","Platelets","Noradrenaline","Vasopressin"],
        "lab_markers": {"Hb": "critically low", "INR": "high", "Lactate": "very high", "pH": "acidotic"},
    },
    "infectious": {
        "diseases": [
            "Sepsis Multi-Organ Dysfunction SOFA >10","Infective Endocarditis Surgical Emergency",
            "Meningococcal Septicaemia with DIC","Cerebral Malaria Severe with Coma",
            "Necrotising Fasciitis Type 1","Invasive Aspergillosis Disseminated",
            "Mucormycosis Rhino-Orbital-Cerebral","Gram-Negative Bacteraemia ESBL",
            "C.Difficile Fulminant Colitis","Legionella Pneumonia Severe","Staphylococcal Toxic Shock",
        ],
        "treatment": "Emergency Blood Transfusion + Antibiotics", "organ": "Multi-Organ", "need_type": "blood",
        "symptoms": ["High fever >40°C","Hypotension","Tachycardia","Altered consciousness","Petechiae","MOF"],
        "medications": ["Meropenem","Vancomycin","Antifungals","Vasopressors","IVIG","Corticosteroids"],
        "lab_markers": {"WBC": "very high or low", "CRP": ">300", "Procalcitonin": ">10", "Lactate": "high"},
    },
    "genetic": {
        "diseases": [
            "Phenylketonuria Classic PKU","Gaucher Disease Type 3 Neuronopathic",
            "Fabry Disease End-Stage","MPS Hurler Syndrome","Cystic Fibrosis FEV1 <20%",
            "Alpha-1 AT Deficiency PiZZ Cirrhosis","Wilson Disease Neurological Crisis",
            "Niemann-Pick Disease Type C","Pompe Disease Late-Onset Severe",
            "Tyrosinaemia Type 1","Maple Syrup Urine Disease","Homocystinuria Classic",
            "OTC Deficiency Urea Cycle","Methylmalonic Acidaemia Severe",
        ],
        "treatment": "Bone Marrow + Enzyme Replacement Therapy", "organ": "Bone Marrow", "need_type": "marrow",
        "symptoms": ["Developmental delay","Hepatosplenomegaly","Neurological regression","Growth failure","Seizures","Coarse facies"],
        "medications": ["Enzyme replacement therapy","Substrate reduction","Dietary restriction","Cofactor therapy"],
        "lab_markers": {"Enzyme activity": "deficient", "Genetic testing": "confirmed mutation", "Substrate": "accumulated"},
    },
}

TASK_INFO: dict[str, dict] = {
    "single_match": {
        "difficulty": "easy",
        "max_steps": 10,
        "desc": "Match 1 critical patient to best compatible donor using HLA + blood type scoring.",
        "baseline": 0.847,
    },
    "batch_allocation": {
        "difficulty": "medium",
        "max_steps": 30,
        "desc": "Optimally match 5 critical patients to 5 donors under full medical constraints.",
        "baseline": 0.693,
    },
    "crisis_routing": {
        "difficulty": "hard",
        "max_steps": 60,
        "desc": "Live ischaemia clocks, trauma injections, hospital overload, geographic pressure.",
        "baseline": 0.521,
    },
}
