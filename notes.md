\# IoT Anomaly Detection — My Project Notes



\## Setup Log



I installed Python, pandas, scikit-learn, and JupyterLab — all globally 

(no virtual environment), after I first tried to isolate scikit-learn 

in a venv (sklearn-env) but that turned out to be unnecessary 

complexity for this project.



Commands I used:

```

pip install pandas

pip install scikit-learn

pip install jupyterlab

```



Why I installed each one:

\- Python: the language the whole project runs in.

\- pandas: to load, clean, and analyze the CICIoT2023 CSV dataset.

\- scikit-learn: to preprocess data, train the Random Forest model, 

&#x20; and evaluate its performance.

\- JupyterLab: an interactive environment so I can write and run code 

&#x20; step-by-step and check results as I go.



I verified this by opening a notebook in JupyterLab and running 

import pandas and import sklearn successfully (pandas 3.0.5, 

scikit-learn 1.9.0).



\---



\## Dataset



I got my data from CICIoT2023 (Canadian Institute for Cybersecurity, UNB).



I downloaded these 4 files:

\- BenignTraffic.pcap.csv

\- DDoS-SYN\_Flood.pcap.csv

\- Mirai-udpplain.pcap.csv

\- Recon-PortScan.pcap.csv



Why I picked these four:

\- BenignTraffic — this is my "normal" class. Without it the model has 

&#x20; nothing to compare attacks against.

\- DDoS-SYN\_Flood — a classic, well-documented, structurally distinct 

&#x20; attack (overwhelms a server with connection requests it never 

&#x20; completes). Good easy pattern for the model to learn first.

\- Mirai-udpplain — Mirai is the most well-known IoT-specific botnet 

&#x20; malware, famous for taking down major internet infrastructure in 

&#x20; 2016. This keeps my project tied to IoT security specifically, not 

&#x20; just generic network intrusion detection.

\- Recon-PortScan — this one is different in kind, not just another 

&#x20; flood. SYN Flood and Mirai-udpplain are both volumetric attacks; 

&#x20; port scanning is quiet reconnaissance instead. This makes my model 

&#x20; actually have to learn more than "spot a traffic spike."



\---



\## Day 2 — Explore and Clean the Data



\*\*1. Loaded the four CSV files\*\*

```python

import pandas as pd



benign = pd.read\_csv("data/BenignTraffic.pcap.csv")

syn\_flood = pd.read\_csv("data/DDoS-SYN\_Flood.pcap.csv")

mirai = pd.read\_csv("data/Mirai-udpplain.pcap.csv")

portscan = pd.read\_csv("data/Recon-PortScan.pcap.csv")



print("Files loaded successfully!")

```

pandas loads CSV data into a DataFrame — basically a programmable 

table, like an Excel spreadsheet Python can work with. I loaded the 

four files separately because they represent different traffic types.



\*\*2. Added a label to each file before combining them\*\*



The raw CSVs don't come with a label column — the only way to know 

which traffic type a row belongs to is by which file it came from, so 

I added the labels myself.

```python

benign\["label"] = "Benign"

syn\_flood\["label"] = "DDoS-SYN\_Flood"

mirai\["label"] = "Mirai-udpplain"

portscan\["label"] = "Recon-PortScan"

```

This matters because I'm doing supervised learning — the model learns 

from examples where the correct answer (the label) is already known. 

Features = the input info used to predict; label = what I want it to 

predict.



\*\*3. Combined all four into one dataset\*\*

```python

df = pd.concat(\[benign, syn\_flood, mirai, portscan], ignore\_index=True)

print("Dataset shape:", df.shape)

```

Result: 747,150 rows, 40 columns.



\*\*4. Checked the class balance\*\*

```python

print(df\["label"].value\_counts())

```

Result:

\- Benign: 362,361

\- DDoS-SYN\_Flood: 266,024

\- Recon-PortScan: 82,284

\- Mirai-udpplain: 36,481



This showed me class imbalance — the classes don't have equal numbers 

of examples. This matters because a model could score high overall 

accuracy while doing badly on a small class like Mirai. That's why 

accuracy alone isn't enough — I need precision, recall, F1-score, and 

the confusion matrix too, especially recall, since I care about how 

many actual attacks got detected.



\*\*5. Checked data quality\*\*

```python

df.isnull().sum()

print(df.duplicated().sum())

df.info()

df.describe()

```

\- isnull().sum() — shows missing values per column

\- duplicated().sum() — shows fully duplicated rows

\- info() — shows row/column counts, data types, non-null counts

\- describe() — shows mean, min, max, std, quartiles



\*\*6. Found specific issues\*\*

```python

print(df.isnull().sum()\[df.isnull().sum() > 0])



import numpy as np

print((df\["Rate"] == np.inf).sum())

```

I found 24 missing values each in Std and Variance (48 total), 27 

infinite values in Rate, and 93,583 fully duplicated rows.



\*\*7. Cleaned the data\*\*

```python

df = df.drop\_duplicates()

df = df.dropna(subset=\["Std", "Variance"])

df = df\[df\["Rate"] != np.inf]



print(df.shape)

print(df.isnull().sum().sum())

print((df\["Rate"] == np.inf).sum())

print(df.duplicated().sum())

```

This brought my dataset down from 747,150 to 653,540 clean rows, with 

0 remaining nulls, infinities, and duplicates.



\---



\## Day 3 — Preprocess and Split



I didn't scale the features, since Random Forest splits data using 

per-feature thresholds, not distance/magnitude comparisons across 

features — so different numeric ranges don't affect it.



```python

from sklearn.model\_selection import train\_test\_split



X = df.drop(columns=\["label"])   # features

y = df\["label"]                   # label/target



X\_train, X\_test, y\_train, y\_test = train\_test\_split(

&#x20;   X, y,

&#x20;   test\_size=0.2,

&#x20;   stratify=y,

&#x20;   random\_state=42

)



print("Train shape:", X\_train.shape)

print("Test shape:", X\_test.shape)

print("\\nTrain class distribution:")

print(y\_train.value\_counts(normalize=True))

print("\\nTest class distribution:")

print(y\_test.value\_counts(normalize=True))

```

\- test\_size=0.2 → 80% train, 20% test

\- stratify=y → keeps class proportions the same in both sets, which 

&#x20; matters because of the imbalance I found earlier

\- random\_state=42 → makes the split reproducible every time I run it



Result: Train (522,832 rows), Test (130,708 rows). The class 

proportions matched to the third decimal place between train and 

test, which confirmed the stratification worked.



\---



\## Day 4-6 — Train the Random Forest Model



```python

from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(

&#x20;   n\_estimators=100,

&#x20;   random\_state=42,

&#x20;   n\_jobs=-1

)



model.fit(X\_train, y\_train)

y\_pred = model.predict(X\_test)



from sklearn.metrics import accuracy\_score

print("Accuracy:", accuracy\_score(y\_test, y\_pred))

```

\- n\_estimators=100 → builds 100 decision trees

\- n\_jobs=-1 → uses all my CPU cores to train faster



How Random Forest actually works: it builds many decision trees, each 

trained on a random slice of the data and a random subset of 

features. For a new row, every tree votes on a label on its own, and 

the majority vote becomes the final prediction. Averaging many trees 

this way cancels out individual trees' quirks, which suits noisy, 

high-dimensional data like network traffic (39 features here).



Result: 93.2% accuracy.



\---



\## Day 7-8 — Evaluate Properly



```python

from sklearn.metrics import classification\_report, confusion\_matrix



print(classification\_report(y\_test, y\_pred))



cm = confusion\_matrix(y\_test, y\_pred, labels=model.classes\_)

cm\_df = pd.DataFrame(cm, index=model.classes\_, columns=model.classes\_)

print(cm\_df)

```



My classification report results:

\- Benign — precision 0.90, recall 0.99, F1 0.94

\- DDoS-SYN\_Flood — precision 1.00, recall 1.00, F1 1.00

\- Mirai-udpplain — precision 1.00, recall 1.00, F1 1.00

\- Recon-PortScan — precision 0.89, recall 0.51, F1 0.65



The important finding: Recon-PortScan recall = 0.51. My model missed 

almost half of the actual port scans. Looking at the confusion 

matrix, 7,826 out of 16,049 actual Recon-PortScan rows were 

misclassified as Benign — almost none were confused with the other 

attack types.



Why I think this happened: a port scan is quiet reconnaissance, one 

connection at a time, unlike a flood attack which is loud and 

obviously abnormal. This makes scans genuinely harder to tell apart 

from low-volume normal traffic — this is actually a known, 

well-documented challenge in network intrusion detection, not just a 

mistake in my setup.



Why accuracy alone would have been misleading: my dataset is 

imbalanced. The overall 93.2% accuracy looked high, but it was mostly 

propped up by near-perfect performance on Benign (55% of my test set) 

and DDoS-SYN\_Flood, while hiding that my model was missing about half 

of all reconnaissance attempts — a real blind spot that a single 

accuracy number completely covers up.



\---



\## Limitations



\- I only used 4 of the dataset's 33 available attack types, because 

&#x20; of my compressed timeline.

\- I used a single train/test split — no cross-validation.

\- I didn't do any hyperparameter tuning — used default Random Forest 

&#x20; settings.



