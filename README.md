# readmission-predictor

Predicting which diabetic patients are readmitted to hospital within 30 days, using pandas and scikit-learn on the UCI Diabetes 130-US Hospitals dataset (101,766 encounters, 1999-2008).

Results, findings report and an in-browser risk calculator are added after the first full run.

## Run it
```bash
pip install -r requirements.txt
scripts/get_data.sh data                     # downloads from UCI and checks a pinned SHA-256
pytest -q
python scripts/run_pipeline.py --data data/diabetic_data.csv --out results --site site
```

Data: Strack, B. et al. (2014), *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*, BioMed Research International. UCI Machine Learning Repository, CC BY 4.0.
