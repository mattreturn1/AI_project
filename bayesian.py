import os
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

REGIONS_ORDER = [
    'Prefrontal Cortex', 'Insula', 'Cingulate Cortex',
    'Hippocampus', 'Amygdala', 'Temporal Region', 'Cerebellum'
]
AGE_GROUPS = ['11-', '12_17', '18_25', '25+']
METRIC_TYPES = ['Closeness', 'Clustering', 'Degree']

def process_data(root_dir='analysis/abide'):
    subjects_data = []
    normalization_params = {ag: {mt: {'min': np.inf, 'max': -np.inf} for mt in METRIC_TYPES} for ag in AGE_GROUPS}
    seen_subjects = set()

    for root, dirs, files in os.walk(root_dir):
        if 'abide_roi_metrics.csv' in files:
            path_parts = root.split(os.sep)
            age_group = next((ag for ag in AGE_GROUPS if ag in path_parts), None)
            label = 0 if 'control' in path_parts else 1

            if not age_group:
                continue

            df = pd.read_csv(os.path.join(root, 'abide_roi_metrics.csv'))

            for subject_id, group in df.groupby('SubjectID'):
                if subject_id in seen_subjects:
                    continue
                seen_subjects.add(subject_id)

                if len(group) != len(REGIONS_ORDER):
                    continue

                group['Region'] = pd.Categorical(group['Region'], categories=REGIONS_ORDER, ordered=True)
                sorted_group = group.sort_values('Region')

                metrics = []
                for _, row in sorted_group.iterrows():
                    for mt in METRIC_TYPES:
                        val = row[mt]
                        metrics.append(val)
                        if val < normalization_params[age_group][mt]['min']:
                            normalization_params[age_group][mt]['min'] = val
                        if val > normalization_params[age_group][mt]['max']:
                            normalization_params[age_group][mt]['max'] = val

                subjects_data.append({
                    'age_group': age_group,
                    'sex': 0 if sorted_group['Sex'].iloc[0] == 'M' else 1,
                    'age': sorted_group['Age'].iloc[0],
                    'metrics': metrics,
                    'label': label
                })

    data_vectors = []
    for subj in subjects_data:
        ag = subj['age_group']
        normalized = []
        for i, mt in enumerate(METRIC_TYPES * len(REGIONS_ORDER)):
            mt_type = METRIC_TYPES[i % 3]
            val = subj['metrics'][i]
            min_val = normalization_params[ag][mt_type]['min']
            max_val = normalization_params[ag][mt_type]['max']
            norm_val = 0.5 if max_val == min_val else (val - min_val) / (max_val - min_val)
            normalized.append(norm_val)
        vector = {
            'Sex': subj['sex'],
            'AgeGroup': ag,
            'Age': subj['age'],
            'Metrics': normalized,
            'Diagnosis': subj['label']
        }
        data_vectors.append(vector)

    return data_vectors

def create_dataframe(processed_data):
    rows = []
    for entry in processed_data:
        row = {
            'Sex': entry['Sex'],
            'AgeGroup': entry['AgeGroup'],
        }

        metrics = entry['Metrics']
        cnt = 0
        for mt in METRIC_TYPES:
            for region in REGIONS_ORDER:
                col_name = f"{mt}_{region}"
                row[col_name] = 1 if metrics[cnt] > 0.5 else 0
                cnt += 1

        row['Diagnosis'] = entry['Diagnosis']
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def build_and_train_bn(df):
    edges = []
    metric_columns = []

    for mt in METRIC_TYPES:
        for region in REGIONS_ORDER:
            col_name = f"{mt}_{region}"
            metric_columns.append(col_name)
            edges.append(('Sex', col_name))
            edges.append(('AgeGroup', col_name))
            edges.append((col_name, 'Diagnosis'))

    model = DiscreteBayesianNetwork(edges)

    # Convert AgeGroup to category for pgmpy compatibility
    df['AgeGroup'] = df['AgeGroup'].astype('category')
    df['Sex'] = df['Sex'].astype(int)
    df['Diagnosis'] = df['Diagnosis'].astype(int)

    model.fit(df, estimator=MaximumLikelihoodEstimator)
    return model

# Esecuzione
processed_data = process_data()
if not processed_data:
    print("Nessun dato processato. Verifica la struttura delle directory e dei file CSV.")
else:
    print(f"Numero di soggetti elaborati: {len(processed_data)}")

    df = create_dataframe(processed_data)
    print("Esempio di dati processati (prime righe):")
    print(df.head())

    model = build_and_train_bn(df)

    print("\nParametri appresi (CPD):")
    for cpd in model.get_cpds():
        print(cpd)