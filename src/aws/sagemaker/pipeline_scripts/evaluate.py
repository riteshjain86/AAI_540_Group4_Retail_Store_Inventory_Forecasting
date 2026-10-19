"""
Pipeline step 3 (Evaluate): extract evaluation.json from the trained
model.tar.gz and publish it as the step's property file.

Only the JSON member is read; the model itself is never deserialized, so
the gate cannot be affected by (or execute) anything inside the pickle.
"""

import json
import os
import tarfile

MODEL_DIR = '/opt/ml/processing/model'
OUTPUT_DIR = '/opt/ml/processing/evaluation'
REQUIRED_KEYS = ('test', 'baseline_seasonal_naive', 'wape_improvement_vs_baseline')


def read_evaluation(model_tar: str) -> dict:
    with tarfile.open(model_tar) as tar:
        member = next((m for m in tar.getmembers()
                       if m.isfile() and os.path.basename(m.name) == 'evaluation.json'), None)
        if member is None:
            raise FileNotFoundError('evaluation.json not found in model artifact; '
                                    'was the training job given a test channel?')
        report = json.load(tar.extractfile(member))
    missing = [k for k in REQUIRED_KEYS if k not in report]
    if missing:
        raise ValueError(f'evaluation.json missing keys: {missing}')
    return report


def main():
    report = read_evaluation(os.path.join(MODEL_DIR, 'model.tar.gz'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'evaluation.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'test': report['test'], 'baseline_seasonal_naive': report['baseline_seasonal_naive'],
                      'wape_improvement_vs_baseline': report['wape_improvement_vs_baseline']}, indent=2))


if __name__ == '__main__':
    main()
