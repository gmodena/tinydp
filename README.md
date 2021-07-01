[![Project Status: Concept – Minimal or no implementation has been done yet, or the repository is only intended to be a limited example, demo, or proof-of-concept.](https://www.repostatus.org/badges/latest/concept.svg)](https://www.repostatus.org/#concept)
![Build status](https://github.com/gmodena/tinydp/workflows/build/badge.svg)

# tinydp
Sprinkle some differential privacy on sklearn pipelines.

# Getting started

This code is a proof of concept. It's expected to break when data presents degenerate cases. YMMV.

The package and its development deps can be installed with:
```bash
pythom -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py install
```

# Private Aggregation of Teacher Ensembles 

`PrivateClassifier` implements the Private Aggregation of Teacher Ensembles (PATE) framework to learn from private data.
PATE assumes two main components: private labelled datasets ("teachers") and a public unlabelled dataset ("student").

Private data is partitioned in non-overlapping training sets. 
An ensamble of "teachers" are trained independently (with no privacy guarantee). 
The "teacher" models are scored on the unlabelled, public, "student" datasets. Their predictions are aggregated and perturbed with random noise. A "student" model can then trained on public data labelled by the ensamble, instead of on the original, private, dataset.

For more details on how (and why) this work see the reference presented [here](https://nowave.it/course-notes-on-differential-privacy.html).

# Example

Currently only classification tasks are supported.
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from dp.ensemble import PrivateClassifier
from sklearn.metrics import classification_report

X, y = make_classification(n_features=5, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, n_samples=10000)

# We train teachers on private data, and use them to label public data.
# The public, labelled, dataset can be shared and used to train student models.
X_private, X_public, y_private, y_public = train_test_split(X, y, test_size=0.33, random_state=42)

# PrivateClassifier implements a PATE ensemble of teachers.
# It behaves like a regular sklearn Classifier. The epsilon
# parameter governs the amount of noise added to pred
clf = PrivateClassifier(n_estimators=10, epsilon=0.1, random_state=1)
clf.fit(X_private, y_private)
y_pred = clf.predict(X_public)

classification_report(y_public, y_pred)
```

# Evaluation

A-posteriori analysis can be performed on the teachers aggregate to determine whether the model satisfies 
the desired epsilon budget. OpenMined `pysyft` package provides some utilities for this.
```python
from syft.frameworks.torch.dp import pate

data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=clf.teacher_preds,
                                                   indices=y_public,
                                                   noise_eps=0.1, delta=1e-5)
print("Data Independent Epsilon:", data_ind_eps)
print("Data Dependent Epsilon:", data_dep_eps)
```
