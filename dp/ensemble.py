import itertools
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble._base import _partition_estimators
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_is_fitted,
    check_random_state,
    column_or_1d,
    check_array,
)

from typing import List, Optional, Tuple

import numpy as np


def _parallel_build_teachers(
    n_estimators: int,
    ensemble: BaseEnsemble,
    X: np.ndarray,
    y: np.ndarray,
    indices: List[np.ndarray],
    random_state: Optional[int],
) -> List[BaseEnsemble]:
    """
    Split a training dataset X in non overlapping partitions, and fit a teacher estimator
     on each of them

    :param n_estimators: number of teachers
    :param ensemble: a sklearn ensamble of estimators
    :param X: training set
    :param y: labels
    :param indices: list of indices that identify a partition
    :param random_state:
    :return: a list of trained estimators
    """
    estimators = []
    for i in range(n_estimators):
        estimator = ensemble._make_estimator(append=False, random_state=random_state)
        estimator.fit(X[indices[i]], y[indices[i]])
        estimators.append(estimator)
    return estimators


def _aggregate_teachers(
    X: np.ndarray,
    estimators: List[BaseEstimator],
    epsilon: float = 0.5,
    mechanism: str = "laplace",
    random_state: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate teacher predictions and apply Laplacian noise

    :param X: unlabelled dataset
    :param estimators: fitted estimators
    :param epsilon: epsilon budget
    :param mechanism: type of noise (default laplace)
    :param random_state:
    :return: (`predicted_labels`, 'student_labels').
        * `predicted_labels` is a (num_teachers, num_labels) ndarray
        of predictions made by each member of the ensemble.
        * `student_labels` is an nd.array of aggregated predictions.
    """
    if mechanism != "laplace":
        raise AttributeError(f"{mechanism} - not supported")

    np.random.seed(random_state)
    predicted_labels = np.zeros((len(estimators), X.shape[0]), dtype=np.int64)
    for i, estimator in enumerate(estimators):
        y_pred = estimator.predict(X)
        predicted_labels[i] = y_pred

    student_labels = np.array([]).astype(int)

    for labels in predicted_labels.T:
        label_counts = np.bincount(labels, minlength=2)
        beta = 1 / epsilon
        for i in range(len(label_counts)):
            label_counts[i] += np.random.laplace(0, beta, 1)
        # vote for the most frequent outcome
        label = np.argmax(label_counts)
        student_labels = np.append(student_labels, label)
    return predicted_labels, student_labels


class PrivateClassifier(BaseEnsemble, ClassifierMixin):
    def __init__(
        self,
        base_estimator=None,
        estimator_params=[],
        n_estimators=1,
        n_jobs=1,
        epsilon=0.5,
        random_state=None,
    ):
        self.base_estimator = base_estimator
        self.estimator_params = estimator_params
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.epsilon = epsilon
        self.random_state = random_state

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(default=LogisticRegression())

    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.label_encoder_ = LabelEncoder().fit(y)
        y = self.label_encoder_.transform(y)
        return y

    def _fit(self, X, y):
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            dtype=None,
            force_all_finite=False,
            multi_output=True,
        )
        self._validate_estimator()
        random_state = check_random_state(self.random_state)
        y = self._validate_y(y)

        n_jobs, n_estimators, starts = _partition_estimators(
            self.n_estimators, self.n_jobs
        )

        # TODO(gmodena): shuffle X before fitting?
        n_samples, _ = X.shape

        # non-overlapping indices for the teacher  datasets/estimators
        indices = np.split(np.array(range(n_samples)), self.n_estimators)
        if self.n_estimators == 1:
            indices = [indices]
        all_results = Parallel(n_jobs=n_jobs)(
            delayed(_parallel_build_teachers)(
                n_estimators[i],  # number of estimators / job
                self,
                X,
                y,
                indices[starts[i] : starts[i + 1]],
                random_state,
            )
            for i in range(n_jobs)
        )
        self.estimators_ = list(
            itertools.chain.from_iterable(estimator for estimator in all_results)
        )
        return self

    def _predict(self, X):
        check_is_fitted(
            self,
            [
                "estimators_",
            ],
        )
        X = check_array(X)
        self.teacher_preds_, y_pred = _aggregate_teachers(
            X, self.estimators_, self.epsilon, "laplace", self.random_state
        )
        return y_pred

    def fit(self, X, y):
        """
        build teacher models
        :param y:
        :return:
        """
        return self._fit(X, y)

    def _more_tags(self):
        tags = super()._more_tags()
        tags["poor_score"] = True
        return tags

    def predict(self, X):
        y_pred = self._predict(X)
        return self.label_encoder_.inverse_transform(y_pred)
