import pytest
from dp.ensemble import PrivateClassifier

from sklearn.utils import estimator_checks

classifier_checks = (
    estimator_checks.check_classifier_data_not_an_array,
    estimator_checks.check_classifiers_one_label,
    estimator_checks.check_classifiers_classes,
    estimator_checks.check_estimators_partial_fit_n_features,
    estimator_checks.check_classifiers_train,
    estimator_checks.check_classifiers_regression_target,
    estimator_checks.check_supervised_y_no_nan,
    estimator_checks.check_supervised_y_2d,
    estimator_checks.check_estimators_unfitted,
    estimator_checks.check_decision_proba_consistency
)


@pytest.mark.parametrize("test_fn", classifier_checks)
def test_estimator_checks_classifier(test_fn):
    estimator = PrivateClassifier()
    name = type(estimator).__name__
    test_fn(name, estimator)