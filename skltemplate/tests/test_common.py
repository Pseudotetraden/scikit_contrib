import pytest

from sklearn.utils.estimator_checks import check_estimator

from scikit_contrib import TemplateEstimator
from scikit_contrib import TemplateClassifier
from scikit_contrib import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
