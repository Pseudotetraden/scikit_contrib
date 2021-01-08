import pytest

from sklearn.utils.estimator_checks import check_estimator

from skltemplate._template import TemplateEstimator
from skltemplate._template import TemplateTransformer
from skltemplate.rbfn_without_keras import RadialBasisFunctionNetwork as TemplateClassifier


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
