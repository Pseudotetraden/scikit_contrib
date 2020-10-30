from rbfn_with_keras import RadialBasisFunctionNetwork as rbfn_with
from rbfn_without_keras import RadialBasisFunctionNetwork as rbfn_without

class RBFN:
    def __new__(self, *args, **kwargs):
        try:
            import keras
            return rbfn_with(*args, **kwargs)
        except:
            return rbfn_without(*args, **kwargs)

rbfn = RBFN()
