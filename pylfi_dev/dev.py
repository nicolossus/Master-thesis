'''
https://numpy.org/doc/stable/reference/typing.html#module-numpy.typing

>>> import numpy as np
>>> import numpy.typing as npt

>>> def as_array(a: npt.ArrayLike) -> np.ndarray:
...     return np.array(a)

The ArrayLike type tries to avoid creating object arrays. For example,

>>> np.array(x**2 for x in range(10))
array(<generator object <genexpr> at ...>, dtype=object)

is valid NumPy code which will create a 0-dimensional object array. Type
checkers will complain about the above example when using the NumPy types
however. If you really intended to do the above, then you can either use
a # type: ignore comment:

>>> np.array(x**2 for x in range(10))  # type: ignore
'''
