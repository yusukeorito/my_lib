import pandas as pd
import numpy as np

#targetが0以上の場合、負の予測値を0で丸める後処理
def round_positive(input:np.ndarray)-> np.ndarray:
    output = input.copy()
    output[output<0] = 0
    return output
