## data_gen_config.py 파일은 24.10.22.
## sig/ground via array 의 unique array를 생성하는 코드임
## 이걸 통해 via array를 생성하고, 이를 통해 reward를 계산하는 코드를 실행할거임


## utils_reward1.py -> Zpara to eye (reward)
## utils_reward2.py -> treat via_config -> boosting reward calculation time 

import matplotlib
matplotlib.use('Agg', force=True)  # GUI가 필요없는 백엔드 사용
import matplotlib.pyplot as plt

import numpy as np
import math
import time
from itertools import combinations
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from reward_utils_1 import *
from reward_utils_2 import *
import math 


import numpy as np
import matplotlib.pyplot as plt




# 테스트를 위한 메인 코드
if __name__ == "__main__":

    # 저장할 디렉토리 생성 (없는 경우)
    save_dir = 'data/via_arrays_config'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    
    size = [4, 6]  # 5x5 array
    num_sig = 12 # 3개의 signal via
    num_samples = 500  # 2개의 unique array 생성

    # 이론적으로 가능한 최대 unique 패턴 수 계산
    total_elements = size[0] * size[1]
    import math
    max_possible = math.comb(total_elements, num_sig)
    print(f"Theoretically possible maximum unique patterns: {max_possible}")
    
    # diff = [[[2,2], [2,3]], [[4,4], [4,3]], [[0,0], [1,0]]]
    
    arrays = via_array_generation(size, num_samples, num_sig) #, diff=diff)
    filename = save_dir+"/"+f"{size[0]}_{size[1]}_{num_sig}_{num_samples}_arrays.npy"
    np.save(filename, arrays)
    print(f"Arrays saved to '{filename}'")

    # 생성된 arrays를 시각화
    # via_config_visual(arrays, title=f"Generated Unique Via Arrays with Constraints")
    
    # print(arrays)

   








