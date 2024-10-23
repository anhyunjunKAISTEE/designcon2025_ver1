## data_gen_reward.py 파일은 24.10.22.
## sig/ground via array 의 unique array에 대한 정보를 가지고 reward를 계산 (reward 전체 다 보여줌 sig별로)
## 이거 서버실행용임

## utils_reward1.py -> Zpara to eye (reward)
## utils_reward2.py -> treat via_config -> boosting reward calculation time 

import matplotlib
matplotlib.use('Agg', force=True)  # GUI가 필요없는 백엔드 사용
import matplotlib.pyplot as plt

import os
import numpy as np
import math
import time
from itertools import combinations
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from reward_utils_1_gpu import *
from reward_utils_2 import *
from datetime import datetime
import glob

def via_slow_reward(via_array, V_op=0.4, f_op=2e9, n_stack=16):
    
    ## parameter information
    V_op = V_op
    f_op = f_op
    n_stack = n_stack

    # parameter calculation
    sig_n = len(np.where(np.array(via_array) == 3)[0])  # sig 개수 
    input_ports = list(range(0, sig_n * 2, 2))          # [ 0, 2, 4] (전체개수=sig 개수)
    output_ports = list(range(1, sig_n * 2 + 1, 2))     # [ 1, 3, 5] (전체개수=sig 개수)
    freq=torch.arange(1e8, 4e10+1e8, step=1e8)     # 0.1G ~ 40GHz까지 400 step

    #######################
    ## Device to GPU ##
    # freq = freq.to(device)

    ###################################
    # (A) make Transfer Function (~S-para)
        # (1) unit TSV

    Z_tsv = TSV_Z_parameter(via_array, freq)      # (6*6*400) len=400, 개당 size=6*6 . 즉, 6*6이 400개 (why 6=2*signal개수)
    # print(Z_tsv)

    S_tsv = Z2S(Z_tsv, 50, 50)                    # (6*6*400)
    T_tsv = S2T(S_tsv, input_ports, output_ports) # (6*6*400)

        # (2) + cap
    Z_cap = Cap_Zparameter(0.2e-12, sig_n, freq)  # (6*6*400)
    S_cap = Z2S(Z_cap, 50, 50)                    # (6*6*400)
    T_cap = S2T(S_cap, input_ports, output_ports) # (6*6*400)
        # (3) termination
    S_termination = Z2S(Z_cap, 50, 100000)        # (6*6*400)
    T_termination = S2T(S_termination, input_ports, output_ports) # (6*6*400)
        # (4) cascading n stacks
    T_cascade = make_via_channel(n_stack, T_tsv, T_cap, T_termination) # (6*6*400)
    S_cascade = T2S(T_cascade, input_ports, output_ports)              # (6*6*400)
    
    # quit()
    ###################################
    # (B) SBR calculation 
    rewards = []
    SBR_mains = []
    
    for pin in range(sig_n): # sig 개수 마다 
        ###################################
        # (1) FEXT Calculation
        tf_cascade = S2tf(S_cascade[:, pin*2+1, pin*2], '21', 50, 100000)
        tf_FEXTs = []   # (sig개수-1 * 400) 
        for i in range(sig_n):
            if i != pin:
                tf_FEXT = S2tf(S_cascade[:,  i * 2 +1, pin*2], '21', 50, 100000) # (400)
                tf_FEXTs.append(tf_FEXT)

        # (2) Impulse response 
        [Impulse_main, time1] = get_Impulse(tf_cascade, freq) # (799), (799)

        Impulse_FEXTs = []
        for tf_FEXT in tf_FEXTs:
            [Impulse_FEXT, time1] = get_Impulse(tf_FEXT, freq)
            Impulse_FEXTs.append(Impulse_FEXT)
        
        n_pre = 2; n_post= 28; bit_pattern = np.array([0]*n_pre+[1]+[0]*n_post)  # SBR, len=31. # [0 0 1 0 0 ..]

        InputPulse = get_InputPulse(bit_pattern, f_op, V_op, time1, 0.1)         #(799)

        SBR_main = get_SBR(Impulse_main, InputPulse) #f_op, time1[1] - time1[0])[:len(Impulse_main)] # 1597
        SBR_main = SBR_main[:len(Impulse_main)] # hj add
        SBR_mains.append(SBR_main)
        
        SBR_FEXTs = []
        bit_pattern = np.array([0] *n_pre + [1] * 1 + [1] * n_post)  # XTLK # [0 0 1 1 1 1 ...]
        InputPulse = get_InputPulse(bit_pattern, f_op, V_op, time1, 0.1)
        
        for Impulse_FEXT in Impulse_FEXTs:
            SBR_FEXT = get_SBR(Impulse_FEXT, InputPulse) #, f_op, time1[1] - time1[0])[:len(Impulse_FEXT)]
            SBR_FEXT = SBR_FEXT[:len(Impulse_FEXT)] # hj add
            SBR_FEXTs.append(SBR_FEXT)
        
        ###################################
        ## savgol filtering -smooth하게 만들어 주기 위해서.. 근데 굳이? (안하니까 너무 꺾여서, 해야할듯)
        
        SBR_main = savgol_filter(SBR_main, 21, 5) # Savitzky-Golay filter : window 길이=21, 다항식 차수=5
                                            # Study 결과: order=1,2,5 증가-> SBR높이 up, eye 넓이 up, right shift
                                            # Study 결과: window 증감도 변화 요인임. 
        for indx, SBR_FEXT in enumerate(SBR_FEXTs):
            SBR_FEXTs[indx] = savgol_filter(SBR_FEXT, 15, 5)
        
        SBR_mains.append(SBR_main)


        ###################################
        time1_interp = np.linspace(0, time1[-1], 3 * len(time1))

        # time 1: len=799 / time1_interp = 2397 =3*799 / SBR_main = 1597 (->799)
        SBR_main = np.interp(time1_interp, time1, SBR_main) #보간하기 -> 2397
        for indx, SBR_FEXT in enumerate(SBR_FEXTs):
            SBR_FEXTs[indx] = np.interp(time1_interp, time1, SBR_FEXT)
        SBR_mains.append(SBR_main)
        # print(get_WorstEye(SBR_main, SBR_FEXTs, time1_interp,f_op,V_op))

        ###################################
        ## Eye diagram contour 얻기 
        
        [eye_height, eye_width, eye_p, eye_isi, time_step_rec] = get_WorstEye(SBR_main, SBR_FEXTs, time1_interp,f_op,V_op)

        #reward = (eye_height*eye_width)/(V_op*(1/V_op/2))
        reward = eye_height*eye_width
        rewards.append(reward)

        
        # tf_FEXTs 계산 직후
        # plot_transfer_functions(freq, tf_FEXTs, pin+1)
        # plot_impulse_responses(time1, Impulse_main, Impulse_FEXTs, pin+1, f_op=f_op, UI_scale=True)

    
    # print("rewards: ", rewards)
    return min(rewards), rewards

def reward(via_array, V_op=0.4, f_op=2e9, n_stack=16, fast=False , fast_img_save=False, size=2, scaling=False):


    if not fast:
        reward_value, _ = via_slow_reward(via_array, V_op, f_op, n_stack)
    else:
        outputs = via_config_all(via_array, size=size)
        selected_outputs = via_config_select(outputs)
        unique_selected_outputs = via_config_contraction(selected_outputs)

        reward_candidates= []
        for arrays in unique_selected_outputs:
            reward_candidates.append(via_slow_reward(arrays, V_op, f_op, n_stack))
        
        if fast_img_save:
            via_config_visual_all(via_array, imag=fast_img_save)

        reward_value = min(reward_candidates)
        
    if scaling:
        return 100*reward_value/(V_op/(2*f_op)), _
    else:
        return reward_value, _



if __name__ == "__main__":
    current_time = datetime.now().strftime('%m%d%H%M')
    #########################################################################################################
    ## User setting ##
    f_ops_Ghz = [ 2.0, 3.0]
    array_data_dir = 'data/via_arrays_config'
    target_files = ["4_4_8_500_arrays.npy", 
                    "5_5_12_500_arrays.npy",
                    "4_6_12_500_arrays.npy",
                    "6_6_18_500_arrays.npy", 
                    "6_8_24_500_arrays.npy",
                    "8_8_32_500_arrays.npy",
                    "8_10_40_500_arrays.npy",
                    "10_10_50_500_arrays.npy",
                     ]
    #########################################################################################################

    # for each frequency step
    for f_op in f_ops_Ghz:

        # if directory does not exist, create it
        reward_data_save_dir = 'data/via_arrays_reward'
        if not os.path.exists(reward_data_save_dir):
            os.makedirs(reward_data_save_dir)
            print(f"Created directory: {reward_data_save_dir}")

        # for each file in the target_files
        for file_name in target_files:

            # if directory does not exist, create it
            file_path = os.path.join(array_data_dir, file_name)
            if not os.path.exists(file_path):
                print(f"Warning: File {file_name} does not exist. Skipping...")
                continue
            # file name processing
            file_unique_name = file_name.replace('_arrays.npy', '')
            n_sig = int(file_unique_name.split('_')[2])
            via_arrays = np.load(file_path, allow_pickle=True)
            reward_arrays = []
            reward_list_like = [-1 for sigals in range(n_sig)]
            total_arrays = len(via_arrays)
            start = time.time()

            ###################################################################################
            # for already existing files, find the latest one and resume from there
            existing_files = glob.glob(os.path.join(reward_data_save_dir, f"{file_unique_name}_{f_op}_*_rewards.txt"))
            print(f"Found {len(existing_files)} existing files for {file_unique_name} at {f_op} GHz")
            start_index = 0
            if existing_files:
                latest_file = max(existing_files)
                print(f"Found existing file: {latest_file}" + f"among existing files: {existing_files}")
                with open(latest_file, 'r') as f:
                    start_index = len(f.readlines())
                print(f"Resuming from index {start_index}")

            # mod: 기존 데이터 로드 (있는 경우)
            if start_index > 0:
                with open(latest_file, 'r') as f:
                    for line in f:
                        reward_arrays.append(eval(line.strip()))  # 문자열을 리스트로 변환
                print(f"Loaded {len(reward_arrays)} existing results")
                
            ##################################################################################

            # reward calculation
            for i, via_array in enumerate(via_arrays):
                # print(reward_arrays)
                if i < start_index:
                    continue
                else:
                    try:
                        r, reward_list = reward(via_array, V_op=1.0, f_op=f_op*1e9, n_stack=16, fast=False, fast_img_save=False, size=2, scaling=False)
                        reward_arrays.append(reward_list)
                    except Exception as e:
                        print(f"Error at iteration {i} for {file_unique_name}: {str(e)}")
                        reward_arrays.append(reward_list_like)  # -1이 원소로 가득한 리스트 추가
                    
                    # quit()
                    # 100개 단위로 진행상황 출력
                    if (i + 1) % 1 == 0:
                        progress = (i + 1) / total_arrays * 100
                        print(f"{file_unique_name}: Processed {i + 1}/{total_arrays} arrays ({progress:.1f}%)")
                        print("time consumption during this cycle: ", time.time()-start)
                    
                    # txt 파일에 reward_arrays를 저장
                    reward_file_txt_name = f"{file_unique_name}"+f"_{f_op}_"+current_time+"_rewards.txt"
                    reward_file_txt_path = os.path.join(reward_data_save_dir, reward_file_txt_name)
                    with open(reward_file_txt_path, 'w') as f:
                        for reward_instance in reward_arrays:
                            f.write(str(reward_instance) + '\n')
                    # print(f"Saved rewards to {reward_file_txt_path}")
                
            
            # reward_arrays를 numpy 파일로 저장
            reward_file_name = f"{file_unique_name}"+f"_{f_op}_"+current_time+"_rewards.npy"
            reward_file_path = os.path.join(reward_data_save_dir, reward_file_name)

            np.save(reward_file_path, reward_arrays)
            print(f"Saved rewards to {reward_file_path}")
            



    