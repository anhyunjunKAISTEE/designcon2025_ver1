# editor: Keunwoo Kim
# date: 20230501
# extract Z parameter and eye diagram from TSV array imformation
## via information: -1: ref, 0: blank, 1: ground, 2: power, 3: signal

import numpy as np
import math
import time
from itertools import combinations
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import pickle
import torch


# all numpy computation should be converted into torch computation

start = time.time()

def TSV_Z_parameter(via_array, freq=torch.arange(1e8, 4e10+1e8, step=1e8)):
    # Move freq to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # print("******************")
    # print("fuck")
    freq = freq.to(device)
    
    ## 1. Define parameters
    u_0 = 4 * math.pi * 1e-7
    u_TSV = 1
    e_0 = 8.85 * 10 ** -12
    e_si = 11.9
    e_ox = 4
    rho_tsv = 1.68e-8
    sigma_si = 10
    h_IMD = 5e-6
    t_ox = 0.5e-6
    p_tsv = 50e-6 
    h_tsv = 50e-6 
    d_tsv = 5e-6

    ## 2. Calculate RLGC values
    # via array to information
    via_info = []
    via_sig = []
    via_num = [0,0,0] #[P,G,S] or [Nan, G+P,S]
    print(via_array)
    
    for indx_y, vias in enumerate(via_array):
        for indx_x, via in enumerate(vias):
            if via == 3:
                via_info.append([indx_x * p_tsv, indx_y * p_tsv, via])
                via_sig.append([indx_x * p_tsv, indx_y * p_tsv, via])
                via_num[2] = via_num[2] + 1
                via_array[indx_y][indx_x]=0
                # via_info.append([indx_x, indx_y, via])
    for indx_y, vias in enumerate(via_array):
        for indx_x, via in enumerate(vias):
            if via:
                via_info.append([indx_x*p_tsv, indx_y*p_tsv, via])
                via_num[0] = via_num[0] + 1
                #via_info.append([indx_x, indx_y, via])

    # set reference via
    for via in via_info:
        if via[2] == 1:
            via[2] = -1
            via_ref = via
            via_info.remove(via)
            break
        # error detect: No ground via
        if via == via_info[-1]:
            print("ERROR: No Ground Via")
            return



    # L Calculation
    via_n = len(via_info) + 1
    L_matrix = torch.zeros((via_n - 1, via_n - 1), device=device)

    for indx1, via1 in enumerate(via_info):
        for indx2, via2 in enumerate(via_info):
            via1_distance = torch.sqrt(torch.tensor((via1[0] - via_ref[0]) ** 2 + (via1[1] - via_ref[1]) ** 2, device=device))
            via2_distance = torch.sqrt(torch.tensor((via2[0] - via_ref[0]) ** 2 + (via2[1] - via_ref[1]) ** 2, device=device))
            pitch = torch.sqrt(torch.tensor((via1[0] - via2[0]) ** 2 + (via1[1] - via2[1]) ** 2, device=device))


            if indx1 == indx2:
                L_matrix[indx1][indx2] = u_0 * u_TSV / math.pi * h_tsv * math.log(2 * via1_distance / d_tsv)
            else:
                L_matrix[indx1][indx2] = u_0 * u_TSV / math.pi / 2 * h_tsv * math.log(
                    2 * via1_distance * via2_distance / pitch / d_tsv)

    L_a =  L_matrix[0:via_num[2],0:via_num[2]]
    L_b =  L_matrix[0:via_num[2],via_num[2]:]
    L_c = L_matrix[via_num[2]:, 0:via_num[2]]
    L_d = L_matrix[via_num[2]:, via_num[2]:]


    L_d_inv = torch.linalg.solve(L_d, torch.eye(L_d.shape[0], device=device))
    Leff = L_a - torch.matmul(torch.matmul(L_b, L_d_inv), L_c)

    #print("Leff:", Leff)
    #print(L_a)


    # C Calculation
    C = u_0 * e_si * e_0 * h_tsv ** 2 * torch.linalg.pinv(Leff, rcond=1e-10)
    C_insulator = 1 / 2 * (2 * math.pi * e_0 * e_ox * (h_tsv - h_IMD) / math.log((d_tsv / 2 + t_ox) / (d_tsv / 2)))
    
    Ceff = torch.zeros(C.shape, device=device)
    for indx1 in range(Ceff.shape[0]):
        for indx2 in range(Ceff.shape[1]):
            if indx1 == indx2:
                Ceff[indx1][indx2] = torch.sum(C[indx1])
            else:
                Ceff[indx1][indx2] = -C[indx1][indx2]

    C_insulator = 1 / 2 * (2 * math.pi * e_0 * e_ox * (h_tsv - h_IMD) / math.log((d_tsv / 2 + t_ox) / (d_tsv / 2)))
    Geff = Ceff * sigma_si / e_si / e_0

    # R Calculation
    Rdc_v = rho_tsv * h_tsv / math.pi / (d_tsv / 2) ** 2
    Rac_v = h_tsv * torch.sqrt(freq * u_0 * rho_tsv / math.pi) / d_tsv
    R_s = torch.sqrt(Rdc_v ** 2 + Rac_v ** 2)
    R_gp = R_s / (via_num[0] + via_num[1])

    ## 3. Make Impedance Array
    sig_n = via_num[2]
    Z_parameter = torch.zeros((len(freq), sig_n*2, sig_n*2), dtype=torch.complex64, device=device)

    for freq_i, freq_v in enumerate(freq):
        Impedance_A = torch.zeros((sig_n*2, sig_n*2), dtype=torch.complex64, device=device)
        Impedance_C = torch.zeros(((sig_n**3-sig_n)//6, sig_n*2), dtype=torch.complex64, device=device)
        Impedance_D = torch.zeros(((sig_n**3-sig_n)//6, (sig_n**3-sig_n)//6), dtype=torch.complex64, device=device)

        for indx_x in range(Impedance_A.shape[0]):
            for indx_y in range(Impedance_A.shape[1]):
                if (indx_x % 2 == 0) and (indx_y == indx_x):
                    Impedance_A[indx_x, indx_y] = (Impedance_A[indx_x,indx_y] + R_s[freq_i] +
                        1/(Ceff[indx_x//2,indx_y//2]*1j*freq_v*2*math.pi+Geff[indx_x//2,indx_y//2])+
                        1/1j/freq_v/C_insulator/2/math.pi)
                    Impedance_A[indx_x+1, indx_y] = (Impedance_A[indx_x+1,indx_y] +
                        1/(Ceff[indx_x//2,indx_y//2]*1j*freq_v*2*math.pi+Geff[indx_x//2,indx_y//2])+
                        1/1j/freq_v/C_insulator/2/math.pi)
                    Impedance_A[indx_x, indx_y+1] = (Impedance_A[indx_x,indx_y+1] +
                        1/(Ceff[indx_x//2,indx_y//2]*1j*freq_v*2*math.pi+Geff[indx_x//2,indx_y//2])+
                        1/1j/freq_v/C_insulator/2/math.pi)
                    Impedance_A[indx_x+1, indx_y+1] = (Impedance_A[indx_x+1,indx_y+1] +
                        1/(Ceff[indx_x//2,indx_y//2]*1j*freq_v*2*math.pi+Geff[indx_x//2,indx_y//2])+
                        1/1j/freq_v/C_insulator/2/math.pi)
                elif (indx_x % 2 == 1) and (indx_y % 2 == 1):
                    Impedance_A[indx_x, indx_y] = (Impedance_A[indx_x, indx_y] + 
                        Leff[indx_x // 2, indx_y // 2] * 1j * freq_v * 2 * math.pi)
                Impedance_A[indx_x, indx_y] = Impedance_A[indx_x, indx_y] + 1/1j/freq_v/C_insulator/2/math.pi

        comb = list(combinations(range(0, sig_n+1, 1), 3))

        for indx_x, a in enumerate(comb):
            for indx_y, b in enumerate(comb):
                c = list(set(a).intersection(b))
                if len(c) == 3:
                    for k in list(combinations(c,2)):
                        if not sig_n in k:
                            Impedance_D[indx_x, indx_y] = (Impedance_D[indx_x, indx_y] + 
                                1 / (Ceff[k] * 1j * freq_v*2*math.pi + Geff[k]))
                        else:
                            Impedance_D[indx_x, indx_y] = (Impedance_D[indx_x, indx_y] + 
                                1 / (Ceff[min(k),min(k)] * 1j * freq_v*2*math.pi + Geff[min(k),min(k)]))
                elif len(c) == 2:
                    sign = 2*((a.index(c[1])-a.index(c[0])) == (b.index(c[1])-b.index(c[0])))-1
                    if not sig_n in c:
                        Impedance_D[indx_x, indx_y] = (Impedance_D[indx_x, indx_y] + sign / 
                            (Ceff[tuple(c)] * 1j*2*math.pi * freq_v + Geff[tuple(c)]))
                    else:
                        Impedance_D[indx_x, indx_y] = (Impedance_D[indx_x, indx_y] + sign / 
                            (Ceff[min(c), min(c)] * 1j*2*math.pi * freq_v + Geff[min(c), min(c)]))

            if sig_n in a:
                Impedance_C[indx_x,2*a[0]:2*a[0]+2] = -1 / (
                    Ceff[a[0], a[0]] * 1j*2*math.pi * freq_v + Geff[a[0], a[0]])
                Impedance_C[indx_x, 2*a[1]:2*a[1] + 2] = 1 / (
                    Ceff[a[1],a[1]] * 1j * freq_v*2*math.pi + Geff[a[1],a[1]])

        Impedance_B = torch.transpose(Impedance_C, 0, 1)
        Impedance_D_inv = torch.linalg.pinv(Impedance_D, rcond=1e-10)
        Z_parameter[freq_i] = Impedance_A - torch.matmul(torch.matmul(Impedance_B, Impedance_D_inv), Impedance_C)

    return Z_parameter

def Cap_Zparameter(Cp, sig_num, freq):
    """Calculate Z-parameters for capacitor using PyTorch
    
    Args:
        Cp: Capacitance value
        sig_num: Number of signals
        freq: Frequency array (torch tensor)
    
    Returns:
        Z_parameter: Tensor of Z-parameters
    """
    device = freq.device
    dtype = torch.complex64
    
    # Initialize Z_parameter as a tensor
    Z_parameter = torch.zeros((len(freq), sig_num*2, sig_num*2), dtype=dtype, device=device)
    
    # Create base capacitor Z matrix
    for freq_i, freq_v in enumerate(freq):
        cap_z = torch.ones(2, 2, dtype=dtype, device=device) * (1/2/math.pi/freq_v/Cp * (-1j))
        # Use kron product
        eye_matrix = torch.eye(sig_num, dtype=dtype, device=device)
        Z_parameter[freq_i] = torch.kron(eye_matrix, cap_z)
    
    return Z_parameter

def Z2S(Z_parameter, source, load):
    """Convert Z-parameters to S-parameters using PyTorch
    
    Args:
        Z_parameter: Tensor of Z-parameters
        source: Source impedance
        load: Load impedance
    
    Returns:
        S_parameter: Tensor of S-parameters
    """
    # Ensure input is a torch tensor
    if not isinstance(Z_parameter, torch.Tensor):
        Z_parameter = torch.tensor(Z_parameter)
    
    device = Z_parameter.device
    dtype = Z_parameter.dtype
    dim = Z_parameter.shape[1]  # Get dimension from first frequency point
    
    # Initialize S_parameter tensor
    S_parameter = torch.zeros_like(Z_parameter)
    
    # Create diagonal matrix
    diagonal_values = [1/math.sqrt(source), 1/math.sqrt(load)] * (dim // 2)
    Zd0 = torch.diag(torch.tensor(diagonal_values, dtype=dtype, device=device))
    
    # Create identity matrix
    identity = torch.eye(dim, dtype=dtype, device=device)
    
    # Process each frequency point
    for freq_i in range(len(Z_parameter)):
        # Calculate normalized impedance matrix
        normalized_Z = torch.matmul(torch.matmul(Zd0, Z_parameter[freq_i]), Zd0)
        
        # Calculate inverse using pseudoinverse
        ZZ_inv = torch.linalg.pinv(normalized_Z + identity, rcond=1e-10)
        
        # Calculate S-parameters
        S_parameter[freq_i] = torch.matmul(normalized_Z - identity, ZZ_inv)
    
    return S_parameter

def S2Z(S_parameter, source, load):
    """Convert S-parameters to Z-parameters using PyTorch
    
    Args:
        S_parameter: Torch tensor of S-parameters
        source: Source impedance
        load: Load impedance
        
    Returns:
        Z_parameter: Torch tensor of Z-parameters
    """
    # Ensure input is a torch tensor
    if not isinstance(S_parameter, torch.Tensor):
        S_parameter = torch.tensor(S_parameter)
        
    device = S_parameter.device
    dtype = S_parameter.dtype
    
    # Create diagonal matrix
    diagonal_values = [math.sqrt(source), math.sqrt(load)] * (S_parameter.shape[1]//2)
    Zd0 = torch.diag(torch.tensor(diagonal_values, dtype=dtype, device=device))
    
    # Create identity matrix
    eye = torch.eye(S_parameter.shape[1], dtype=dtype, device=device)
    
    # Calculate Z parameters
    Z_parameter = torch.matmul(
        Zd0,
        torch.matmul(
            torch.matmul(eye + S_parameter, torch.linalg.pinv(eye - S_parameter, rcond=1e-10)), Zd0)
    )
    
    return Z_parameter

def S2T(S_parameter,inputs,outputs):
    ## # of input & output ports are same

    S_ = np.array(S_parameter)
    S_ = S_[:, inputs+outputs, :]
    S_ = S_[:, :, inputs + outputs]

    S11 = S_[:, 0:S_.shape[1] // 2 , 0:S_.shape[2] // 2 ]
    S21 = S_[:,S_.shape[1] // 2:, 0:S_.shape[2] // 2 ]
    S12 = S_[:, 0:S_.shape[1] // 2 , S_.shape[2] // 2:]
    S22 = S_[:, S_.shape[1] // 2:, S_.shape[2] // 2:]

    inv_S21 = np.linalg.pinv(S21, rcond=1e-10)
    T_parameter = np.zeros([S_.shape[0], S_.shape[1], S_.shape[2]],dtype=complex)
    T_parameter[:, 0:S_.shape[1] // 2, 0:S_.shape[2] // 2 ] = S12 - np.matmul(np.matmul(S11, inv_S21), S22)
    T_parameter[:,S_.shape[1] // 2:, 0:S_.shape[2] // 2 ] = - 1 * np.matmul(inv_S21, S22)
    T_parameter[:, 0:S_.shape[1] // 2 , S_.shape[2] // 2:] = np.matmul(S11,inv_S21)
    T_parameter[:, S_.shape[1] // 2:, S_.shape[2] // 2:] = inv_S21

    return T_parameter

def S2T(S_parameter, inputs, outputs):
    """Convert S-parameters to T-parameters using PyTorch
    
    Args:
        S_parameter: Torch tensor of S-parameters
        inputs: List of input port indices
        outputs: List of output port indices
        
    Returns:
        T_parameter: Torch tensor of T-parameters
    """
    # Ensure input is a torch tensor
    if not isinstance(S_parameter, torch.Tensor):
        S_parameter = torch.tensor(S_parameter)
        
    device = S_parameter.device
    dtype = S_parameter.dtype
    
    # Select required ports
    # Convert inputs and outputs to tensor for indexing
    input_output_indices = torch.tensor(inputs + outputs, device=device)
    S_ = S_parameter[:, input_output_indices][:, :, input_output_indices]
    
    # Split into submatrices
    half_size = S_.shape[1] // 2
    S11 = S_[:, :half_size, :half_size]
    S21 = S_[:, half_size:, :half_size]
    S12 = S_[:, :half_size, half_size:]
    S22 = S_[:, half_size:, half_size:]
    
    # Calculate inverse of S21
    inv_S21 = torch.linalg.pinv(S21, rcond=1e-10)
    
    # Initialize T-parameter tensor
    T_parameter = torch.zeros_like(S_)
    
    # Calculate T-parameters
    T_parameter[:, :half_size, :half_size] = S12 - torch.matmul(
        torch.matmul(S11, inv_S21),
        S22
    )
    T_parameter[:, half_size:, :half_size] = -torch.matmul(inv_S21, S22)
    T_parameter[:, :half_size, half_size:] = torch.matmul(S11, inv_S21)
    T_parameter[:, half_size:, half_size:] = inv_S21
    
    return T_parameter

def multi_matmul(list_mul):
    """
    Multiply matrices in reverse order (CBA order) using recursion
    
    Args:
        list_mul: List of torch tensors to multiply
        
    Returns:
        Result of matrix multiplication in reverse order ABC -> CBA (correction 필수적으로 해야할듯)
    """
    if len(list_mul) == 1:
        return list_mul.pop()
    # Pop first matrix and multiply with recursive result (ensures CBA order)
    result = torch.matmul(list_mul.pop(), multi_matmul(list_mul))
    return result

def make_via_channel(n_stack, T_para, T_cap, T_termination):
    """
    Create via channel by cascading T-parameters
    
    Args:
        n_stack: Number of stacks
        T_para: Via T-parameters
        T_cap: Capacitor T-parameters
        T_termination: Termination T-parameters
        
    Returns:
        T_cascade: Cascaded T-parameters
    """
    # Ensure all inputs are torch tensors
    if not isinstance(T_para, torch.Tensor):
        T_para = torch.tensor(T_para)
    if not isinstance(T_cap, torch.Tensor):
        T_cap = torch.tensor(T_cap)
    if not isinstance(T_termination, torch.Tensor):
        T_termination = torch.tensor(T_termination)
    
    # Create list of matrices to multiply
    # We use list.copy() to avoid modifying the original list in multi_matmul
    cascade_list = [T_cap, T_para] * n_stack
    
    # Perform cascade multiplication (CBA order due to multi_matmul implementation)
    T_cascade = torch.matmul(multi_matmul(cascade_list.copy()), T_termination)
    
    return T_cascade

def S2tf(S_parameter, direction, source, load):
    """
    Convert S-parameters to transfer function using PyTorch
    
    Args:
        S_parameter: Torch tensor of S-parameters (shape: freq*2*2)
        direction: Direction of transfer ('21' or '12')
        source: Source impedance
        load: Load impedance
        
    Returns:
        tf: Transfer function
    """
    # Ensure input is a torch tensor
    if not isinstance(S_parameter, torch.Tensor):
        S_parameter = torch.tensor(S_parameter)
    
    if direction == '21':
        tf = S_parameter * math.sqrt(load/source) / 2
    elif direction == '12':
        tf = S_parameter * math.sqrt(source/load) / 2
    else:
        raise ValueError("direction must be either '21' or '12'")
        
    return tf

def S2tf_50_inf(S_parameter, in_z, out_z):
    """
    Convert S-parameters to transfer function for 50Ω source and infinite load
    
    Args:
        S_parameter: Torch tensor of S-parameters
        in_z: Input port index
        out_z: Output port index
        
    Returns:
        TF: Transfer function
    """
    # Ensure input is a torch tensor
    if not isinstance(S_parameter, torch.Tensor):
        S_parameter = torch.tensor(S_parameter)
        
    # Extract S21 and S22
    S21 = S_parameter[:, out_z, in_z]
    S22 = S_parameter[:, out_z, out_z]
    
    # Calculate transfer function
    # Using torch.divide for better numerical stability
    TF = torch.divide(S21, 1 - S22)
    
    return TF




## 이해 완료 ##
def get_Impulse(transfer_function,freq, mode = 1):

    # transfer function은 -1~1까지의 값이 들어있음. (len=400)
    # freq도 마찬가지 L=400
    # Transforer function 한 칸이 0.1GHz 이므로, 1/0.1GHz = 10ns (time축에서 10ns 단위임)
    # Transfer function은 Real signal 특성: Hermiltian: X(-f)=X*(f)인 특징이있음.
    # Impulse main은 

    
    f_s = freq[-1]*2 # 2f_max까지 (80GHz)
    L = len(freq)*2-1 # 2L-1=799 
    time = np.linspace(0,L-1,L)/f_s # 0.1GHZ 단위일때 -> time: 0~ 10ns 정도로 scale됨. 

    if mode ==1:
        filter_sym = np.r_[transfer_function[:-1], np.array([0]), np.flip(np.conjugate(transfer_function[:-1]))] # (799)
        #filter_sym: 해석:
        # tf = [ a b c d] 일 경우, np.r_(연결) 후 [a b c 0 c* b* a*] 이 됨. 
        # 즉 [원래 주파수 대역, 0, 복소수 켤레 대역]
    else: 
        filter_sym = np.r_[np.flip(np.conjugate(transfer_function[:-1])), np.array([0]), transfer_function[:-1]]
        ## [ a b c d]인 경우에 [d* c* b* a* 0 a b c d] 로 변환 되어야 하는거 아니냐? 

    
    impulse_response = np.fft.ifft(filter_sym).real # (799)
    return impulse_response,time

## 이해 완료 ## : time 간격 더 촘촘(f_time을 new 시간 간격)하게 해서 smooth하게 만드는 용도임. (일종의 기교)
def get_interpolation_function(func,time,f_time):
    # time 간격 더 촘촘하게 해서 new tf , new_time 만드는거임. 
    new_time = np.arange(time[0],time[-1],f_time) # time[0]부터 time[-1]까지 f_time 간격으로 새로운 time을 만듦

    tf_function_interp = interp1d(time,func,kind='quadratic') # time, func을 quadratic interpolation으로 새로운 함수 만듦
    new_tf = tf_function_interp(new_time)

    return new_tf, new_time

## Ideal한 Input pulse 그리는 함수 (time domain에서) 
## time step까지 input으로 받아서 그 time step에 맞는 length에서 변환함.
## input의 time range 안에서 적절히 UI 고려해서 precursor 반영해서 함. postcursor는 크면 알아서 짤림.
def get_InputPulse(bit_pattern,f_op,Voltage,time,rise_time,bit_prev = 0):#rise_time=0.1 %
    InputPulse = np.zeros(len(time)) # not error at ones
    #InputPulse = np.zeros(len(bit_pattern)*int(1/(time[1]-time[0])/f_op))  # not error at ones
    t = time[1]-time[0]
    for indx,bit in enumerate(bit_pattern):
        if bit == bit_prev:
            InputPulse[int(indx / f_op / 2 / t):int((indx + rise_time) / f_op / 2 / t)] = bit*Voltage
        elif bit != bit_prev:
            InputPulse[int(indx / f_op / 2 / t):int((indx + rise_time) / f_op / 2 / t)] = Voltage*(1-bit+(2*bit-1)*np.linspace(0,1,int((indx+rise_time) / f_op / 2 / t)-int(indx / f_op / 2 / t)))

        InputPulse[int((indx + rise_time) / f_op / 2 / t):int((indx + 1) / f_op / 2 / t)] = bit * Voltage

        bit_prev=bit
    return InputPulse
## impulse response랑 input pulse (step function+pre/post cursors) conv해서 SBR 구하는 함수
def get_SBR(Impulse,input_pulse):

    SBR2 = np.convolve(input_pulse,Impulse)
    return SBR2


def get_WorstEye(SBR_main,SBR_FEXTs,time,f_op,Vop):

    time_step = time[1]-time[0]

    # make 101 Response
    Response_010_main = SBR_main-SBR_main[-1].copy()

    
    Response_101_main = Vop-Response_010_main
    Response_FEXTs=[]
    for FEXT in SBR_FEXTs:
        Response_010_FEXT = FEXT - FEXT[-1]
        Response_FEXTs.append(Response_010_FEXT)
    Response_FEXT = np.array(Response_FEXTs)

    #hj_modification 240814
    # fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    # axs[0,0].plot(Response_010_main)
    # axs[0,0].set_title("Response 010 main")
    # axs[0,1].plot(Response_101_main)
    # axs[0,1].set_title("Response 101 main")
    
    # get start,center, and end point of eye
    UI = 1/f_op/2
    UI_step = int(UI/(time[1]-time[0]))

    # center_indx = int(np.average(np.abs(Response_010_main-Response_101_main).argsort()[0:2]))
    center_indx = np.average(np.where((Response_010_main - Response_101_main)>0))
    start_indx = int(center_indx -UI_step*0.5)

    # find number of Pre/Post cursors
    num_Precursor = start_indx//UI_step

    error = 0.5e-2

    for i in range(len(time[start_indx:])//UI_step-1):
        if np.max(Response_010_main[start_indx+i*UI_step:start_indx+(i+1)*UI_step]) < error:
            num_Postcursor = i
            break
    num_Postcursor = 6
    # get eye matrix

    cursors_010_main = Response_010_main[start_indx-num_Precursor*UI_step:start_indx+(num_Postcursor+1)*UI_step].reshape(-1,UI_step)#(cursor indx, UI_step)
    Main_010_main = Response_010_main[start_indx:start_indx+UI_step]
    #cursors_101_main = Response_101_main[start_indx-num_Precursor*UI_step:start_indx + (num_Postcursor+1)* UI_step].reshape(-1,UI_step)
    FEXTs_010_main = Response_FEXT[:,start_indx - num_Precursor * UI_step:start_indx + (num_Postcursor + 1) * UI_step].reshape(len(SBR_FEXTs),-1,UI_step) # (FEXTnum, cursor indx, UI_step)
    
    #hj_modification 240814
    # fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # # 각 서브플롯에 데이터 그리기
    # for i in range(6):
    #     row = i // 3
    #     col = i % 3
    #     axs[row, col].plot(FEXTs_010_main[2, i, :])
    #     axs[row, col].set_title(f'FEXTs_010_main[2,{i},:]')
    #     axs[row, col].set_xlabel('X-axis label')  # X축 레이블 추가 (필요에 따라 변경 가능)
    #     axs[row, col].set_ylabel('Y-axis label')  # Y축 레이블 추가 (필요에 따라 변경 가능)

    # 레이아웃 조정
    # plt.tight_layout()
    # plt.show()

    # quit()

    ISI_010_n = np.sum(np.where(cursors_010_main[list(range(num_Precursor))+list(range(num_Precursor+1,num_Postcursor+num_Precursor+1)),:] > 0, 0, cursors_010_main[list(range(num_Precursor))+list(range(num_Precursor+1,num_Postcursor+num_Precursor+1)),:]), axis=0)
    #FEXTs_010_n = np.sum(np.sum(np.where(FEXTs_010_main[:,list(range(num_Precursor))+list(range(num_Precursor,num_Postcursor+num_Precursor+1)),:] > 0, -FEXTs_010_main[:,list(range(num_Precursor))+list(range(num_Precursor,num_Postcursor+num_Precursor+1)),:],FEXTs_010_main[:,list(range(num_Precursor))+list(range(num_Precursor,num_Postcursor+num_Precursor+1)),:]), axis=1),axis=0)
    FEXTs_010_n = np.sum(np.sum(np.where(FEXTs_010_main[:,list(range(num_Precursor))+list(range(num_Precursor,num_Postcursor+num_Precursor+1)),:] > 0, 0,FEXTs_010_main[:,list(range(num_Precursor))+list(range(num_Precursor,num_Postcursor+num_Precursor+1)),:]), axis=1),axis=0)
    FEXTs_010_p = np.sum(np.sum(np.where(FEXTs_010_main[:,list(range(num_Precursor))+list(range(num_Precursor,num_Postcursor+num_Precursor+1)),:] < 0, 0,FEXTs_010_main[:,list(range(num_Precursor))+list(range(num_Precursor,num_Postcursor+num_Precursor+1)),:]), axis=1),axis=0)
    #FEXTs_010_n = np.sum(np.sum(np.where(FEXTs_010_main[:,list(range(num_Precursor))+list(range(num_Precursor+1,num_Postcursor+num_Precursor+1)),:] > 0, 0,FEXTs_010_main[:,list(range(num_Precursor))+list(range(num_Precursor+1,num_Postcursor+num_Precursor+1)),:]), axis=1),axis=0)

    # get eye height & width
    eye_p = Main_010_main + ISI_010_n + FEXTs_010_n #- FEXTs_010_p
    eye_isi = Main_010_main + ISI_010_n
    #plt.plot(eye_p)
    #plt.show()
    eye_height = np.max(eye_p)*2-Vop
    eye_center_indx = np.argmax(eye_p)
    eye_below_half = np.where(eye_p-Vop/2 < 0)
    eye_start = np.max(np.where(eye_below_half>eye_center_indx,0,eye_below_half))
    eye_end = np.min(np.where(eye_below_half<eye_center_indx,len(eye_p),eye_below_half))

    eye_width = (eye_end-eye_start)*time_step
    # return eye_heigt,eye_width

    return eye_height,  eye_width,  eye_p,  eye_isi,  time_step*range(UI_step)



# if __name__ == "__main__":
#     via_array = [
#         [1, 1, 0, 1, 3],
#         [3, 1, 3, 1, 1],
#         [3, 3, 1, 1, 1]
#     ]
#     t1=time.time()
#     print(sum(TSV_Z_parameter(via_array)[5][5])) # 400 * 6 * 6
#     print("time consumption: ",time.time()-t1)
#     print()



import matplotlib.pyplot as plt
import numpy as np

def img_save(data, var_name):
    """
    Save the input data as a PNG image.

    Parameters:
    data (array-like): The data to plot.
    var_name (str): The name of the variable to use as the filename.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(data)

    # Set title and labels
    ax.set_title(f'{var_name} Plot')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    # Save the plot as a PNG image with the variable name as the filename
    filename = f'{var_name}.png'
    fig.savefig(filename)

    # Close the plot to free up memory
    plt.close(fig)


def img_save_2(x, y, filename="x_y"):
    """
    Save the input data as a PNG image.

    Parameters:
    x (array-like): The data for the x-axis.
    y (array-like): The data for the y-axis.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(x, y)

    # Set title and labels
    ax.set_title('x vs y Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Save the plot as a PNG image with the variable names as the filename
    filename = filename+'.png'
    fig.savefig(filename)

    # Close the plot to free up memory
    plt.close(fig)

if __name__ == "__main__":
    # Load tf_cascade and freq from the file
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
        tf_cascade = data['tf_cascade']
        freq = data['freq']

    [Impulse_main, time1] = get_Impulse(tf_cascade, freq, mode=1)
    [Impulse_main2, time2 ] = get_interpolation_function(Impulse_main,time1,1.25e-12)

    img_save_2(time1[:30],Impulse_main[:30], "raw resolution" )
    img_save_2(time2[:300],Impulse_main2[:300], "high resolution" )


    impulse_response = np.fft.ifft(tf_cascade).real # (799)


    f_s = freq[-1]# 2f_max까지 (80GHz)
    L = len(freq) # 2L-1=799 
    time = np.linspace(0,L-1,L)/f_s



    n_pre = 2; n_post= 20; bit_pattern = np.array([0]*n_pre+[1]+[0]*n_post)  # SBR, len=31. # [0 0 1 0 0 ..]

    V_op = 0.4
    f_op = 2e9
    InputPulse = get_InputPulse(bit_pattern, f_op, V_op, time1, 0.1)         #(799)

    SBR_main = get_SBR(Impulse_main, InputPulse) #f_op, time1[1] - time1[0])[:len(Impulse_main)] # 1597
    img_save(SBR_main, "SBR_main")
    img_save(InputPulse, "InputPulse")
    img_save(Impulse_main, "Impulse_main")