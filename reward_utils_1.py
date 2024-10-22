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
start = time.time()

def TSV_Z_parameter(via_array, freq=np.arange(1e8, 4e10+1e8, step=1e8) ):

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
    #print(via_array)
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
    L_matrix = [[0] * (via_n - 1) for i in range(via_n - 1)]

    for indx1, via1 in enumerate(via_info):
        for indx2, via2 in enumerate(via_info):
            via1_distance = math.sqrt((via1[0] - via_ref[0]) ** 2 + (via1[1] - via_ref[1]) ** 2)
            via2_distance = math.sqrt((via2[0] - via_ref[0]) ** 2 + (via2[1] - via_ref[1]) ** 2)
            pitch = math.sqrt((via1[0] - via2[0]) ** 2 + (via1[1] - via2[1]) ** 2)

            if indx1 == indx2:
                L_matrix[indx1][indx2] = u_0 * u_TSV / math.pi * h_tsv * math.log(2 * via1_distance / d_tsv)
            else:
                L_matrix[indx1][indx2] = u_0 * u_TSV / math.pi / 2 * h_tsv * math.log(
                    2 * via1_distance * via2_distance / pitch / d_tsv)

    L_matrix_n = np.array(L_matrix)
    L_a =  L_matrix_n[0:via_num[2],0:via_num[2]]
    L_b =  L_matrix_n[0:via_num[2],via_num[2]:]
    L_c = L_matrix_n[via_num[2]:, 0:via_num[2]]
    L_d = L_matrix_n[via_num[2]:, via_num[2]:]


    L_d_inv = np.linalg.solve(L_d,np.eye(L_d.shape[0]))
    Leff = L_a - np.dot(np.dot(L_b, L_d_inv), L_c)

    #print("Leff:", Leff)
    #print(L_a)


    # C Calculation
    C = u_0 * e_si * e_0 * h_tsv ** 2 * np.linalg.pinv(Leff, rcond=1e-10)
    C_insulator = 1 / 2 * (2 * math.pi * e_0 * e_ox * (h_tsv - h_IMD) / math.log((d_tsv / 2 + t_ox) / (d_tsv / 2)))
    # print("Cs",C_insulator)
    Ceff = np.zeros(C.shape)
    for indx1, c1 in enumerate(Ceff):
        for indx2, c2 in enumerate(Ceff):  #????? Ceff right?? not C1?
            if indx1 == indx2:
                Ceff[indx1][indx2] = sum(C[indx1])
            else:
                Ceff[indx1][indx2] = -C[indx1][indx2]

    C_insulator = 1 / 2 * (2 * math.pi * e_0 * e_ox * (h_tsv - h_IMD) / math.log((d_tsv / 2 + t_ox) / (d_tsv / 2)))
    #print("Cs", C_insulator)
    #print("Ceff", Ceff)
    # G Calculation
    Geff = Ceff * sigma_si / e_si / e_0
    #print("Geff", Geff)

    # R Calculation
    Rdc_v = rho_tsv * h_tsv / math.pi / (d_tsv / 2) ** 2
    Rac_v = h_tsv*np.sqrt(freq*u_0*rho_tsv/math.pi)/d_tsv
    R_s = np.sqrt(Rdc_v ** 2 + Rac_v ** 2)
    R_gp = R_s / (via_num[0]+via_num[1])
    #print("Rdc",Rdc_v)

    ## 3. Make Impedance Array
    sig_n = via_num[2]
    Z_parameter = []


    for freq_i,freq_v in enumerate(freq):

        Impedance_A = np.zeros((sig_n*2, sig_n*2),dtype=complex)
        Impedance_C = np.zeros(((sig_n**3-sig_n)//6, sig_n*2),dtype=complex)
        Impedance_D = np.zeros(((sig_n**3-sig_n)//6,(sig_n**3-sig_n)//6),dtype=complex)

        for indx_x, m in enumerate(Impedance_A):
            for indx_y, n in enumerate(m):
                if (indx_x%2==0) & (indx_y==indx_x): # R,CS,Y
                    Impedance_A[indx_x, indx_y] = Impedance_A[indx_x,indx_y] + R_s[freq_i] +1/(Ceff[indx_x//2,indx_y//2]*1j*freq_v*2*math.pi+Geff[indx_x//2,indx_y//2])+1/1j/freq_v/C_insulator/2/math.pi
                    Impedance_A[indx_x+1, indx_y] = Impedance_A[indx_x+1,indx_y] +1/(Ceff[indx_x//2,indx_y//2]*1j*freq_v*2*math.pi+Geff[indx_x//2,indx_y//2])+1/1j/freq_v/C_insulator/2/math.pi
                    Impedance_A[indx_x, indx_y+1] = Impedance_A[indx_x,indx_y+1] +1/(Ceff[indx_x//2,indx_y//2]*1j*freq_v*2*math.pi+Geff[indx_x//2,indx_y//2])+1/1j/freq_v/C_insulator/2/math.pi
                    Impedance_A[indx_x+1, indx_y+1] = Impedance_A[indx_x+1,indx_y+1] +1/(Ceff[indx_x//2,indx_y//2]*1j*freq_v*2*math.pi+Geff[indx_x//2,indx_y//2])+1/1j/freq_v/C_insulator/2/math.pi
                elif indx_x%2==1 & indx_y%2==1: # mutual & self L
                    # Impedance_A[indx_x,indx_y] = Impedance_A[indx_x,indx_y]+(2*(indx_x==indx_y)-1)*Leff[indx_x//2,indx_y//2]*1j*freq_v
                    Impedance_A[indx_x, indx_y] = Impedance_A[indx_x, indx_y] + Leff[
                        indx_x // 2, indx_y // 2] * complex(0, 1) * freq_v *2*math.pi
                Impedance_A[indx_x, indx_y] = Impedance_A[indx_x, indx_y] + 1/1j/freq_v/C_insulator/2/math.pi

        comb = list(combinations(range(0,sig_n+1,1), 3)) # 0 ~ sig_n-1 : signal, sig_n : GND

        for indx_x,a in enumerate(comb):
            for indx_y,b in enumerate(comb):

                c = list(set(a).intersection(b))
                if len(c) == 3:
                    for k in list(combinations(c,2)):
                        if not sig_n in k:
                            Impedance_D[indx_x, indx_y] = Impedance_D[indx_x, indx_y] + 1 / (
                                        Ceff[k] * complex(0, 1) * freq_v*2*math.pi + Geff[k])
                        else:
                            Impedance_D[indx_x, indx_y] = Impedance_D[indx_x, indx_y] + 1 / (
                                    Ceff[min(k),min(k)] * complex(0, 1) * freq_v*2*math.pi + Geff[min(k),min(k)])
                elif len(c) == 2:
                    sign = 2*((a.index(c[1])-a.index(c[0])) == (b.index(c[1])-b.index(c[0])))-1
                    if not sig_n in c:
                        Impedance_D[indx_x, indx_y] = Impedance_D[indx_x, indx_y] + sign / (
                                Ceff[tuple(c)] * complex(0, 1)*2*math.pi * freq_v + Geff[tuple(c)])
                    else:
                        Impedance_D[indx_x, indx_y] = Impedance_D[indx_x, indx_y] + sign / (
                                Ceff[min(c), min(c)] * complex(0, 1)*2*math.pi * freq_v + Geff[min(c), min(c)])

            if sig_n in a:
                Impedance_C[indx_x,2*a[0]:2*a[0]+2] = -1 / (
                                Ceff[a[0], a[0]] * complex(0, 1)*2*math.pi * freq_v + Geff[a[0], a[0]])
                Impedance_C[indx_x, 2*a[1]:2*a[1] + 2] = 1 / (
                        Ceff[a[1],a[1]] * complex(0, 1) * freq_v*2*math.pi + Geff[a[1],a[1]])
        Impedance_B = np.transpose(Impedance_C)



        Impedance_D_inv = np.linalg.pinv(Impedance_D, rcond=1e-10)

        Z_parameter.append(Impedance_A - np.dot(np.dot(Impedance_B, Impedance_D_inv), Impedance_C))


    return Z_parameter #,As,Bs,Cs,Ds,R_s,Leff

def Cap_Zparameter(Cp,sig_num,freq):

    Z_parameter = []
    for freq_i, freq_v in enumerate(freq):
        cap_z = np.ones([2,2])*1/2/math.pi/freq_v/Cp*(-1j)
        Z_parameter.append(np.kron(np.eye(sig_num,dtype=complex),cap_z))
    return Z_parameter

def Z2S(Z_parameter,source,load):
    dim = Z_parameter[0].shape[0]
    S_parameter = []
    for freq_i,freq_v in enumerate(Z_parameter):
        # keunwoo library
        #ZZ_inv = np.array(getMatrixInverse((1 / load * Z_parameter[freq_i] + np.identity(dim))))

        # numpy library
        # ZZ_inv = np.linalg.solve(1 / load * Z_parameter[freq_i] + np.identity(dim),np.eye(dim))

        Zd0 = np.diag([1/math.sqrt(source), 1/math.sqrt(load)] * (len(Z_parameter[0]) // 2))

        ZZ_inv = np.linalg.pinv( np.matmul(np.matmul(Zd0,Z_parameter[freq_i]),Zd0) + np.identity(dim), rcond=1e-10)

        S_p = np.dot((np.matmul(np.matmul(Zd0,Z_parameter[freq_i]),Zd0) - np.identity(dim)), ZZ_inv)
        S_parameter.append(S_p)
        # if freq_i>50 and np.sum(abs(S_parameter[-1]-S_p)) > 10*np.sum(abs(S_parameter[-1]-S_parameter[-2])):
        #     S_parameter.append(S_parameter[-1])
        # else:
        #     S_parameter.append(S_p)
        # print(1 / load * Z_parameter[freq_i] - np.identity(dim))
        # print(np.linalg.inv( 1 / load * Z_parameter[freq_i] + np.identity(dim)))
        # print(np.dot((1 / load * Z_parameter[freq_i] - np.identity(dim)),
        #                           np.linalg.inv(1 / load * Z_parameter[freq_i] + np.identity(dim))))
    return S_parameter

def S2Z(S_parameter,source,load):
    Zd0 = np.diag([math.sqrt(source),math.sqrt(load)]*(S_parameter.shape[1]//2))
    Z_parameter = np.matmul(np.matmul(Zd0,np.matmul(np.eye(S_parameter.shape[1])+S_parameter,np.linalg.pinv(np.eye(S_parameter.shape[1])-S_parameter))),Zd0, rcond=1e-10)
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

def T2S(T_parameter,inputs,outputs):
    ## # of input & output ports are same

    T11 = T_parameter[:, 0:T_parameter.shape[1] // 2, 0:T_parameter.shape[2] // 2]
    T21 = T_parameter[:, T_parameter.shape[1] // 2:, 0:T_parameter.shape[2] // 2]
    T12 = T_parameter[:, 0:T_parameter.shape[1] // 2, T_parameter.shape[2] // 2:]
    T22 = T_parameter[:, T_parameter.shape[1] // 2:, T_parameter.shape[2] // 2:]

    inv_T22 = np.linalg.pinv(T22, rcond=1e-10)

    S_parameter = np.zeros([T_parameter.shape[0], T_parameter.shape[1], T_parameter.shape[2]],dtype=complex)

    S_parameter[:, 0:S_parameter.shape[1] // 2, 0:S_parameter.shape[2] // 2] = np.matmul(T12, inv_T22)
    S_parameter[:, S_parameter.shape[1] // 2:, 0:S_parameter.shape[2] // 2] = inv_T22
    S_parameter[:, 0:S_parameter.shape[1] // 2, S_parameter.shape[2] // 2:] = T11 - np.matmul(np.matmul(T12, inv_T22),T21)
    S_parameter[:, S_parameter.shape[1] // 2:, S_parameter.shape[2] // 2:] = -1*np.matmul(inv_T22,T21)

    S_parameter = S_parameter[:,np.argsort(inputs+outputs).tolist(),:]
    S_parameter = S_parameter[:,:,np.argsort(inputs+outputs).tolist()]

    return S_parameter

def multi_matmul(list_mul):
    if len(list_mul) == 1:
        return list_mul.pop()
    result = np.matmul(list_mul.pop(),multi_matmul(list_mul))
    return result
    # usage
    # A= np.array([ [1,2],[3,4]]) or A=[ [1,2],[3,4]] 으로 행렬 정의
    # multi_matmul([A,B,C]) 이런 식으로 정의하면 됨.
    # 결과: C*B*A 임 (ABC 아닌것 유의)

def make_via_channel(n_stack,T_para,T_cap,T_termination):

    T_cascade = np.matmul(multi_matmul([T_cap, T_para]*n_stack),T_termination)
    return T_cascade

def S2tf(S_parameter,direction,source,load):
    S = np.array(S_parameter) #shape =  freq*2*2 (S11 S12 S21 S22)
    if direction == '21':
        tf = S_parameter/2*math.sqrt(load/source)
    elif direction == '12':
        tf = S_parameter/2*math.sqrt(source/load)
    return tf

def S2tf_50_inf(S_parameter,in_z,out_z): # (source 50, load inf)
    S = np.array(S_parameter)
    S21 = S[:,out_z,in_z]
    S22 = S[:,out_z,out_z]
    TF = np.divide(S21,(1-S22))
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