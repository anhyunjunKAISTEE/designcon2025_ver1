## ref: from keunwoo kim's code / mod: 240702 modificated by hyunjun An
## For Reward calculation 
## via array 조합중 worst case를 추려서 reward calculation시 모든 signal via에 대해서 평가하지 않도록 하기 위함임


import matplotlib.pyplot as plt
import numpy as np
import os

from reward_utils_1_gpu import *

def via_config_all(via_array, size=2):
    # Find all positions of the value 3
    positions = [(i, j) for i in range(len(via_array)) for j in range(len(via_array[i])) if via_array[i][j] == 3]
    
    # List to store all possible sub-arrays
    result = []
    
    # Iterate over each position of 3
    for pos in positions:
        # Create a deep copy of the original array
        new_array = [row[:] for row in via_array]
        # Replace the specific 3 with -1
        new_array[pos[0]][pos[1]] = -1
        
        # Extract the sub-array of size `size` around the replaced -1
        sub_array = []
        for i in range(pos[0] - size, pos[0] + size + 1):
            row = []
            for j in range(pos[1] - size, pos[1] + size + 1):
                if 0 <= i < len(new_array) and 0 <= j < len(new_array[0]):
                    row.append(new_array[i][j])
            if row:
                sub_array.append(row)
        
        # Append the sub-array to the result list
        result.append(sub_array)
    
    return result

def via_config_select(outputs):
    max_count = 0
    selected_sub_arrays = []

    for sub_array in outputs:
        count = sum(row.count(3) for row in sub_array)
        if count > max_count:
            max_count = count
            selected_sub_arrays = [sub_array]
        elif count == max_count:
            selected_sub_arrays.append(sub_array)
    
    return selected_sub_arrays

def via_util_rotate90(matrix):
    return [list(reversed(col)) for col in zip(*matrix)]

def via_util_rotate180(matrix):
    return [list(reversed(row)) for row in reversed(matrix)]

def via_util_rotate270(matrix):
    return [list(col) for col in zip(*matrix[::-1])]

def via_util_flip_horizontal(matrix):
    return [list(reversed(row)) for row in matrix]

def via_util_flip_vertical(matrix):
    return list(reversed(matrix))

def via_util_is_symmetric(matrix):
    return matrix == via_util_flip_horizontal(matrix) or matrix == via_util_flip_vertical(matrix)

def via_util_is_same_or_rotate(matrix1, matrix2):
    transformations = [
        lambda x: x,
        via_util_rotate90,
        via_util_rotate180,
        via_util_rotate270,
        via_util_flip_horizontal,
        via_util_flip_vertical
    ]
    
    for transform in transformations:
        transformed_matrix = transform(matrix2)
        for _ in range(4):
            if matrix1 == transformed_matrix:
                return True
            transformed_matrix = via_util_rotate90(transformed_matrix)
    
    return False

def via_config_contraction(selected_outputs):
    unique_matrices = []
    
    for matrix in selected_outputs:
        if not any(via_util_is_same_or_rotate(matrix, unique_matrix) for unique_matrix in unique_matrices):
            unique_matrices.append(matrix)
    
    return unique_matrices

def via_config_visual(via_arrays, title="Via Configuration", labels=None, output_dir = 'data/via_array_figure'):
    # Define the color map and corresponding labels
    cmap = {
        -1: 'yellow',
        0: 'white',
        1: 'gray',
        2: 'blue',
        3: 'red',
        4: 'green',
        5: 'black'
    }
    
    if labels is None:
        labels = {
            -1: 'Unknown',
            0: 'Empty',
            1: 'GND',
            2: 'Power',
            3: 'Signal',
            4: 'Diff+',
            5: 'Diff-'
        }
    
    num_arrays = len(via_arrays)
    max_cols = 5
    num_cols = min(num_arrays, max_cols)
    num_rows = (num_arrays + max_cols - 1) // max_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    
    if num_rows == 1:
        axes = [axes]
    if num_cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, via_array in enumerate(via_arrays):
        row = idx // max_cols
        col = idx % max_cols
        ax = axes[row][col]
        colored_array = np.array([[cmap[val] for val in row] for row in via_array])
        
        for i in range(len(via_array)):
            for j in range(len(via_array[i])):
                rect = plt.Rectangle((j, len(via_array) - i - 1), 1, 1, facecolor=colored_array[i][j])
                ax.add_patch(rect)
                ax.text(j + 0.5, len(via_array) - i - 0.5, str(via_array[i][j]), ha='center', va='center', color='black')
        
        ax.set_xlim(0, len(via_array[0]))
        ax.set_ylim(0, len(via_array))
        ax.set_xticks(range(len(via_array[0]) + 1))
        ax.set_yticks(range(len(via_array) + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for idx in range(num_arrays, num_rows * num_cols):
        row = idx // max_cols
        col = idx % max_cols
        fig.delaxes(axes[row][col])
    
    # Add a legend at the bottom
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[key]) for key in cmap]
    legend_labels = [f"{key}: {labels[key]}" for key in cmap]
    fig.legend(handles, legend_labels, loc='lower center', ncol=len(cmap))
    
    fig.suptitle(title)
    
    # Create directory if it doesn't exist
    output_dir = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the figure
    fig.savefig(os.path.join(output_dir, f"{title}.png"))
    # plt.show()

def via_config_visual_all(via_array, imag=True, size=2):
    outputs = via_config_all(via_array,size=size)
    selected_outputs = via_config_select(outputs)
    unique_selected_outputs = via_config_contraction(selected_outputs)

    print("via_array:",via_array)
    print("output_array:", outputs)
    print("selected_outputs:", selected_outputs)
    print("unique_selected_outputs:", unique_selected_outputs)


    if imag:
        via_config_visual([via_array], title="Via Configuration")
        via_config_visual(outputs, title="Visual of Outputs")
        via_config_visual(selected_outputs, title="Selected Outputs")
        via_config_visual(unique_selected_outputs, title="Unique Selected Outputs")
    

#############
#  FEXT and impulse response plot function
def plot_transfer_functions(freq, tf_FEXTs, pin_number):
    """
    Plot frequency domain transfer functions for FEXT
    """
    plt.figure(figsize=(12, 6))
    for i, tf_FEXT in enumerate(tf_FEXTs):
        # Convert to dB scale
        tf_FEXT_db = 20 * np.log10(np.abs(tf_FEXT))
        # plt.plot(freq/1e9, tf_FEXT_db, label=f'FEXT to Pin {i+1}')
        plt.semilogx(freq/1e9, tf_FEXT_db, label=f'FEXT to Pin {i+1}')
        
    plt.title(f'FEXT Transfer Functions from Pin {pin_number}')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot_tf_functions.png')

def plot_impulse_responses(time1, Impulse_main, Impulse_FEXTs, pin_number, f_op=2e9, UI_scale=True):

    if (UI_scale):
        """
        Plot time domain impulse responses for FEXT with UI units
        """
        target_UI = 3
        UI = 0.5/f_op  # Unit Interval 계산
        
        plt.figure(figsize=(12, 6))
        
        # time을 UI 단위로 변환
        time_UI = time1/UI
        
        # target UI까지의 인덱스 찾기
        idx_target_UI = np.where(time_UI <= target_UI)[0][-1]
        
        # target UI까지만 플롯
        plt.plot(time_UI[:idx_target_UI], Impulse_main[:idx_target_UI], 'k-', linewidth=2, label='Main Signal')
        for i, Impulse_FEXT in enumerate(Impulse_FEXTs):
            plt.plot(time_UI[:idx_target_UI], 
                    Impulse_FEXT[:idx_target_UI], 
                    label=f'FEXT to Pin {i+1}')
        
        plt.title(f'FEXT Impulse Responses from Pin {pin_number}')
        plt.xlabel('Time (UI)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.xlim([0, target_UI])  # x축 범위를 0-target UI로 제한
        plt.legend()
        plt.tight_layout()
        plt.savefig('plot_FEXTimp_response.png')

    else:
        """
        Plot time domain impulse responses for FEXT
        """
        plt.figure(figsize=(12, 6))
        for i, Impulse_FEXT in enumerate(Impulse_FEXTs):
            plt.plot(time1*1e12, Impulse_FEXT, label=f'FEXT to Pin {i+1}')
        
        plt.title(f'FEXT Impulse Responses from Pin {pin_number}')
        plt.xlabel('Time (ps)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plot_FEXTimp_response.png')



def via_array_generation(size, num_samples, num_sig, diff=None):
    """
    Generate random via arrays with only 1s and 3s, where number of 3s equals num_sig.
    Generated arrays are unique under rotation and reflection.
    Specific positions can be preset using the diff parameter.
    
    Parameters:
    size (list): Size of the array [rows, cols]. Ex: [3,2] for 3x2 matrix
    num_samples (int): Required number of unique arrays to generate
    num_sig (int): Desired number of signal vias (3s) in each array
    diff (list, optional): List of position pairs where each sublist contains two position lists
          Ex: [[[2,2], [2,3]], [[4,4], [4,3]], [[0,0], [1,0]]] means
          positions (2,2), (4,4), (0,0) have value 4,
          positions (2,3), (4,3), (1,0) have value 5
    
    Returns:
    list: List of exactly num_samples unique via arrays
    """
    output_arrays = []
    total_elements = size[0] * size[1]
    
    # Check if num_sig is valid
    if num_sig > total_elements:
        raise ValueError(f"num_sig ({num_sig}) cannot be larger than total elements ({total_elements})")
    
    import math
    max_possible = math.comb(total_elements, num_sig)
    if num_samples > max_possible:
        raise ValueError(f"Requested num_samples ({num_samples}) exceeds maximum possible unique patterns ({max_possible})")
    
    attempts = 0
    while len(output_arrays) < num_samples:
        # Create initial array with all 1s
        array = [[1 for _ in range(size[1])] for _ in range(size[0])]
        
        # Apply diff constraints if provided
        preset_positions = set()  # 이미 값이 설정된 위치들을 추적
        if diff is not None:
            # 모든 4의 위치 처리
            for pos_pair in diff:
                row, col = pos_pair[0]  # 첫 번째 리스트의 위치
                array[row][col] = 4
                preset_positions.add((row, col))
            
            # 모든 5의 위치 처리
            for pos_pair in diff:
                row, col = pos_pair[1]  # 두 번째 리스트의 위치
                array[row][col] = 5
                preset_positions.add((row, col))
        
        # Count remaining positions needed for signal vias (3s)
        remaining_sig = num_sig - sum(row.count(3) for row in array)
        if remaining_sig < 0:
            continue
        
        # Get available positions (not preset)
        available_positions = [(i, j) for i in range(size[0]) for j in range(size[1])
                             if (i, j) not in preset_positions]
        
        if len(available_positions) < remaining_sig:
            continue
            
        # Randomly select positions for remaining signal vias
        selected_positions = np.random.choice(len(available_positions), 
                                           remaining_sig, 
                                           replace=False)
        
        # Place 3s in selected positions
        for pos_idx in selected_positions:
            i, j = available_positions[pos_idx]
            array[i][j] = 3
        
        # Check if this array or its rotations/reflections already exist
        is_unique = not any(via_util_is_same_or_rotate(array, existing_array) 
                          for existing_array in output_arrays)
        
        if is_unique:
            output_arrays.append(array)
        
        attempts += 1
        if attempts % 1000 == 0:
            print(f"Progress: {len(output_arrays)}/{num_samples} arrays found after {attempts} attempts")
    
    print(f"Successfully generated {num_samples} unique arrays after {attempts} attempts")
    return output_arrays





if __name__ == "__main__":
    via_array = [
        [1, 1, 0, 1, 3, 0, 1],
        [3, 1, 3, 3, 3, 1, 0],
        [2, 0, 3, 1, 2, 0, 1],
        [2, 0, 0, 1, 3, 1, 0],
        [2, 0, 3, 3, 3, 0, 1],
        [1, 1, 0, 1, 3, 0, 1],
        [3, 1, 3, 3, 3, 1, 0]
    ]
    # via_array = [
    #     [1, 1, 1, 1, 3, 3, 1, 1,],
    #     [1, 1, 1, 1, 3, 3, 1, 1,],
    #     [1, 1, 1, 1, 3, 3, 1, 1,],
    #     [1, 1, 1, 1, 3, 3, 1, 1,],
    #     [1, 1, 1, 1, 3, 3, 1, 1,],
    #     [1, 1, 1, 1, 3, 3, 1, 1,],
    #     [1, 1, 1, 1, 3, 3, 1, 1,],
    #     [1, 1, 1, 1, 3, 3, 1, 1,],
    # ]
    print("h")
    via_config_visual_all(via_array, imag=False, size=10)
    quit()
