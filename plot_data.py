import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# ==========================================
# CONFIGURATION & MAPPING
# ==========================================

ALGORITHM = 'PPO'  # 'PPO' or 'SAC'
LAYOUT = 'horizontal'  # 'vertical' or 'horizontal'

# --- TIME CONFIGURATION ---
if ALGORITHM == 'PPO':
    DT = 0.016                 # Time per step in seconds (e.g., 0.032s = 30Hz, 0.016s = 60Hz)
else:
    DT = 0.032                # Time per step in seconds (e.g., 0.032s = 30Hz, 0.016s = 60Hz)
MAX_PLOT_DURATION = 10.0  # Limit x-axis to first X seconds (set to None to show all)

JOINTS = {
    'hip':   {'json': 'hip_flexion_r',   'npz': 'q_hip_flexion_r',   'label': 'Hip Flexion'},
    'knee':  {'json': 'knee_angle_r',    'npz': 'q_knee_angle_r',    'label': 'Knee Angle'},
    'ankle': {'json': 'ankle_angle_r',   'npz': 'q_ankle_angle_r',   'label': 'Ankle Angle'},
    
    # Left leg for comparison
    'hip_l':   {'json': 'hip_flexion_l', 'label': 'Hip Flexion (L)'},
    'knee_l':  {'json': 'knee_angle_l',  'label': 'Knee Angle (L)'},
    'ankle_l': {'json': 'ankle_angle_l', 'label': 'Ankle Angle (L)'}
}

if ALGORITHM == 'PPO':
    FILES = {
        'eval': 'rl_train/results/PPO/train_session_20260108-110504/analyze_results_58261504_00/gait_evaluated_data_00.json',
        'ref': 'rl_train/reference_data/short_reference_gait.npz'
    }
    OUTPUTFOLDER = 'ppo_gait_analysis_outputs'
else:  # SAC
    FILES = {
        'eval': 'rl_train/results/SAC_05/train_session_20260108-233349/analyze_results_4710912_00/gait_evaluated_data_00.json',
        'ref': 'rl_train/reference_data/short_reference_gait.npz'
    }
    OUTPUTFOLDER = 'sac_gait_analysis_outputs'

DARK_BLUE = (33/255, 82/255, 88/255)
LIGHT_CYAN = (80/255,170/255,171/255)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

output_dir = os.path.join(os.getcwd(), OUTPUTFOLDER)
os.makedirs(output_dir, exist_ok=True)

print(f"Plots and stats will be saved to: {output_dir}")

def load_data():
    """Loads the JSON and NPZ files and converts RAD -> DEG."""
    print(f"Loading {FILES['eval']}...")
    with open(FILES['eval'], 'r') as f:
        eval_data = json.load(f)
        
    print(f"Loading {FILES['ref']}...")
    ref_data = np.load(FILES['ref'], allow_pickle=True)
    
    extracted_eval = {}
    joint_data_root = eval_data['series_data']['joint_data']
    
    for key, mapping in JOINTS.items():
        j_name = mapping['json']
        if j_name in joint_data_root:
            raw_rad = np.array(joint_data_root[j_name]['qpos']).flatten()
            extracted_eval[key] = np.degrees(raw_rad) # Convert to Deg
        else:
            print(f"Warning: Joint '{j_name}' not found in JSON data.")

    extracted_ref = {}
    for key, mapping in JOINTS.items():
        if 'npz' in mapping:
            n_name = mapping['npz']
            raw_ref = None
            if n_name in ref_data:
                raw_ref = ref_data[n_name]
            elif 'series_data' in ref_data and n_name in ref_data['series_data'].item():
                 raw_ref = ref_data['series_data'].item()[n_name]
            else:
                 clean_name = n_name.replace('q_', '') 
                 if clean_name in ref_data:
                     raw_ref = ref_data[clean_name]
            
            if raw_ref is not None:
                extracted_ref[key] = np.degrees(raw_ref) # Convert to Deg

    return extracted_eval, extracted_ref

def segment_gait_cycles(signal, segmentation_signal=None, n_points=100):
    """Segments data into cycles (0-100%) using peaks."""
    if segmentation_signal is None:
        segmentation_signal = signal

    peaks, _ = find_peaks(segmentation_signal, distance=int(1.0/DT), prominence=5) 
    
    cycles = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i+1]
        cycle_raw = signal[start:end]
        
        x_old = np.linspace(0, 1, len(cycle_raw))
        x_new = np.linspace(0, 1, n_points)
        f = interp1d(x_old, cycle_raw, kind='linear')
        cycles.append(f(x_new))
        
    return np.array(cycles)

def get_stats(cycles):
    if len(cycles) == 0: return None, None
    return np.mean(cycles, axis=0), np.std(cycles, axis=0)

def save_statistics(eval_data):
    """Calculates scalar stats and saves to a .txt file."""
    def fmt_line(name, data):
        if data is None or len(data) == 0:
            return f"  {name:<6} - No Data found"
        mu = np.mean(data)
        std = np.std(data)
        mn = np.min(data)
        mx = np.max(data)
        return (f"  {name:<6} - Mean: {mu:>6.1f}deg | Std: {std:>5.1f}deg | "
                f"Min: {mn:>6.1f}deg | Max: {mx:>6.1f}deg")

    lines = []
    lines.append("Right Leg Statistics:")
    lines.append(fmt_line("Hip", eval_data.get('hip')))
    lines.append(fmt_line("Knee", eval_data.get('knee')))
    lines.append(fmt_line("Ankle", eval_data.get('ankle')))
    lines.append("") 
    lines.append("Left Leg Statistics:")
    lines.append(fmt_line("Hip", eval_data.get('hip_l')))
    lines.append(fmt_line("Knee", eval_data.get('knee_l')))
    lines.append(fmt_line("Ankle", eval_data.get('ankle_l')))
    
    filepath = os.path.join(OUTPUTFOLDER, 'gait_statistics.txt')
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))
    print("-" * 30)
    print("\n".join(lines))
    print("-" * 30)

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def plot_kinematics(eval_data):
    """Plots raw values over time (s) with optional zoom limits."""
    # Modified to include Left and Right legs
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Pairs of (Right, Left) keys to plot together
    joint_pairs = [('hip', 'hip_l'), ('knee', 'knee_l'), ('ankle', 'ankle_l')]
    
    # Generate Time Axis
    if not eval_data:
        return
    any_key = next(iter(eval_data))
    n_frames = len(eval_data[any_key])
    time_axis = np.arange(n_frames) * DT
    
    for ax, (r_key, l_key) in zip(axes, joint_pairs):
        # Plot Right Leg
        if r_key in eval_data:
            ax.plot(time_axis, eval_data[r_key], label='Right', color=DARK_BLUE, linewidth=1)
            
        # Plot Left Leg
        if l_key in eval_data:
            ax.plot(time_axis, eval_data[l_key], label='Left', color=LIGHT_CYAN, linewidth=1)
            
        ax.set_ylabel(f"{JOINTS[r_key]['label']} (deg)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Apply Limit if configured
        if MAX_PLOT_DURATION is not None:
            ax.set_xlim(0, MAX_PLOT_DURATION)
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Kinematics (Raw Data over Time), {ALGORITHM}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTFOLDER}/kinematics.png')
    # plt.show()

def plot_segmented_avg(eval_data):
    """Plots average gait cycle (deg)."""
    ref_signal = eval_data.get('hip', None)
    if ref_signal is None: return
    if LAYOUT == 'vertical':
        fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    else:  # horizontal
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    joints = ['hip', 'knee', 'ankle']

    for ax, j_key in zip(axes, joints):
        if j_key in eval_data:
            cycles = segment_gait_cycles(eval_data[j_key], ref_signal)
            if len(cycles) > 0:
                mean, std = get_stats(cycles)
                x = np.linspace(0, 100, len(mean))
                ax.plot(x, mean, color=DARK_BLUE, label='Mean')
                ax.fill_between(x, mean-std, mean+std, color=DARK_BLUE, alpha=0.2)
            
            ax.set_title(JOINTS[j_key]['label'])
            ax.set_xlabel('% Gait Cycle')
            ax.set_ylabel('Angle (deg)')
            ax.grid(True, alpha=0.3)
            if j_key == 'hip': ax.legend()

    fig.suptitle(f'Segmented Joint Data (Average), {ALGORITHM}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTFOLDER}/segmented_joint_data_avg_{LAYOUT}.png')

def plot_right_ref_comparison(eval_data, ref_data):
    """Plots Evaluated vs Reference (deg)."""
    seg_signal_eval = eval_data.get('hip', None)
    ref_is_cycle = False
    if 'hip' in ref_data:
        ref_is_cycle = len(ref_data['hip']) < 200

    if LAYOUT == 'vertical':
        fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    else:  # horizontal
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    joints = ['hip', 'knee', 'ankle']

    for ax, j_key in zip(axes, joints):
        if j_key in eval_data and seg_signal_eval is not None:
            cycles = segment_gait_cycles(eval_data[j_key], seg_signal_eval)
            if len(cycles) > 0:
                mean_eval, std_eval = get_stats(cycles)
                x = np.linspace(0, 100, len(mean_eval))
                ax.plot(x, mean_eval, color=DARK_BLUE, label='Evaluated')
                ax.fill_between(x, mean_eval-std_eval, mean_eval+std_eval, color=DARK_BLUE, alpha=0.1)

        if j_key in ref_data:
            x_plot = np.linspace(0, 100, 100)
            if ref_is_cycle:
                y_ref = ref_data[j_key]
                x_ref_raw = np.linspace(0, 100, len(y_ref))
                f_ref = interp1d(x_ref_raw, y_ref)
                ax.plot(x_plot, f_ref(x_plot), color=LIGHT_CYAN, label='Reference')
            else:
                if 'hip' in ref_data:
                    cycles_ref = segment_gait_cycles(ref_data[j_key], ref_data['hip'])
                    if len(cycles_ref) > 0:
                        mean_ref, _ = get_stats(cycles_ref)
                        ax.plot(x_plot, mean_ref, color=LIGHT_CYAN, label='Reference')

        ax.set_title(JOINTS[j_key]['label'])
        ax.set_xlabel('% Gait Cycle')
        ax.set_ylabel('Angle (deg)')
        ax.grid(True, alpha=0.3)
        if j_key == 'hip': ax.legend()

    fig.suptitle(f'Right Leg: Evaluated vs Reference, {ALGORITHM}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTFOLDER}/right_ref_comparison_avg_{LAYOUT}.png')

def plot_left_right_comparison(eval_data):
    """Plots Left vs Right leg symmetry (deg)."""
    seg_signal_r = eval_data.get('hip', None)
    seg_signal_l = eval_data.get('hip_l', None)

    if seg_signal_r is None or seg_signal_l is None:
        return

    if LAYOUT == 'vertical':
        fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    else:  # horizontal
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [('hip', 'hip_l'), ('knee', 'knee_l'), ('ankle', 'ankle_l')]

    for ax, (r_key, l_key) in zip(axes, pairs):
        if r_key in eval_data and l_key in eval_data:
            cycles_r = segment_gait_cycles(eval_data[r_key], seg_signal_r)
            if len(cycles_r) > 0:
                mean_r, std_r = get_stats(cycles_r)
                x = np.linspace(0, 100, len(mean_r))
                ax.plot(x, mean_r, color=DARK_BLUE, label='Right')
                ax.fill_between(x, mean_r-std_r, mean_r+std_r, color=DARK_BLUE, alpha=0.1)
            
            cycles_l = segment_gait_cycles(eval_data[l_key], seg_signal_l)
            if len(cycles_l) > 0:
                mean_l, std_l = get_stats(cycles_l)
                x = np.linspace(0, 100, len(mean_l))
                ax.plot(x, mean_l, color=LIGHT_CYAN, label='Left')
                ax.fill_between(x, mean_l-std_l, mean_l+std_l, color=LIGHT_CYAN, alpha=0.1)
            
            ax.set_title(JOINTS[r_key]['label'].split(' ')[0])
            ax.set_xlabel('% Gait Cycle')
            ax.set_ylabel('Angle (deg)')
            ax.grid(True, alpha=0.3)
            if r_key == 'hip': ax.legend()

    fig.suptitle(f'Symmetry: Left vs Right Leg, {ALGORITHM}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTFOLDER}/left_right_comparison_avg_{LAYOUT}.png')

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        e_data, r_data = load_data()
        
        if not e_data:
            print("Error: No evaluated data found. Check JSON keys.")
        else:
            print("Calculating Statistics...")
            save_statistics(e_data)

            print("Generating Kinematics Plot...")
            plot_kinematics(e_data)
            
            print("Generating Segmented Average Plot...")
            plot_segmented_avg(e_data)
            
            print("Generating Reference Comparison Plot...")
            plot_right_ref_comparison(e_data, r_data)
            
            print("Generating Left/Right Comparison Plot...")
            plot_left_right_comparison(e_data)
            
            print("Done! All plots and stats saved.")
            
    except FileNotFoundError:
        print("Error: Ensure data files are in the correct paths.")
    except Exception as e:
        print(f"An error occurred: {e}")