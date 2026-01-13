import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_losses(file_path):
    """
    Plots training losses from a JSON log file.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # Ensure the key 'log_datas' exists
    if 'log_datas' not in data:
        print("Error: JSON structure does not match expected format (missing 'log_datas').")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data['log_datas'])
    
    # List of losses to plot
    losses_to_plot = ['entropy_loss', 'loss', 'policy_gradient_loss', 'value_loss']
    
    plt.figure(figsize=(12, 8))
    has_valid_data = False

    for loss_name in losses_to_plot:
        if loss_name in df.columns:
            # Filter out placeholder values (-1)
            valid_data = df[df[loss_name] != -1]
            
            if not valid_data.empty:
                plt.plot(valid_data['num_timesteps'], valid_data[loss_name], label=loss_name)
                has_valid_data = True
            else:
                print(f"Warning: No valid data found for '{loss_name}' (all values are -1).")
        else:
            print(f"Warning: Column '{loss_name}' not found in data.")

    if has_valid_data:
        plt.xlabel('Num Timesteps')
        plt.ylabel('Loss Value')
        plt.title('Training Losses Over Time')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()
    else:
        print("\nNo valid loss data available to plot. The log file may contain only placeholder values (-1).")

if __name__ == "__main__":
    # Update this path if your file is named differently
    log_file = 'rl_train/results/PPO/train_session_20260108-110504/train_log.json'
    #log_file = 'rl_train/results/SAC/train_session_20260108-233349/train_log.json'
    plot_losses(log_file)