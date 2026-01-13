import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math

def save_plots_to_pdf(csv_file, output_pdf='progress_plots.pdf'):
    # Load the data
    df = pd.read_csv(csv_file)

    # Define x-axis
    x_col = 'time/total_timesteps'
    if x_col not in df.columns:
        x_data = df.index
        x_label = 'Index'
    else:
        x_data = df[x_col]
        x_label = x_col

    # Identify y-columns (all columns except x_col)
    y_cols = [col for col in df.columns if col != x_col]

    # Settings for the PDF layout
    plots_per_page = 6
    rows_per_page = 3
    cols_per_page = 2
    num_pages = math.ceil(len(y_cols) / plots_per_page)

    # Create the PDF
    with PdfPages(output_pdf) as pdf:
        for page in range(num_pages):
            # Create a figure for the current page (Letter size: 8.5x11 inches)
            fig, axes = plt.subplots(rows_per_page, cols_per_page, figsize=(8.5, 11))
            axes = axes.flatten()
            
            # Determine which columns go on this page
            start_idx = page * plots_per_page
            end_idx = min((page + 1) * plots_per_page, len(y_cols))
            current_cols = y_cols[start_idx:end_idx]
            
            # Plot each column for this page
            for i, col in enumerate(current_cols):
                ax = axes[i]
                ax.plot(x_data, df[col], linewidth=1.5)
                ax.set_title(col, fontsize=10, fontweight='bold')
                ax.set_xlabel(x_label, fontsize=8)
                ax.grid(True, linestyle='--', alpha=0.6)
                
            # Hide unused subplots on the last page
            for j in range(len(current_cols), len(axes)):
                axes[j].axis('off')
                
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for page margins
            fig.suptitle(f'Page {page + 1} of {num_pages}', fontsize=12)
            
            # Save the page
            pdf.savefig(fig)
            plt.close(fig)

    print(f"All plots saved to {output_pdf}")

if __name__ == "__main__":
    #save_plots_to_pdf('rl_train/results/SAC/train_session_20260108-110504/progress.csv')
    save_plots_to_pdf('rl_train/results/SAC/train_session_20260108-233349/progress.csv')