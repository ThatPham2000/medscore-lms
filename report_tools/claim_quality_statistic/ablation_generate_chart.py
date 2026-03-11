import os

import matplotlib.pyplot as plt
import pandas as pd


def generate_ablation_claim_quality_chart():
    """
\begin{table}[t]
\centering
\small
\begin{tabular}{lcccccccc}
\hline
\textbf{Method} & \textbf{Total} & \textbf{Valid} & \textbf{Unverifiable} & \textbf{Hallucinated} & \textbf{Incomplete} & \textbf{Incorrectly-structured} & \textbf{Context-dependent} & \textbf{Redundant} \\
\hline
MedScore-LMS$*$ (ministral-3:14b) & 1204 & 900 (74.75\%) & 1 (0.08\%) & 100 (8.31\%) & 104 (8.64\%) & 2 (0.17\%) & 77 (6.40\%) & 20 (1.66\%) \\
MedScore-LMS$**$ (ministral-3:14b) & 1204 & 874 (72.59\%) & 0 (0\%) & 128 (10.63\%) & 46 (3.82\%) & 7 (0.58\%) & 28 (2.33\%) & 121 (10.05\%) \\
MedScore-LMS$*$ (phi4:14b) & 1288 & 893 (69.33\%) & 5 (0.39\%) & 48 (3.73\%) & 94 (7.30\%) & 78 (6.06\%) & 153 (11.88\%) & 17 (1.32\%) \\
MedScore-LMS$**$ (phi4:14b) & 1288 & 892 (69.25\%) & 4 (0.31\%) & 50 (3.88\%) & 50 (3.88\%) & 40 (3.11\%) & 87 (6.75\%) & 165 (12.81\%) \\
MedScore-LMS$*$ (deepseek-r1:14b) & 1027 & 738 (71.86\%) & 1 (0.1\%) & 23 (2.24\%) & 64 (6.23\%) & 48 (4.67\%) & 145 (14.12\%) & 8 (0.78\%) \\
MedScore-LMS$**$ (deepseek-r1:14b) & 1027 & 763 (74.29\%) & 4 (0.39\%) & 37 (3.60\%) & 49 (4.77\%) & 36 (3.51\%) & 86 (8.37\%) & 52 (5.06\%) \\
\hline
\end{tabular}
\caption{Automatic taxonomy profiling across decomposition methods for 100 AskDocsAI samples. For ($*$), we report outputs of \textbf{\textit{Decomposition + Classification}}. For ($**$), we report outputs of \textbf{\textit{Decomposition+ Classification + Normalization + Reclassification}}}
\label{tab:ablation_claim_quality_stats}
\end{table}
    """
    # 1. Data from Table
    methods = [
        'MedScore-LMS× (ministral-3:14b)',
        'MedScore-LMS✓ (ministral-3:14b)',
        'MedScore-LMS× (phi4:14b)',
        'MedScore-LMS✓ (phi4:14b)',
        'MedScore-LMS× (deepseek-r1:14b)',
        'MedScore-LMS✓ (deepseek-r1:14b)'
    ]

    # Total column
    totals = [1204, 1204, 1288, 1288, 1027, 1027]

    # Claim quality columns
    counts_data = {
        'Valid':                  [900, 874, 893, 892, 738, 763],
        'Unverifiable':           [1,   0,   5,   4,   1,   4],
        'Hallucinated':           [100, 128, 48,  50,  23,  37],
        'Incomplete':             [104, 46,  94,  50,  64,  49],
        'Incorrectly-structured': [2,   7,   78,  40,  48,  36],
        'Context-dependent':      [77,  28,  153, 87,  145, 86],
        'Redundant':              [20,  121, 17,  165, 8,   52]
    }

    df_counts = pd.DataFrame(counts_data, index=methods)
    # Calculate percentages based on the actual Total numbers from the table
    df_pct = df_counts.div(totals, axis=0) * 100

    # 2. Set claim quality label colors
    colors = ['#27ae60', '#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#d35400', '#7f8c8d']

    # fig, ax = plt.subplots(figsize=(16, 10))
    fig, ax = plt.subplots(figsize=(14, 7)) # Adjusted for better aspect ratio
    df_pct.plot(kind='barh', stacked=True, color=colors, ax=ax, width=0.8, edgecolor='white', linewidth=0.5)

    # 3. Adding Percentage Labels
    for i, col in enumerate(df_counts.columns):
        for j, method in enumerate(methods):
            patch = ax.patches[i * len(methods) + j]
            width = patch.get_width()

            if width > 0.5:  # Display only if percentage > 0.5%
                x_pos = patch.get_x() + width / 2
                y_pos = patch.get_y() + patch.get_height() / 2

                # Decide font size and rotation
                if width > 6.0:
                    label = f"{width:.2f}%"
                    fs, rot = 9, 0
                else:
                    label = f"{width:.2f}%"
                    fs, rot = 8, 90  # Rotate vertically if too narrow

                ax.text(x_pos, y_pos, label, ha='center', va='center',
                        fontsize=fs, color='white', fontweight='bold', rotation=rot)

    # 4. Adding Total Sample Count Labels
    for i, total in enumerate(totals):
        ax.text(101, i, f"Total: {total}", va='center', fontsize=11,
                fontweight='bold', color='#333333')

    # 5. Finalizing layout
    ax.set_title('Ablation Study: Taxonomy Profiling with Normalization Stage', fontsize=18, fontweight='bold', pad=30)
    ax.set_xlabel('Percentage of Total Claims (%)', fontsize=12, labelpad=10)
    ax.set_xlim(0, 115)
    ax.legend(title='Claim Quality Labels', bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4, frameon=False,
              fontsize=11)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()

    # save file
    file_name = 'ablation_claim_quality_statistic'
    plt.savefig(f'{file_name}.pdf', bbox_inches='tight')
    plt.savefig(f'{file_name}.png', dpi=300, bbox_inches='tight')
    print(f"Files were created: {os.getcwd()}")


if __name__ == "__main__":
    generate_ablation_claim_quality_chart()
