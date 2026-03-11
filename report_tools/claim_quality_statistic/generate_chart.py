import matplotlib.pyplot as plt
import pandas as pd
import os


def generate_claim_quality_chart():
    """
\begin{table}[t]
\centering
\small
\begin{tabular}{lcccc}
\hline
\textbf{Method} & \textbf{Total} & \textbf{Valid} & \textbf{Unverifiable} & \textbf{Hallucinated} & \textbf{Incomplete} & \textbf{Incorrectly-structured} & \textbf{Context-dependent} & \textbf{Redundant} \\
\hline
FActScore (ministral-3:14b) & 4978 & 1463 (29.39\%) & 273 (5.48\%) & 311 (6.25\%) & 728 (14.62\%) & 349 (7.01\%) & 1398 (28.08\%) & 452 (9.08\%) \\
FActScore (phi4:14b) & 3925 & 1258 (32.05\%) & 274 (6.98\%) & 170 (4.33\%) & 600 (15.29\%) & 378 (9.63\%) & 1126 (28.69\%) & 117 (2.98\%) \\
FActScore (deepseek-r1:14b) & 2848 & 731 (25.67\%) & 197 (6.92\%) & 113 (3.97\%) & 496 (17.42\%) & 223 (7.83\%) & 1055 (37.04\%) & 31 (1.09\%) \\
MedScore (ministral-3:14b) & 1516 & 950 (62.66\%) & 0 (0\%) & 403 (26.58\%) & 98 (6.46\%) & 0 (0\%) & 17 (1.12\%) & 48 (3.17\%) \\
MedScore (phi4:14b) & 2532 & 1291 (50.99\%) & 171 (6.75\%) & 230 (9.08\%) & 365 (14.42\%) & 213 (8.41\%) & 204 (8.06\%) & 58 (2.29\%) \\
MedScore (deepseek-r1:14b) & 860 & 160 (18.60\%) & 82 (9.53\%) & 24 (2.79\%) & 467 (54.30\%) & 56 (6.51\%) & 64 (7.44\%) & 7 (0.81\%) \\
MedScore (GPT-4o-mini) & 1209 & 930 (76.92\%) & 0 (0\%) & 124 (10.26\%) & 73 (6.04\%) & 4 (0.33\%) & 72 (5.96\%) & 6 (0.50\%) \\
MedScore-LMS (ministral-3:14b) & 1204 & 874 (72.59\%) & 0 (0\%) & 128 (10.63\%) & 46 (3.82\%) & 7 (0.58\%) & 28 (2.33\%) & 121 (10.05\%) \\
MedScore-LMS (phi4:14b) & 1288 & 892 (69.25\%) & 4 (0.31\%) & 50 (3.88\%) & 50 (3.88\%) & 40 (3.11\%) & 87 (6.75\%) & 165 (12.81\%) \\
MedScore-LMS (deepseek-r1:14b) & 1027 & 763 (74.29\%) & 4 (0.39\%) & 37 (3.60\%) & 49 (4.77\%) & 36 (3.51\%) & 86 (8.37\%) & 52 (5.06\%) \\
\hline
\end{tabular}
\caption{Automatic taxonomy profiling across decomposition methods for 100 AskDocsAI samples}
\label{tab:taxonomy_profile}
\end{table}
    """
    # 1. Data from Table 2
    methods = [
        'FActScore (ministral-3:14b)', 'FActScore (phi4:14b)', 'FActScore (deepseek-r1:14b)',
        'MedScore (ministral-3:14b)', 'MedScore (phi4:14b)', 'MedScore (deepseek-r1:14b)',
        'MedScore (GPT-4o-mini)',
        'MedScore-LMS (ministral-3:14b)', 'MedScore-LMS (phi4:14b)', 'MedScore-LMS (deepseek-r1:14b)'
    ]

    # Total column
    totals = [4978, 3925, 2848, 1516, 2532, 860, 1209, 1204, 1288, 1027]

    # Claim quality columns
    counts_data = {
        'Valid':                  [1463, 1258, 731,  950, 1291, 160, 930, 874, 892, 763],
        'Unverifiable':           [273,  274,  197,  0,   171,  82,  0,   0,   4,   4],
        'Hallucinated':           [311,  170,  113,  403, 230,  24,  124, 128, 50,  37],
        'Incomplete':             [728,  600,  496,  98,  365,  467, 73,  46,  50,  49],
        'Incorrectly-structured': [349,  378,  223,  0,   213,  56,  4,   7,   40,  36],
        'Context-dependent':      [1398, 1126, 1055, 17,  204,  64,  72,  28,  87,  86],
        'Redundant':              [452,  117,  31,   48,  58,   7,   6,   121, 165, 52]
    }

    df_counts = pd.DataFrame(counts_data, index=methods)
    # Calculate percentages based on the actual Total numbers from the table
    df_pct = df_counts.div(totals, axis=0) * 100

    # 2. Set claim quality label colors
    colors = ['#27ae60', '#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#d35400', '#7f8c8d']

    # fig, ax = plt.subplots(figsize=(16, 10))
    fig, ax = plt.subplots(figsize=(14, 8))
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
    ax.set_title('Automatic taxonomy profiling distribution', fontsize=18, fontweight='bold', pad=30)
    ax.set_xlabel('Percentage of Total Claims (%)', fontsize=12, labelpad=10)
    ax.set_xlim(0, 115)  # Leave space to the right for the word "Total"
    ax.legend(title='Claim quality Labels', bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4, frameon=False,
              fontsize=11)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('claim_quality_statistic.pdf', bbox_inches='tight')
    plt.savefig('claim_quality_statistic.png', dpi=300, bbox_inches='tight')
    print(f"Files were created: {os.getcwd()}")


if __name__ == "__main__":
    generate_claim_quality_chart()