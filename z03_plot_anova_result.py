import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams['axes.labelsize'] = 8  # Font size for x and y axis labels
plt.rcParams['xtick.labelsize'] = 7  # Set x-axis tick label size
plt.rcParams['ytick.labelsize'] = 7  # Set y-axis tick label size
plt.rcParams['axes.titlesize'] = 9  # Font size for plot title
df = pd.read_excel("/mnt/d/project/hydraulic/test/plot/fig_final/fig4/anova_for_plot.xlsx")
xname_labels = {
"wb_soilmean": fr'$\theta_{{crit}}$',
"wb_rwc_soilmean": fr'$REW_{{crit}}^{{\theta}}$',
"rwc_soilmean_recal":fr'$REW_{{crit}}$',
"wb_psi_soilmean": fr'$\psi_{{crit}}^{{\theta}}$',
"psi_soilmean": fr'$\psi_{{crit}}$',
}  
x_labels = list(xname_labels.keys())
methods = df['method'].unique()
bar_width = 0.3
x = np.arange(len(x_labels))
fig, ax = plt.subplots(figsize=(6, 3.5))  # width × height in inches


colors = {'soil':'sandybrown', 'climate':'skyblue'}
colors = {
    'NTD_soil': 'sandybrown',
    'NTD_climate': 'skyblue',
    'EF_soil': 'peru',
    'EF_climate': 'dodgerblue'
}
group_spacing = 0.1
for i, m in enumerate(methods):
    bottom = np.zeros(len(x_labels))
    df_m = df[df['method']==m].set_index('xname').reindex(x_labels)
    x_pos = x + i*(bar_width + group_spacing)
    for factor in ['soil', 'climate']:
        heights = df_m[factor].values
        label = f"{factor} ({m})"
       
        ax.bar(x_pos, heights, width=bar_width, bottom=bottom,
            color=colors[f"{m}_{factor}"],label=label)
 
        # Annotate the middle of each segment
        for xi, h, b in zip(x_pos, heights, bottom):
            if not np.isnan(h) and h > 0:
                ax.text(xi, b + h/2, f"{h:.0f}", ha='center', va='center', fontsize=6, color='black')
        
        bottom += heights

  
# Format x-axis
ax.set_xticks(x + (bar_width + group_spacing)/2)


ax.set_xticklabels([xname_labels[x] for x in x_labels])

ax.set_ylabel('Explained Percentage (%)')

# Legend
ax.legend(frameon=False,fontsize=5.5)
plt.tight_layout()

fig.savefig("stacked_grouped_bar.png", dpi=300, bbox_inches='tight')
plt.show()