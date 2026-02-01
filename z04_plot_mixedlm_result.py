import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams['axes.labelsize'] = 8  # Font size for x and y axis labels
plt.rcParams['xtick.labelsize'] = 7  # Set x-axis tick label size
plt.rcParams['ytick.labelsize'] = 7  # Set y-axis tick label size
plt.rcParams['axes.titlesize'] = 9  # Font size for plot title
df = pd.read_excel("/mnt/d/project/hydraulic/test/plot/f05_test_LHS/anova_results_smallDaily_mix.xlsx",sheet_name='forplot')
xname_labels = {
"wb_soilmean": fr'$\theta_{{crit}}$',
"wb_rwc_soilmean": fr'$REW_{{crit}}^{{\theta}}$',
"rwc_soilmean_recal":fr'$REW_{{crit}}$',
"wb_psi_soilmean": fr'$\psi_{{crit}}^{{\theta}}$',
"psi_soilmean": fr'$\psi_{{crit}}$',
}  

x_labels = list(xname_labels.keys())
methods = df['method'].unique()
models = df['model'].unique()
bar_width = 0.14
x = np.arange(len(x_labels))
fig, ax = plt.subplots(figsize=(6, 3.5))  # width × height in inches


colors = {'soil':'sandybrown', 'climate':'skyblue'}
colors = {
    'NTD_soil': 'sandybrown',
    'NTD_climate': 'skyblue',
    'EF_soil': 'peru',
    'EF_climate': 'dodgerblue',
    'NTD_hydraulic': 'lightgreen',  # changed color
    'EF_hydraulic': 'limegreen'     # changed color
}
group_spacing = 0.04
for j, model in enumerate(models):
    for i, m in enumerate(methods):
        bottom = np.zeros(len(x_labels))
        df_m = df[(df['model']==model) & (df['method']==m)].set_index('xname').reindex(x_labels)
        x_pos = x + j*(len(methods)*(bar_width + group_spacing)) + i*(bar_width + group_spacing)
        if j == 1:  # model1
            edgecolor = 'black'
            lw = 1.0
        else:       # model2
            edgecolor = None
            lw = 0
        show_label = (j == 0)
        for factor in ['soil', 'climate', 'hydraulic']:
            heights = df_m[factor].values
            label = f"{factor} ({m})"
            if show_label:
                ax.bar(x_pos, heights, width=bar_width, bottom=bottom,
                    color=colors[f"{m}_{factor}"],edgecolor=edgecolor, linewidth=lw,label=label)
            else:
                ax.bar(x_pos, heights, width=bar_width, bottom=bottom,
                    color=colors[f"{m}_{factor}"],edgecolor=edgecolor, linewidth=lw)
            # Annotate the middle of each segment
            for xi, h, b in zip(x_pos, heights, bottom):
                if not np.isnan(h) and h > 0:
                    ax.text(xi, b + h/2, f"{h:.0f}", ha='center', va='center', fontsize=6, color='black')
            
            bottom += heights

        # if show_label:
        #     # Stack soil and climate
        #     ax.bar(x_pos, df_m['soil'], width=bar_width, bottom=bottom, edgecolor=edgecolor, linewidth=lw,color=colors[f'{m}_soil'], label=f"Soil type ({m})")

        #     bottom += df_m['soil'].values
        #     ax.bar(x_pos, df_m['climate'], width=bar_width, bottom=bottom, edgecolor=edgecolor, linewidth=lw, color=colors[f'{m}_climate'], label=f"climate ({m})")
        #     bottom += df_m['climate'].values
        #     ax.bar(x_pos, df_m['hydraulic'], width=bar_width, bottom=bottom,edgecolor=edgecolor, linewidth=lw, color=colors[f'{m}_hydraulic'], label=f"climate ({m})")
        # else:
        #     ax.bar(x_pos, df_m['soil'], width=bar_width, bottom=bottom, edgecolor=edgecolor, linewidth=lw,color=colors[f'{m}_soil'])

        #     bottom += df_m['soil'].values
        #     ax.bar(x_pos, df_m['climate'], width=bar_width, bottom=bottom, edgecolor=edgecolor, linewidth=lw, color=colors[f'{m}_climate'])
        #     bottom += df_m['climate'].values
        #     ax.bar(x_pos, df_m['hydraulic'], width=bar_width, bottom=bottom,edgecolor=edgecolor, linewidth=lw, color=colors[f'{m}_hydraulic'])
# Add representative empty bars for models
x_dummy =np.nan  # off the plot
ax.bar(x_dummy, np.nan, facecolor='lightgrey', edgecolor=None, linewidth=0, label=models[0])
ax.bar(x_dummy,np.nan, facecolor='lightgrey', edgecolor='black', linewidth=1, label=models[1])

        
# Format x-axis
ax.set_xticks(x + (len(models)*len(methods)-1)*(bar_width + group_spacing)/2)

ax.set_xticklabels([xname_labels[x] for x in x_labels])

ax.set_ylabel('Explained Percentage (%)')

# Legend
ll = ax.legend(frameon=False, ncol=3,
        # labelspacing=self.llrspace,
        # handletextpad=self.llhtextpad,
        # handlelength=self.llhlength,
        loc='upper left',
        bbox_to_anchor=(0,1.16),
        scatterpoints=1, numpoints=1,fontsize=5.5)
plt.tight_layout()

fig.savefig("stacked_grouped_bar_mixedlm.png", dpi=300, bbox_inches='tight')
plt.show()