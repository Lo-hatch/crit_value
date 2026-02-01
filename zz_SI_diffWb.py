
from mcplot import mcPlot
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import xarray as xr
from scipy.stats import linregress
import matplotlib.ticker as mtick
from mcplot import position, str2tex, abc2plot
import glob
import os
import re
from scipy.optimize import curve_fit
from matplotlib.colors import ListedColormap
import sys
import platform
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import pearsonr
if platform.node()=='zlhp':
    file_path = "/mnt/d/project/hydraulic/test/code/python_public_function/"  # Change this to the actual directory
else:
    file_path = "/home/zlu/python_public_function/"
sys.path.append(file_path)
import read_functions as my_rf
import plot_functions as my_pf
import analyse_functions as my_af
def anova_func(df_new):
    yvar_labels = {
    "(Edemand - TVeg)/Edemand": r'$\frac{(E_{sat}-E)}{E_{sat}}$',
    "(GPP_wb - GPP)/GPP_wb":r'$\frac{(GPP_{sat}-GPP)}{GPP_{sat}}$',
    "(gsw_sw - gs)/gsw_sw":r'$\frac{(gs_{sat}-gs)}{gs_{sat}}$',
    'latent heat fraction': r'$f_{LE}$',

}
    xname_labels = {
    "wb_soilmean": fr'$\theta_{{crit}}$',
    "wb_rwc_soilmean": fr'$REW_{{crit}}^{{\theta}}$',
    "rwc_soilmean_recal":fr'$REW_{{crit}}$',
    "wb_psi_soilmean": fr'$\psi_{{crit}}^{{\theta}}$',
    "psi_soilmean": fr'$\psi_{{crit}}$',
}  
    indexnames=['MaxCur','MaxCurChange','BreakPoint']
    results_list = []  # to collect all results
    results_list2 = []
    for iseg in [0,1,2,3,4]:
        for xname, ylabel in xname_labels.items():
            for yvar, yvarlabel in yvar_labels.items():
                for idxname in indexnames:
                    # subset rows for this combination
                    subset = df_new[(df_new["segID"] == iseg) & (df_new["Xname"] == xname) & (df_new["yvar"] == yvar)]

                    
                    if subset.shape[0] > 0:  # only if data exists
                        try:
                            # build model: value ~ soiltype + sites
                            model = ols(f"{idxname} ~ C(soiltype) + C(site)", data=subset).fit()
                            anova_res = sm.stats.anova_lm(model, typ=2)  # Type II ANOVA
                            
                            # flatten table into dataframe with identifiers
                            res_df = anova_res.reset_index().rename(columns={"index": "Source"})
                            if iseg<4:
                                seglab = f'seg{iseg+1}'
                            else:
                                seglab = 'whole'
                            res_df["segID"] = seglab
                            res_df["yvar"] = yvar
                            res_df["xname"] = xname
                            res_df["indexnames"] = idxname
                            results_list.append(res_df)

                            total_ss = anova_res["sum_sq"].sum()
                            rel_soil = anova_res.loc["C(soiltype)", "sum_sq"] / total_ss if "C(soiltype)" in anova_res.index else None
                            rel_sites = anova_res.loc["C(site)", "sum_sq"] / total_ss if "C(site)" in anova_res.index else None
                            
                            rel_df = pd.DataFrame([{
                                "Relative_soiltype": rel_soil,
                                "Relative_site": rel_sites,
                                "segID":seglab,
                                "yvar": yvar,
                                "xname": xname,
                                "indexnames": idxname
                            }])
                            
                            results_list2.append(rel_df)
                        except Exception as e:
                            print(f"Skipping {yvarlabel}, {ylabel}, {idxname} due to error: {e}")
                            continue
    final_results = pd.concat(results_list, ignore_index=True)
    final_results2 = pd.concat(results_list2, ignore_index=True)
    return final_results,final_results2
def logi_excle(file_path,tag):
    df_all = []
    for isite in sites:
        # if isite in ['AU-ASM', 'AU-Gin', 'US-SRM']:
        #     df = pd.read_excel(file_path, sheet_name=f'{isite}_wbLAI')
        #     #df = pd.read_excel(file_path, sheet_name=f'{isite}')
        # else:
        #     df = pd.read_excel(file_path, sheet_name=isite)
        df = pd.read_excel(file_path, sheet_name=f'{isite}{tag}')
        #if ifdrydown:
        df['method'] = 'NTD'
        # conditions = [
        #     df["Xname"].isin(["wb_soilmean", "wb_rwc_soilmean"]),
        #     df["Xname"] == "rwc_soilmean_recal"
        # ]

        # choices = [
        #     df["MaxCurChange"],
        #     df["MinCurChange"]
        # ]
        # df["best"] = np.select(conditions, choices, default=np.nan)

        conditions = [
            (df["Xname"] == "rwc_soilmean_recal") & (df["MinCurChange"] > df["MaxCur"]),
            (df["Xname"] == "wb_soilmean") & ((df["MaxCur"] - df["MaxCurChange"]) < 0.02) & ((df["MaxCur"] - df["MaxCurChange"]) > 0)
        ]

        choices = [
            (df["MinCurChange"]+df["MaxCur"])/2,
            df["MaxCurChange"]
        ]

        df["best"] = np.select(conditions, choices, default=df["MaxCur"])
        cols = ['MaxCur', 'MaxCurChange', 'MinCurChange','best','cons0p05','cons0p05Kernel','cons0p04','cons0p03','slope0p05','slope0p1','slope0p15']


        ################################################### convert wb to rwc as append as new rows where xname='wb_rwc_soilmean'
        df_new = df.copy()
        wb_to_rwc_list = ['wb_soilmean','wb_30','wb_fr_rootzone','wb_depth_rootzone']
        for iwb in wb_to_rwc_list:
            df_wb = df[df['Xname'] == iwb].copy()

            # 2. Change Xname for these duplicated rows
            df_wb['Xname'] = f'{iwb}_rwc'

            # 3. Merge to get swilt and sfc values for each soiltype (if not already present)
            # (skip if already in df_wb)
            df_wb = df_wb.merge(soil_df[['soiltype', 'swilt', 'sfc']], on='soiltype', how='left')

            # 4. Recalculate MaxCur and MaxCurChange with the formula:
            def normalize_val(val, swilt, sfc):
                if pd.isna(val) or pd.isna(swilt) or pd.isna(sfc) or sfc == swilt:
                    return float('nan')
                return (val - swilt) / (sfc - swilt)
            for c in cols:
                df_wb[c] = df_wb.apply(lambda row: normalize_val(row[c], row['swilt'], row['sfc']), axis=1)
    
            # 5. Drop the helper columns if you want
            df_wb = df_wb.drop(columns=['swilt', 'sfc'])

            # 6. Append new rows to original dataframe
            df_new = pd.concat([df_new, df_wb], ignore_index=True)

            #################################################### convert wb to psi
            df_wb = df[df['Xname'] == iwb].copy()
            df_wb['Xname'] = f'{iwb}_psi'
            df_wb = df_wb.merge(soil_df[['soiltype', 'ssat', 'sucs','bch']], on='soiltype', how='left')
            # 4. Recalculate MaxCur and MaxCurChange with the formula:
            def theta2psi(val, ssat,sucs, bch):
                if val < 0:
                    return np.nan
                psi_sat = sucs * 1000 * 9.81  / 1e6
                return psi_sat * (val/ssat)**(-bch)
            for c in cols:
                df_wb[c] = df_wb.apply(lambda row: theta2psi(row[c], row['ssat'], row['sucs'],row['bch']), axis=1)
            df_wb = df_wb.drop(columns=[ 'ssat', 'sucs','bch'])
            df_new = pd.concat([df_new, df_wb], ignore_index=True)
            ################################################### convert wb to psi_vG
            df_wb = df[df['Xname'] == iwb].copy()
            df_wb['Xname'] = f'{iwb}_psivG'
            df_wb = df_wb.merge(soil_df[['soiltype', 'ssat', 'sres','alpha','n']], on='soiltype', how='left')

            for c in cols:
                df_wb[c] = df_wb.apply(lambda row: my_af.theta2psi_vG(row[c], row['sres'], row['ssat'],row['alpha'],row['n']), axis=1)
            df_wb = df_wb.drop(columns=[ 'ssat', 'sres','alpha','n'])
            df_new = pd.concat([df_new, df_wb], ignore_index=True)

        df_new = df_new.merge(soil_df[['soiltype', 'sand fraction']], on='soiltype', how='left')

        df_new["site"] = isite
        df_all.append(df_new)
    df_new = pd.concat(df_all, axis=0, ignore_index=True)
    df_new["PET_P"] = df_new["site"].map(petp_dict)
    return df_new
def LP_excle(file_path,tag):
    df_all = []
    for isite in sites:
        try:
            df = pd.read_excel(file_path, sheet_name=f'{isite}{tag}')
        except Exception as e:
            continue
        #if ifdrydown:
        df['method'] = 'EF'
        df["BreakPoint"] = df["BreakPoint"].where((df["BreakPoint"] > 0) & (df["BreakPoint"] < 1))
        df["best"] = np.where(
            df["R2_LP"] > 0.75, 
            df["BreakPoint"], 
            np.where((df["R2"] - df["R2_LP"])>0.05, df["MaxCur"], df["BreakPoint"])
        )
        df_new = df.copy()
        wb_to_rwc_list = ['wb_soilmean','wb_30','wb_fr_rootzone','wb_depth_rootzone']
        for iwb in wb_to_rwc_list:
        ############################## convert wb to rwc as append as new rows where xname='wb_rwc_soilmean'
            df_wb = df[df['Xname'] == iwb].copy()

            # 2. Change Xname for these duplicated rows
            df_wb['Xname'] = f'{iwb}_rwc'
            cols = ['MaxCur', 'MaxCurChange', 'MinCurChange','best','BreakPoint']
            # 3. Merge to get swilt and sfc values for each soiltype (if not already present)
            # (skip if already in df_wb)
            df_wb = df_wb.merge(soil_df[['soiltype', 'swilt', 'sfc']], on='soiltype', how='left')

            # 4. Recalculate MaxCur and MaxCurChange with the formula:
            def normalize_val(val, swilt, sfc):
                if pd.isna(val) or pd.isna(swilt) or pd.isna(sfc) or sfc == swilt:
                    return float('nan')
                return (val - swilt) / (sfc - swilt)
            for c in cols:
                df_wb[c] = df_wb.apply(lambda row: normalize_val(row[c], row['swilt'], row['sfc']), axis=1)

            # 5. Drop the helper columns if you want
            df_wb = df_wb.drop(columns=['swilt', 'sfc'])
            df_new = pd.concat([df_new, df_wb], ignore_index=True)
            

            ################################ convert wb to psi
            df_wb = df[df['Xname'] == iwb].copy()
            df_wb['Xname'] = f'{iwb}_psi'
            df_wb = df_wb.merge(soil_df[['soiltype', 'ssat', 'sucs','bch']], on='soiltype', how='left')
            # 4. Recalculate MaxCur and MaxCurChange with the formula:
            def theta2psi(val, ssat,sucs, bch):
                if val < 0:
                    return np.nan
                psi_sat = sucs * 1000 * 9.81  / 1e6
                return psi_sat * (val/ssat)**(-bch)
            for c in cols:
                df_wb[c] = df_wb.apply(lambda row: theta2psi(row[c], row['ssat'], row['sucs'],row['bch']), axis=1)

            df_wb = df_wb.drop(columns=[ 'ssat', 'sucs','bch'])
            df_new = pd.concat([df_new, df_wb], ignore_index=True)
            ################################################### convert wb to psi_vG
            df_wb = df[df['Xname'] == iwb].copy()
            df_wb['Xname'] = f'{iwb}_psivG'
            df_wb = df_wb.merge(soil_df[['soiltype', 'ssat', 'sres','alpha','n']], on='soiltype', how='left')

            for c in cols:
                df_wb[c] = df_wb.apply(lambda row: my_af.theta2psi_vG(row[c], row['sres'], row['ssat'],row['alpha'],row['n']), axis=1)
            df_wb = df_wb.drop(columns=[ 'ssat', 'sres','alpha','n'])
            df_new = pd.concat([df_new, df_wb], ignore_index=True)

        df_new = df_new.merge(soil_df[['soiltype', 'sand fraction']], on='soiltype', how='left')
        df_new["site"] = isite
        df_all.append(df_new)
    df_new = pd.concat(df_all, axis=0, ignore_index=True)
    df_new["PET_P"] = df_new["site"].map(petp_dict)
    return df_new

class PlotIt(mcPlot):

    # -------------------------------------------------------------------------
    # init
    #
    def __init__(self, *args, **kwargs):
        """ initialisation """
        # self.plotname = '/mnt/d/project/hydraulic/test/test.pdf'
        # self.outtype = 'pdf'
        super().__init__(*args, **kwargs)
      
        self.bottom = 0.05
        self.top = 0.85
        self.nrow     = 5     # # of rows of subplots per figure
        self.ncol     = 1    # # of columns of subplots per figure
        self.hspace   = 0.07  # x-space between subplots
        self.vspace   = 0.03  # y-space between subplots
        self.textsize = 10    # standard text size
        self.dxabc    = 0  # % of (max-min) shift to the right
                              # of left y-axis for a,b,c,... labels
        self.dyabc    = 1.04  # % of (max-min) shift up from lower x-axis
                              # for a,b,c,... labels
        # legend
        self.llxbbox    = 0.75  # x-anchor legend bounding box
        self.llybbox    = 0.99  # y-anchor legend bounding box
        self.llrspace   = 1    # spacing between rows in legend
        self.llcspace   = 1.0   # spacing between columns in legend
        self.llhtextpad = 0.4   # pad between the legend handle and text
        self.llhlength  = 1.5   # length of the legend handles

        self.rasterized = True
        self.dpi = 150

        self.set_matplotlib_rcparams()

        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 7,            # Default global font size
            "axes.labelsize": 8,       # Axis title size
            "axes.titlesize": 8,       # Individual panel title size
            "xtick.labelsize": 7,      # Tick label size
            "ytick.labelsize": 7,      # Tick label size
            "legend.fontsize": 7,      # Legend text size
           # "figure.figsize": (3.5, 3) # 3.5 inches is roughly 88-90mm (one column)
        })

    def plot_new(self,df_merge,xvar,yvar,clcvar,xlab,ylab,tit,iplot,huevar=None,colors=None,y_lim=None,iflegend=False,ncol=1,plottype='scatter'):
        nplotperpage = self.nrow * self.ncol
        if iplot == 0:
            self.ifig += 1
            fig = plt.figure(self.ifig)
            self.fig = fig
        else:
            fig = self.fig
        iplot += 1
        pos  = position(self.nrow, self.ncol, iplot,
                            hspace=self.hspace, vspace=self.vspace)
        if self.ncol>1:
            pos2  = position(self.nrow, self.ncol, iplot+1,
                                hspace=self.hspace, vspace=self.vspace)
            iplot += 1
            pos[2] = pos2[0]-pos[0]+pos2[2]
        ax1 = fig.add_axes(pos)
        ax1.tick_params(axis='y', colors='k')
        ax1.yaxis.label.set_color('k')
        n = 0
        from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype
        
        yvars = yvar
        if isinstance(yvar, str):
            yvars = [yvar]
        mrks = ['o','x']
        for i,yvar in enumerate(yvars):
            df_plot = df_merge.copy()
            if is_string_dtype(df_merge[xvar]) or is_object_dtype(df_merge[xvar]) or is_categorical_dtype(df_merge[xvar]):
                categories = df_plot[xvar].unique()
                cat_to_num = {cat: i for i, cat in enumerate(categories)}

                df_plot["x_pos"] = df_plot[xvar].map(cat_to_num)
            else:
                df_plot["x_pos"] =  df_plot[xvar]
            jitter_amount = 0.1

            # create jitter grouped by xvar
            df_plot[xvar] = df_plot.groupby("x_pos")["x_pos"].transform(
                lambda x: x + np.linspace(-jitter_amount, jitter_amount, len(x))
            )
            # Custom color list (3 colors for 3 yvar values)
            if colors is None:
                colors = ["#34119c", "#ff7f0e", "#2ca02c"]  # blue, orange, green
            if plottype=='scatter':
                sns.scatterplot(
                    data=df_plot,
                    x=xvar,
                    #x = "x_jitter",
                    y=yvar,
                    hue=clcvar,
                    marker=mrks[i],
                    #style=clcvar,  
                    alpha=0.6,
                    palette=colors,  # List of 3 colors
                    ax=ax1,
                    s=100,legend=iflegend)
            elif plottype=='box':
                sns.boxplot(
                    data=df_merge,
                    x=xvar,
                    y=yvar,
                    ax=ax1,
                    fliersize=2,
                    linewidth=0.8,
                    width=0.6,
                    legend=iflegend 
                )
            elif plottype=='box2':
                df_merge = df_merge.dropna()

                sns.boxplot(
                    data=df_merge,
                    x=xvar,
                    y=yvar,
                    hue=huevar,
                    ax=ax1,
                    fliersize=2,
                    linewidth=0.8,
                    width=0.6,
                    legend=iflegend,
                    palette='Set2',
                )
            if y_lim == (0,1):
                plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])  

        if len(yvars)==2:
            for i,iwhole in enumerate([ 'UIFC', 'CIFC']):
                df_merge_whole = df_merge[(df_merge["site"] == iwhole)]
                x = df_merge_whole[yvars[0]]
                y = df_merge_whole[yvars[1]]
                rmse = np.sqrt(np.mean((x - y)**2))
                plt.text(0.8,0.95-i*0.1,f'{iwhole} RMSE:{my_pf.numstrFormat(rmse,3)}' ,verticalalignment='top', horizontalalignment='left',
            transform=ax1.transAxes, fontsize=7)
        # x_positions = df_merge[xvar].unique()
        # soil_types = df_merge.sort_values(xvar)['soiltype'].unique()

        # # Loop through each tick position and add text
        # for x, soil in zip(x_positions, soil_types):
        #     ax1.text(
        #         x,                              # x position (tick)
        #         ax1.get_ylim()[0] - 0.05 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),  # slightly below axis
        #         soil,                           # text label
        #         ha='center', va='top',
        #         rotation=45, fontsize=6
        #     )
        #plt.xlabel('Soil type')
        plt.ylabel(ylab)
       
        plt.title(tit,fontsize=6)
        #plt.grid(True, color='gray', linewidth=0.4)  
        plt.grid(axis='y',color='gray', linestyle='--', linewidth=0.4, alpha=0.7)
        xticklabels = ax1.get_xticklabels()
        # If tick labels are numeric or convertible to numbers
        try:
            numeric_ticks = [float(x.get_text()) for x in xticklabels]
            ax1.set_xticklabels([f"{x:.2f}" for x in numeric_ticks])
        except ValueError:
            # If not numeric, skip formatting
            pass
        if is_string_dtype(df_merge[xvar]) or is_object_dtype(df_merge[xvar]) or is_categorical_dtype(df_merge[xvar]):
            ax1.set_xticks(list(cat_to_num.values()))
            ax1.set_xticklabels(list(cat_to_num.keys()))
        if iflegend:
            ll = ax1.legend(frameon=self.frameon, ncol=ncol,
                    labelspacing=self.llrspace,
                    handletextpad=self.llhtextpad,
                    handlelength=self.llhlength,
                    loc='upper left',
                    bbox_to_anchor=(self.llxbbox,self.llybbox),
                    scatterpoints=1, numpoints=1,fontsize=5.5)
        abc2plot(ax1, self.dxabc, self.dyabc,
                (self.ifig - 1) * nplotperpage + iplot,
                lower=True, bold=True, usetex=self.usetex, mathrm=True,transform=ax1.transAxes,large=True)
        plt.tight_layout()

        if y_lim is not None:
            plt.setp(ax1, ylim=y_lim) 
        if (iplot == nplotperpage):
        # save one pdf page, zihanlu
            self.plot_save(fig)
            plt.savefig(f"{fout}{figname}{self.ifig}.png", dpi=300, bbox_inches="tight")
            iplot = 0
        return iplot
    def plot_diffwb(self,df,iplot,y_lim = None,y_label=None,tit = None):
        nplotperpage = self.nrow * self.ncol
        if iplot == 0:
            self.ifig += 1
            fig = plt.figure(self.ifig)
            self.fig = fig
        else:
            fig = self.fig
        iplot += 1
        pos  = position(self.nrow, self.ncol, iplot,
                            hspace=self.hspace, vspace=self.vspace)
        if self.ncol>1:
            pos2  = position(self.nrow, self.ncol, iplot+1,
                                hspace=self.hspace, vspace=self.vspace)
            iplot += 1
            pos[2] = pos2[0]-pos[0]+pos2[2]
        ax = fig.add_axes(pos)
        ax.tick_params(axis='y', colors='k')
        ax.yaxis.label.set_color('k')
        # df has columns: soil_type, crit, method

        soil_types = df["soiltype"].unique()
        methods = df["method"].unique()
        var_legend_handles = {}
        method_legend_handles = {}
        bar_width = 0.35
        offsets = np.linspace(-bar_width/2, bar_width/2, len(methods))

        # Assign a default color map
        colors = plt.cm.tab10(range(len(methods)))
        mrks = ['o','x']
        for m_idx, method in enumerate(methods):
            color = colors[m_idx]
            x_offset = offsets[m_idx]
            if m_idx == 0:
                method_legend_handles[method] = plt.Line2D(
                    [0], [0],
                    marker=mrks[m_idx],
                    linestyle='',
                    markerfacecolor='none',
                    markeredgecolor='k',
                    label=method
    )
            else:
                method_legend_handles[method] = plt.Line2D(
                    [0], [0],
                    marker=mrks[m_idx],
                    linestyle='',
                    markerfacecolor='k',
                    markeredgecolor='k',
                    label=method
    )
            for i, soil in enumerate(soil_types):
                subset = df[(df["soiltype"] == soil) & (df["method"] == method)]
                subset = subset.reset_index(drop=True)
                x_pos = i + x_offset
                
                point_colors = plt.cm.tab10(range(len(subset)))
                for idx, row in subset.iterrows():
                    y = row["crit"]
                    var = row["Xname"]
                    if 'soilmean' in var:
                        lab = 'mean'
                    elif '30' in var:
                        lab = '30cm'
                    elif '_fr_' in var:
                        lab = 'Wfr'
                    elif '_depth_' in var:  
                        lab = 'Wdp'
                    

                    if pd.isna(y):
                        continue
                    if not np.isnan(y):
                        if m_idx==0:
                            sc = ax.scatter(
                                x_pos,
                                y,
                                marker=mrks[m_idx],
                                s=55,
                                facecolors="none",
                                edgecolors=point_colors[idx],
                                label = lab
                            )
                            var_legend_handles[lab] = sc
                        else:
                            ax.scatter(
                                x_pos,
                                y,
                                marker=mrks[m_idx],
                                s=55,
                                facecolors=point_colors[idx],
                                edgecolors=point_colors[idx],
                            
                            )
                    # varname legend: only add first time each var appears

                # vals_clean = np.array(vals)[~np.isnan(vals)]
                
                # if len(vals_clean) == 0:
                #     continue
                
                # 
                # vals_sorted = np.sort(vals_clean)   
                # if len(vals_sorted) == 1:
                #     ax.scatter(
                #         x_pos, vals_sorted[0],
                #         marker='o', s=40, facecolors='none', edgecolors=color
                #     )
                # elif len(vals_sorted) == 2:
                #     ymin, ymax = vals_sorted
                #     ax.bar(x_pos, ymax - ymin, bottom=ymin, width=bar_width, color=color, alpha=0.3)

                # elif len(vals_sorted) > 2:
                #     ymin = vals_sorted[0]
                #     ymax = vals_sorted[-1]

                #     # always draw bar
                #     ax.bar(x_pos, ymax - ymin, bottom=ymin, width=0.4,
                #         color=color, alpha=0.3)

                #     # if more than 2 values, draw dots for middle ones
                
                #     mids = vals_sorted[1:-1]
                #     ax.scatter([x_pos] * len(mids), mids,
                #             marker='o', facecolors='none',
                #             edgecolors=color, s=40)
        crit_values = df["crit"].dropna()

        crit_min = crit_values.min()
        crit_max = crit_values.max()
        if iplot==1:
            xpos = 0.15
            ypos = 0.02
        else:
            xpos = 0.98
            ypos = 0.02
        ax.text(
            xpos, ypos,
            f"min = {crit_min:.3f}\nmax = {crit_max:.3f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=7,
          #  bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
        )
        ax.set_xticks(range(len(soil_types)))
        if iplot<4:
            ax.set_xticklabels([])
        else:
            
            ax.set_xticklabels(soil_types)
        #ax.set_xlabel("Soil Type")
        ax.set_ylabel(y_label)
        plt.tight_layout()
        if y_lim is not None:
            plt.setp(ax, ylim=y_lim) 
        if tit is not None:
            plt.title(tit,fontsize=6)
        if iplot==1:
            legend1 = ax.legend(
                handles=method_legend_handles.values(),
                ncol = 2,
                loc="upper left",
                bbox_to_anchor=(0.2, 1.3),
                frameon=self.frameon
            )

            legend2 = ax.legend(
                handles=var_legend_handles.values(),
                ncol = 4,
                loc="upper left",
                bbox_to_anchor=(0.2, 1.2),
                frameon=self.frameon
            )

            ax.add_artist(legend1)
        abc2plot(ax, self.dxabc, self.dyabc,
                (self.ifig - 1) * nplotperpage + iplot,
                lower=True, bold=True, usetex=self.usetex, mathrm=True,transform=ax.transAxes,large =True)
        if (iplot == nplotperpage):
        # save one pdf page, zihanlu
            self.plot_save(fig)
            plt.savefig(f"{fout}{figname}{self.ifig}.png", dpi=300, bbox_inches="tight")
            iplot = 0
        return iplot
if __name__ == '__main__':

    # one model in one figure
    # three figures for each value of the tested variable: GPP,Qle, psi_can
    # show 2015-2016 period, dry and wet year of FR-Hes

    import time as ptime
    import glob
    t1 = ptime.time()


    self = PlotIt('Scatter Plot')

    suffmod = '/outputs/site_out_cable_*.nc'
    if platform.node()=='zlhp':
        
        fp1='/mnt/d/project/hydraulic/test/'
        fpobs = '/mnt/d/project/hydraulic/test/obs4/'
        fmet = '/mnt/d/project/hydraulic/test/input/met/'

        file_path1 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_BASEdc98fe39.xlsx'
        file_path2 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_drydown_BASEdc98fe39.xlsx'
        file_path3 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_LP_BASEdc98fe39.xlsx'
        file_path4 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_LP_drydown_BASEdc98fe39.xlsx'

        file_path1 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_BASEdc98fe39_LAILBr.xlsx'
        file_path1 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_BASEdc98fe39_gompertz_LBr.xlsx'
        file_path1 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_BASEdc98fe39.xlsx'
        #file_path1 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_BASEdc98fe39_xcons_Dsig_LBr.xlsx'
      
        file_path2 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_drydown_BASEdc98fe39_LAILBr.xlsx'
        file_path3 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_LP_BASEdc98fe39_LAILBr.xlsx'
     
        file_path4 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_BASEdc98fe39.xlsx'
        fout = '/mnt/d/project/hydraulic/test/plot/fig_final/v20251217/'
        fout3 = '/mnt/d/project/hydraulic/test/plot/fig_final/V20251111/'
        fout4 = '/mnt/d/project/hydraulic/test/plot/fig_final/V20251111/'
        outname = f"{fout4}anova_results_san2wet.xlsx"
        outname_stat1 = f'{fout3}stats_san2wet.xlsx'
        outname_multicomp = f'{fout4}multicomp.xlsx'
    else:
        fp1='/home/zlu/project/test1105/'
        fpobs='/home/mcuntz/projects/coco2/obs4/'
        fmet = '/home/mcuntz/projects/coco2/cable-pop/input/'

        mapping = {3: 1, 6: 2, 2: 3, 5:4, 7:5, 4:6, 1:7}

    plotorstat = {
        'plotsingle':{'plot':False,'figname':'z03_SI_diffwb'},
         'plotall':{'plot':False,'figname':'z03_theta_soiltype_climate_ALLplotV2'},
          'plotallBox':{'plot':True,'figname':'z03_SI_diffwb'},
                   }
   
  
    soil_data = {
    1: {'soiltype':'fine clay','sres':0.098,'swilt': 0.286,'swilt0':0.286 , 'sfc': 0.367, 'sfc0':0.401 , 'ssat': 0.482,'bch':11.4,'sucs':-0.405,'sand fraction': 0.16,'alpha':0.008,'n':1.09},
    2: {'soiltype':'silty clay','sres':0.111,'swilt': 0.283, 'swilt0':0.277 , 'sfc': 0.370, 'sfc0': 0.401, 'ssat': 0.482,'bch':10.4,'sucs':-0.49,'sand fraction': 0.27,'alpha':0.005,'n':1.23},
    3: {'soiltype':'clay loam','sres':0.078,'swilt': 0.216, 'swilt0':0.219 , 'sfc': 0.301, 'sfc0': 0.376, 'ssat': 0.479,'bch':7.1,'sucs':-0.591,'sand fraction': 0.37,'alpha':0.019,'n':1.31},
    4: {'soiltype':'sandy clay','sres':0.117,'swilt': 0.219,'swilt0':0.219 ,  'sfc': 0.310, 'sfc0': 0.317, 'ssat': 0.426,'bch':10.4,'sucs':-0.153,'sand fraction': 0.52,'alpha':0.059,'n':1.48},
    5: {'soiltype':'sandy clay loam','sres':0.063,'swilt': 0.175,'swilt0': 0.175,  'sfc': 0.255, 'sfc0': 0.300, 'ssat': 0.420,'bch':7.12,'sucs':-0.299,'sand fraction': 0.58,'alpha':0.059,'n':1.48},
    6: {'soiltype':'Sandy loam','sres':0.061,'swilt': 0.135,'swilt0':0.136 ,  'sfc': 0.218,  'sfc0': 0.286,'ssat': 0.443,'bch':5.15,'sucs':-0.348,'sand fraction': 0.60,'alpha':0.036,'n':1.56},
    7: {'soiltype': 'sand','sres':0.051,'swilt': 0.072,'swilt0': 0.070,  'sfc': 0.143,  'sfc0':0.176 ,'ssat': 0.398,'bch':4.2,'sucs':-0.106,'sand fraction': 0.83,'alpha':0.145,'n':2.68}
    }

    soil_df = pd.DataFrame.from_dict(soil_data, orient='index')
    soil_df = soil_df.drop(['swilt', 'sfc'], axis=1)
    soil_df = soil_df.rename(columns={'swilt0': 'swilt','sfc0': 'sfc'})
    ####################################################################################################################
    sites=["AU-ASM", "US-SRM" ,"CA-SF2",
           "AU-Gin" ,
           "US-Me6","CN-Cng" ,"ZM-Mon","IT-Cp2" ,
           "IT-SRo","IT-Ro1" ,"MY-PSO", 
           "AU-How", "FR-LBr","FR-Pue","CA-TPD", "CN-Qia", "SE-Nor","GF-Guy", ]
    ####################################################################################################################
    sites=["AU-ASM", "US-SRM" ,"CA-SF2",
           "AU-Gin" ,
           "US-Me6","CN-Cng" ,"ZM-Mon","IT-Cp2" ,
           "IT-SRo","IT-Ro1" ,"MY-PSO", 
           "AU-How", "FR-LBr","FR-Pue","CA-TPD", "CN-Qia" ]
    sites = ['FR-LBr']
    sites = ["US-SRM" , "CA-SF2" , "IT-Cp2" ,'FR-LBr',"CN-Qia","UIFC" ]
    petp_dict = {
        "AU-ASM":4.76445522,
          "US-SRM": 3.580179244,
          "CA-SF2":2.767625955,
          "AU-Gin": 2.341442582,
          "US-Me6":2.112320235,
          "CN-Cng":2.1106894 ,
          "ZM-Mon":2.109279645,
           "IT-Cp2":2.086824352,
           "IT-SRo":1.199628595,
           "IT-Ro1":1.144765423,
           "MY-PSO":1.025191584, 
           "AU-How":1.009518835,
             "FR-LBr":0.919247979,
             "FR-Pue":0.889934449,
             "CA-TPD":0.804239921,
               "CN-Qia":0.717219071, 
               "SE-Nor":0.499837519,
               "GF-Guy":0.494577805,
               "UIFC":np.nan,
            "CIFC":np.nan,
    }
    
    idxname='best'
    imethod='percentile'
    iplot=0

    df1 = logi_excle(file_path1,'_NTD')
    
    #df1dd = logi_excle(file_path2,True)
    #df1 = pd.concat([df1, df1dd], axis=0)
    #df2 = LP_excle(file_path3,False)
    df2 = LP_excle(file_path4,'_EF')
    
    #df2 = pd.concat([df2, df2dd], axis=0)
    xname_labels = {
     "wb_soilmean": fr'$\theta_{{mean,crit}}$',
 "wb_30": fr'$\theta_{{30,crit}}$',
  "wb_fr_rootzone": fr'$\theta_{{fr,crit}}$',
  "wb_depth_rootzone": fr'$\theta_{{dp,crit}}$',
    "wb_soilmean_rwc": fr'$REW_{{crit}}^{{\theta_{{mean}}}}$',
    "wb_30_rwc": fr'$REW_{{crit}}^{{\theta_{{30}}}}$',
    "wb_fr_rootzone_rwc": fr'$REW_{{crit}}^{{\theta_{{fr}}}}$',
    "wb_depth_rootzone_rwc": fr'$REW_{{crit}}^{{\theta_{{dp}}}}$',
    #"wb_pos_rwc_soilmean": fr'$REW_{{critpos}}^{{\theta}}$',
    "rwc_soilmean_recal":fr'$REW_{{mean,crit}}$',
     "rwc_30":fr'$REW_{{30,crit}}$',
      "rwc_fr_rootzone":fr'$REW_{{fr,crit}}$',
       "rwc_depth_rootzone":fr'$REW_{{dp,crit}}$',
     "wb_soilmean_psi": fr'$\psi_{{crit}}^{{\theta_{{mean}}}}$',
    "wb_30_psi": fr'$\psi_{{crit}}^{{\theta_{{30}}}}$',
    "wb_fr_rootzone_psi": fr'$\psi_{{crit}}^{{\theta_{{fr}}}}$',
    "wb_depth_rootzone_psi": fr'$\psi_{{crit}}^{{\theta_{{dp}}}}$',

    "psi_soilmean": fr'$\psi_{{mean,crit}}$',
 "psi_30": fr'$\psi_{{30,crit}}$',
 "psi_fr_rootzone": fr'$\psi_{{fr,crit}}$',
  "psi_depth_rootzone": fr'$\psi_{{dp,crit}}$',
    }  
    xname_unit ={'wb':fr'$\theta_{{crit}}$',
                    'wb_rwc': fr'$REW_{{crit}}^{{\theta}}$',
                    'rwc':fr'$REW_{{crit}}$',
                    'wb_psi': fr'$\psi_{{crit}}^{{\theta}}$',
                    'wb_psi_vG': fr'$\psi vG_{{crit}}^{{\theta}}$'}
    xname_labels = {'wb':["wb_soilmean","wb_30", "wb_fr_rootzone", "wb_depth_rootzone"],
                    'wb_rwc':["wb_soilmean_rwc","wb_30_rwc", "wb_fr_rootzone_rwc", "wb_depth_rootzone_rwc"],
                   # 'rwc':["rwc_soilmean_recal","rwc_30", "rwc_fr_rootzone", "rwc_depth_rootzone"],
                    'wb_psi':["wb_soilmean_psi","wb_30_psi", "wb_fr_rootzone_psi", "wb_depth_rootzone_psi"],
                    'wb_psi_vG':["wb_soilmean_psivG","wb_30_psivG", "wb_fr_rootzone_psivG", "wb_depth_rootzone_psivG"]}
    # xname_labels = {'wb':["wb_soilmean","wb_30", "wb_fr_rootzone"],
    #                 'wb_rwc':["wb_soilmean_rwc","wb_30_rwc", "wb_fr_rootzone_rwc"],
    #                 'rwc':["rwc_soilmean_recal","rwc_30", "rwc_fr_rootzone"],
    #                 'wb_psi':["wb_soilmean_psi","wb_30_psi", "wb_fr_rootzone_psi"]}
    # xname_labels = {'wb':["wb_soilmean","wb_30"],
    #                 'wb_rwc':["wb_soilmean_rwc","wb_30_rwc", ],
    #                 'rwc':["rwc_soilmean_recal","rwc_30",],
    #                 'wb_psi':["wb_soilmean_psi","wb_30_psi",]}
    yvarsdf1 = [
        ("(Edemand - TVeg)/Edemand", r'$\frac{(E_{sat}-E)}{E_{sat}}$'),
        #  ("(GPP_wb - GPP)/GPP_wb", r'$\frac{(GPP_{sat}-GPP)}{GPP_{sat}}$'),
        #   ("(gsw_sw - gs)/gsw_sw", r'$\frac{(gs_{sat}-gs)}{gs_{sat}}$'),
        #  ("(Edemand - TVeg)/Edemand", r'$\frac{(E_{sat}-E)}{E_{sat}}$'),
        #  ("(GPP_wb - GPP)/GPP_wb", r'$\frac{(GPP_{sat}-GPP)}{GPP_{sat}}$'),
        #  ("(gsw_sw - gs)/gsw_sw", r'$\frac{(gs_{sat}-gs)}{gs_{sat}}$')
    ]

    yvarsdf2 = [
        # ("Transpiration",'E'),
        # ("GPP","GPP"),
        # ("gsw","gsw"),
        ('latent heat fraction', r'$f_{LE}$'),
        #  ('latent heat fraction','latent heat fraction'),
        #  ('latent heat fraction','latent heat fraction')
    ]
    if plotorstat["plotsingle"]["plot"]:
        figname = plotorstat["plotsingle"]["figname"]
        self.nrow     = 5
        self.llxbbox    = 0.75  # x-anchor legend bounding box
        self.llybbox    = 0.99  # y-anchor legend bounding box
        iplot=0
        #sites = ['FR-LBr','AU-ASM','GF-Guy','US-SRM','SE-Nor']
        sites = ['FR-LBr']
        indexnames1=['best']
        indexnames2 =['best'] 
        indexnames1=['MaxCurChange']
        indexnames1=['MaxCurChange','cons0p05','cons0p05Kernel','slope0p05','slope0p1']
        indexnames2 =['BreakPoint','BreakPoint','BreakPoint','BreakPoint','BreakPoint'] 
        keyvar = ['soiltype','sand fraction']
        iseg1 = 3
        iseg2 = 4
        y_lim = None
        for xname, xlabel in xname_labels.items():
            if xname == 'wb':
                y_lim = (0.1,0.4)
            elif xname == 'wb_rwc':
                y_lim = (0, 1)
            elif xname == 'rwc':
                y_lim = (0, 1)
            elif xname == 'wb_psi':
                y_lim = (-2,0)
            for (yvar1, val1), (yvar2, val2) in zip(yvarsdf1, yvarsdf2):
                for isite in sites:
                    for idxname1,idxname2 in zip(indexnames1,indexnames2):
                    
                        
                        #for yvar1, yvar1 in zip(yvarsdf1, yvarsdf2):
                            df1_filtered = df1[(df1["site"] == isite) & (df1["Xname"].isin(xlabel)) & (df1["yvar"] == yvar1)  & (df1["segID"] == iseg1) ]
                                     
                            df2_filtered_dd = df2[(df2["site"] == isite) & (df2["Xname"].isin(xlabel)) & (df2["yvar"] == yvar2)  & (df2["segID"] == iseg2) ]
                            df1_filtered=df1_filtered[keyvar + [idxname1]]
                   
                         
                            df2_filtered_dd=df2_filtered_dd[keyvar + [idxname2]]

                            
                            #df1_filtered_dd = df1_filtered_dd.rename(columns={idxname1: f'DD_{idxname1}'})
                            #df2_filtered = df2_filtered.rename(columns={idxname2: f'{val2} {idxname2}'})
                            #df2_filtered_dd = df2_filtered_dd.rename(columns={idxname2: f'{val2} DD_{idxname2}'})

                            #df_merged = df1_filtered.merge(df1_filtered_dd, on=keyvar).merge(df2_filtered, on=keyvar).merge(df2_filtered_dd, on=keyvar)
                            ################# final figure
                            df1_filtered = df1_filtered.rename(columns={idxname1: f'NTD'})
                            df2_filtered_dd = df2_filtered_dd.rename(columns={idxname2: f'EF'})

                            df_merged = df1_filtered.merge(df2_filtered_dd, on=keyvar)

                            df_long = df_merged.melt(id_vars=keyvar, var_name='dataset', value_name='value')
                            #iplot=self.plot_new(df_long,'sand fraction','value','dataset','',xlabel,isite,iplot,y_lim = y_lim,iflegend=True)
                            ################# figure final
                            tit = idxname1
                            iplot=self.plot_new(df_long,'sand fraction','value','dataset','',xlabel,tit,iplot,y_lim = y_lim,iflegend=(iplot==0))

    if plotorstat["plotall"]["plot"]:
        figname = plotorstat["plotall"]["figname"]
        self.nrow = 5
        self.llxbbox    = 0.02  # x-anchor legend bounding box
        self.llybbox    = 1.5  # y-anchor legend bounding box

        self.filename =''
        indexnames=['MaxCur',]#'MaxCurChange','MinCurChange','best']

        colors = ["#34119c", "#090df4", "#3172cb", "#54d3f3","#31d5a7","#0aa16f","#96a014","#e970b4","#920a50","#E70C0C" ]  # blue, orange, green
        colors = [c for c in colors for _ in range(2)]
        colors =  ["#1D1D1D","#4D4D4D","#34119c", "#54d3f3","#96a014","#b06f91", "#920a50"]
        colors = colors[::-1]
        mrks = ['o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^']
        plotname0 = self.plotname
        indexname = 'cons0p05'
        indexnames1=['MaxCurChange','cons0p05','cons0p05Kernel','slope0p05','slope0p1']
        keyvar = ['site','soiltype','sand fraction']
        for xname, xlabel in xname_labels.items():
            if xname in ['wb_soilmean']:
                y_lim = (0.1,0.4)
            elif xname in ['wb_rwc_soilmean']:
                y_lim = (0, 1)
            elif xname in ['rwc_soilmean_recal']:
                y_lim = (0, 1)
            elif xname in ['wb_psi_soilmean','psi_soilmean']:
                y_lim = (-0.6,0)
            else:
                y_lim = (-2,0)
            for (yvar1, val1), (yvar2, val2) in zip(yvarsdf1, yvarsdf2):
                for idxname1 in indexnames1:
                        tit = f'NTD-{idxname1}'
                        #self.filename = f'{plotname0}_{idxname}_{ylabel}'
                        df1_filtered = df1[(df1["segID"] == 3) & (df1["method"] == 'percentile') & (df1["Xname"] == xname) & (df1["yvar"] == yvar1)]
                        
                        df2_filtered_dd = df2[(df2["segID"] == 4) & (df2["method"] == 'drydown') & (df2["Xname"] == xname) & (df2["yvar"] == yvar2)]
                        df1_filtered = df1_filtered.rename(columns={idxname1: f'NTD'})
                        df2_filtered_dd = df2_filtered_dd.rename(columns={'BreakPoint': f'EF'})

                        df_merged = df1_filtered.merge(df2_filtered_dd, on=keyvar)

                       
                        iplot=self.plot_new(df_merged,'soiltype',['NTD','EF'],'site','',xlabel,tit,iplot,colors=colors,iflegend=(iplot==0),y_lim = y_lim)
                        # df_filtered = df1[(df1["segID"] == 4) & (df1["method"] == 'drydown') & (df1["Xname"] == xname) & (df1["yvar"] == yvar1)]
                        # iplot=self.plot_new(df_filtered,'soiltype',indexname,'site','',xlabel,'',iplot,colors=colors,y_lim = y_lim)
                        # df_filtered = df2[(df2["segID"] == 3) & (df2["method"] == 'percentile') & (df2["Xname"] == xname) & (df2["yvar"] == yvar2)]
                        # iplot=self.plot_new(df_filtered,'soiltype',indexname,'site','',xlabel,'',iplot,colors=colors,y_lim = y_lim)
                # df_filtered = df2[(df2["segID"] == 4) & (df2["method"] == 'drydown') & (df2["Xname"] == xname) & (df2["yvar"] == yvar2)]
                # tit = 'EF-breakpoint'
                #iplot=self.plot_new(df_filtered,'soiltype',indexname,'site','',xlabel,'',iplot,colors=colors,y_lim = y_lim)
                #iplot=self.plot_new(df_filtered,'soiltype','BreakPoint','site','',xlabel,tit,iplot,colors=colors,y_lim = y_lim)
    if plotorstat["plotallBox"]["plot"]:
        sites = ['FR-LBr']
        figname = plotorstat["plotallBox"]["figname"]
        indexnames=['MaxCur',]#'MaxCurChange','MinCurChange','best']
        indexnames1=['MaxCurChange','cons0p05','cons0p05Kernel','slope0p05','slope0p1']
        indexnames2 =['BreakPoint','BreakPoint','BreakPoint','BreakPoint','BreakPoint'] 
        indexnames1=['slope0p1']
        indexnames2 =['BreakPoint'] 
        self.nrow=5
        colors = ["#34119c", "#090df4", "#3172cb", "#54d3f3","#31d5a7","#0aa16f","#96a014","#e970b4","#920a50","#E70C0C" ]  # blue, orange, green
        colors = [c for c in colors for _ in range(2)]
        colors = colors[::-1]
        colors = ["#0aa16f","#e970b4",]
        mrks = ['o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^']
        plotname0 = self.plotname
        keyvar = ['Xname','site','soiltype','sand fraction','method','PET_P']
        for xname, xlabel in xname_labels.items():
            ylab = xname_unit[xname]
            if xname == 'wb':
                y_lim = (0.08,0.42)
            elif xname == 'wb_rwc':
                y_lim = (0, 1.1)
            elif xname == 'rwc':
                y_lim = (0, 1.1)
            elif xname == 'wb_psi':
                y_lim = (-0.7,0)
            elif xname == 'wb_psi_vG':
                y_lim = (-0.5,0)
            for (yvar1, val1), (yvar2, val2) in zip(yvarsdf1, yvarsdf2):
                for isite in sites:
                    for idxname1,idxname2 in zip(indexnames1,indexnames2):
                        df1_filtered = df1[(df1["site"] == isite) & (df1["segID"] == 3) & (df1["Xname"].isin(xlabel)) & (df1["yvar"] == yvar1)]
                        df2_filtered_dd = df2[(df2["site"] == isite) & (df2["segID"] == 4) & (df2["Xname"].isin(xlabel)) & (df2["yvar"] == yvar2)]
                        df1_filtered=df1_filtered[keyvar + [idxname1]]
                        #df1_filtered_dd=df1_filtered_dd[keyvar + [idxname1]]
                        #df2_filtered=df2_filtered[keyvar + [idxname2]]
                        df2_filtered_dd=df2_filtered_dd[keyvar + [idxname2]]  
                        df1_filtered = df1_filtered.rename(columns={idxname1: 'crit'})    
                        df2_filtered_dd = df2_filtered_dd.rename(columns={idxname2: 'crit'})    
                        df_merged = pd.concat([df1_filtered, df2_filtered_dd], axis=0)   
                        tit = idxname1
                        iplot = self.plot_diffwb(df_merged,iplot,y_lim = y_lim,y_label = ylab)
                        #df_merged = df1_filtered
                        #self.filename = f'{plotname0}_{idxname}_{ylabel}'
                        #iplot=self.plot_new(df_merged,'PET_P','crit','soiltype','',xlabel,'',iplot,huevar='method',colors=colors,plottype='box2',y_lim = y_lim,iflegend=(iplot==0),ncol=2)
                        #iplot=self.plot_new(df_merged,'sand fraction','crit','','',xlabel,tit,iplot,huevar='method',colors=colors,plottype='box2',y_lim = y_lim,iflegend=(iplot==0),ncol=2)
                        # df_filtered = df1[(df1["segID"] == 4) & (df1["method"] == 'drydown') & (df1["Xname"] == xname) & (df1["yvar"] == yvar1)]
                        # iplot=self.plot_new(df_filtered,'PET_P',indexname,'soiltype','',xlabel,'',iplot,colors=colors,plottype='box',y_lim = y_lim)
                        # df_filtered = df2[(df2["segID"] == 3) & (df2["method"] == 'percentile') & (df2["Xname"] == xname) & (df2["yvar"] == yvar2)]
                        # iplot=self.plot_new(df_filtered,'PET_P',indexname,'soiltype','',xlabel,'',iplot,colors=colors,plottype='box',y_lim = y_lim)
                        #df_filtered = df2[(df2["segID"] == 4) & (df2["method"] == 'drydown') & (df2["Xname"] == xname) & (df2["yvar"] == yvar2)]
                        #iplot=self.plot_new(df_filtered,'PET_P',indexname,'soiltype','',xlabel,'',iplot,colors=colors,plottype='box',y_lim = y_lim)
 
    if iplot>0:
        self.plot_save(self.fig)
        plt.savefig(f"{fout}{figname}{self.ifig}.png", dpi=300, bbox_inches="tight")

    self.close()

    t2    = ptime.time()
    strin = ( '[m]: {:.1f}'.format((t2 - t1) / 60.)
              if (t2 - t1) > 60.
              else '[s]: {:d}'.format(int(t2 - t1)) )
    print('    Time elapsed', strin)