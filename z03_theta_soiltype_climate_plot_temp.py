
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
def logi_excle(file_path,ifdrydown,tag):
    df_all = []
    for isite in sites:
        # if isite in ['AU-ASM', 'AU-Gin', 'US-SRM']:
        #     df = pd.read_excel(file_path, sheet_name=f'{isite}_wbLAI')
        #     #df = pd.read_excel(file_path, sheet_name=f'{isite}')
        # else:
        #     df = pd.read_excel(file_path, sheet_name=isite)
        df = pd.read_excel(file_path, sheet_name=f'{isite}{tag}')
        if ifdrydown:
            df['method'] = 'drydown'
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
        df_wb = df[df['Xname'] == 'wb_soilmean'].copy()

        # 2. Change Xname for these duplicated rows
        df_wb['Xname'] = 'wb_rwc_soilmean'

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
        df_new = pd.concat([df, df_wb], ignore_index=True)

        #################################################### convert wb to psi
        df_wb = df[df['Xname'] == 'wb_soilmean'].copy()
        df_wb['Xname'] = 'wb_psi_soilmean'
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
        df_new = df_new.merge(soil_df[['soiltype', 'sand fraction']], on='soiltype', how='left')

        df_new["site"] = isite
        df_all.append(df_new)
    df_new = pd.concat(df_all, axis=0, ignore_index=True)
    df_new["PET_P"] = df_new["site"].map(petp_dict)
    return df_new
def LP_excle(file_path,ifdrydown,tag):
    df_all = []
    for isite in sites:
        try:
            df = pd.read_excel(file_path, sheet_name=f'{isite}{tag}')
        except Exception as e:
            continue
        if ifdrydown:
            df['method'] = 'drydown'
        df.loc[df["R2_LP"] < df["R2_PLP"], "BreakPoint"] = df["BreakPoint_PLP"]
        df["BreakPoint"] = df["BreakPoint"].where((df["BreakPoint"] > 0) & (df["BreakPoint"] < 1))
        df["best"] = np.where(
            df["R2_LP"] > 0.75, 
            df["BreakPoint"], 
            np.where((df["R2"] - df["R2_LP"])>0.05, df["MaxCur"], df["BreakPoint"])
        )

        ############################## convert wb to rwc as append as new rows where xname='wb_rwc_soilmean'
        df_wb = df[df['Xname'] == 'wb_soilmean'].copy()

        # 2. Change Xname for these duplicated rows
        df_wb['Xname'] = 'wb_rwc_soilmean'
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
        df_new = pd.concat([df, df_wb], ignore_index=True)
        

        ################################ convert wb to psi
        df_wb = df[df['Xname'] == 'wb_soilmean'].copy()
        df_wb['Xname'] = 'wb_psi_soilmean'
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
        self.top = 0.95
        self.nrow     = 4     # # of rows of subplots per figure
        self.ncol     = 1    # # of columns of subplots per figure
        self.hspace   = 0.1  # x-space between subplots (increased from 0.07)
        self.vspace   = 0.10  # y-space between subplots (increased from 0.05)
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
            "font.size": 14,            # Default global font size (increased from 7)
            "axes.labelsize": 16,       # Axis title size (increased from 8)
            "axes.titlesize": 16,       # Individual panel title size (increased from 8)
            "xtick.labelsize": 14,      # Tick label size (increased from 7)
            "ytick.labelsize": 14,      # Tick label size (increased from 7)
            "legend.fontsize": 14,      # Legend text size (increased from 7)
           # "figure.figsize": (3.5, 3) # 3.5 inches is roughly 88-90mm (one column)
        })
    
    def plot_save(self, fig):
        """Override plot_save to use non-blocking display"""
        super().plot_save(fig)
        plt.show(block=False)
        plt.pause(0.1)  # Small pause to ensure figure is displayed
        #plt.rcParams['font.family'] = 'Helvetica'
        #plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
        #plt.rcParams.update({
        #    "text.usetex": True,
        #})

    def plot_new(self,df_merge,xvar,yvar,clcvar,xlab,ylab,tit,iplot,huevar=None,colors=None,y_lim=None,iflegend=False,ncol=1,plottype='scatter',jitter_amount=0):
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
        ax1.tick_params(axis='y', colors='k', labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
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
            transform=ax1.transAxes, fontsize=14)
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
        plt.xlabel(xlab)
        plt.title(tit,fontsize=14)
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
                    scatterpoints=1, numpoints=1,fontsize=14)
        abc2plot(ax1, self.dxabc, self.dyabc,
                (self.ifig - 1) * nplotperpage + iplot,
                lower=True, bold=True, usetex=self.usetex, mathrm=True,transform=ax1.transAxes,large=True)
        # Set y-ticks for subfigure 'a' (first subplot)
        if (self.ifig - 1) * nplotperpage + iplot == 1:
            ax1.set_yticks([0.2, 0.3, 0.4])
        # Set y-ticks for subfigure 'b' (second subplot)
        elif (self.ifig - 1) * nplotperpage + iplot == 2:
            ax1.set_yticks([0.2, 0.4, 0.6])
        plt.tight_layout()
        if y_lim is not None:
            plt.setp(ax1, ylim=y_lim)
        # Override y-limits for subfigure 'b' (second subplot) after y_lim is set
        if (self.ifig - 1) * nplotperpage + iplot == 2:
            ax1.set_ylim(0, 0.6) 
        if (iplot == nplotperpage):
        # save one pdf page, zihanlu
            self.plot_save(fig)
            plt.savefig(f"{fout}{figname}{self.ifig}.png", dpi=300, bbox_inches="tight")
            plt.show(block=False)  # Show figure without blocking
            plt.pause(0.1)  # Small pause to ensure figure is displayed
            iplot = 0
        return iplot
if __name__ == '__main__':

    # one model in one figure
    # three figures for each value of the tested variable: GPP,Qle, psi_can
    # show 2015-2016 period, dry and wet year of FR-Hes

    import time as ptime
    import glob
    t1 = ptime.time()

    # Enable interactive mode to show multiple figures simultaneously
    plt.ion()

    self = PlotIt('Scatter Plot')

    suffmod = '/outputs/site_out_cable_*.nc'
    if platform.node()=='zlhp':
        
        fp1='/mnt/d/project/hydraulic/test/'
        fpobs = '/mnt/d/project/hydraulic/test/obs4/'
        fmet = '/mnt/d/project/hydraulic/test/input/met/'

        file_path1 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_BASEdc98fe39.xlsx'
        file_path2 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_drydown_BASEdc98fe39.xlsx'
        file_path3 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_LP_BASEdc98fe39.xlsx'
        file_path4 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_BASEdc98fe39.xlsx'

        # file_path1 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_BASEdc98fe39_LAILBr.xlsx'
        # file_path1 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_BASEdc98fe39_gompertz_LBr.xlsx'
        # file_path1 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_BASEdc98fe39_xcons_LBr.xlsx'
        # #file_path1 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_BASEdc98fe39_xcons_Dsig_LBr.xlsx'
      
        # file_path2 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_drydown_BASEdc98fe39_LAILBr.xlsx'
        # file_path3 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_LP_BASEdc98fe39_LAILBr.xlsx'
        # file_path4 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_LP_drydown_BASEdc98fe39_LAILBr.xlsx'
        # file_path4 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_BASEdc98fe39_xcons_LBr.xlsx'
        fout = '/mnt/d/project/hydraulic/test/plot/fig_final/ppt0129/'

    else:
        fp1='/home/zlu/project/test1105/'
        fpobs='/home/mcuntz/projects/coco2/obs4/'
        fmet = '/home/mcuntz/projects/coco2/cable-pop/input/'

        mapping = {3: 1, 6: 2, 2: 3, 5:4, 7:5, 4:6, 1:7}

    plotorstat = {
        'plotsingle':{'plot':True,'figname':'z03_theta_soiltype_climate_singleplot'},
         'plotall':{'plot':False,'figname':'z03_theta_soiltype_climate_ALLplotV2'},
          'plotallBox':{'plot':False,'figname':'z03_theta_soiltype_climate_ALLplotV2'},
          'stat1':{'plot':False,'figname':'stat_slope0p1_log10_nosand.xlsx'},
          'ifanova':{'plot':False,'figname':'z03_theta_soiltype_climate_ALLplotV2'},
          'multicomp':{'plot':False,'figname':'z03_theta_soiltype_climate_ALLplotV2'},
                  }
   
  
    soil_data = {
    1: {'soiltype':'fine clay','swilt': 0.286,'swilt0':0.286 , 'sfc': 0.367, 'sfc0':0.401 , 'ssat': 0.482,'bch':11.4,'sucs':-0.405,'sand fraction': 0.16},
    2: {'soiltype':'silty clay','swilt': 0.283, 'swilt0':0.277 , 'sfc': 0.370, 'sfc0': 0.401, 'ssat': 0.482,'bch':10.4,'sucs':-0.49,'sand fraction': 0.27},
    3: {'soiltype':'clay loam','swilt': 0.216, 'swilt0':0.219 , 'sfc': 0.301, 'sfc0': 0.376, 'ssat': 0.479,'bch':7.1,'sucs':-0.591,'sand fraction': 0.37},
    4: {'soiltype':'sandy clay','swilt': 0.219,'swilt0':0.219 ,  'sfc': 0.310, 'sfc0': 0.317, 'ssat': 0.426,'bch':10.4,'sucs':-0.153,'sand fraction': 0.52},
    5: {'soiltype':'sandy clay loam','swilt': 0.175,'swilt0': 0.175,  'sfc': 0.255, 'sfc0': 0.300, 'ssat': 0.420,'bch':7.12,'sucs':-0.299,'sand fraction': 0.58},
    6: {'soiltype':'Sandy loam','swilt': 0.135,'swilt0':0.136 ,  'sfc': 0.218,  'sfc0': 0.286,'ssat': 0.443,'bch':5.15,'sucs':-0.348,'sand fraction': 0.60},
    7: {'soiltype': 'sand','swilt': 0.072,'swilt0': 0.070,  'sfc': 0.143,  'sfc0':0.176 ,'ssat': 0.398,'bch':4.2,'sucs':-0.106,'sand fraction': 0.83}
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
    sites = ["UIFC" ,'FR-LBr' ]
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

    df1 = logi_excle(file_path1,False,'_NTD')
    #df1dd = logi_excle(file_path2,True)
    #df1 = pd.concat([df1, df1dd], axis=0)
    #df2 = LP_excle(file_path3,False)
    df2 = LP_excle(file_path1,True,'_EF')
    #df2 = pd.concat([df2, df2dd], axis=0)
    xname_labels = {
     "wb_soilmean": fr'$\theta_{{crit}}$',
     "wb_rwc_soilmean": fr'$REW_{{crit}}^{{\theta}}$',
   # "rwc_soilmean_recal":fr'$REW_{{crit}}$',
     "wb_psi_soilmean": fr'$\psi_{{crit}}^{{\theta}}$',
   # "psi_soilmean": fr'$\psi_{{crit}}$',
    }  
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
        sites = ['FR-LBr']
        figname = plotorstat["plotsingle"]["figname"]+'_LBr'
        self.nrow     = 4
        self.vspace   = 0.08  # y-space between subplots (increased from 0.03)
        self.llxbbox    = 0.75  # x-anchor legend bounding box
        self.llybbox    = 0.99  # y-anchor legend bounding box
        iplot=0
        #sites = ['FR-LBr','AU-ASM','GF-Guy','US-SRM','SE-Nor']
        
        indexnames1=['best']
        indexnames2 =['best'] 
        indexnames1=['MaxCurChange']
        indexnames1=['MaxCurChange','cons0p05','cons0p05Kernel','slope0p05','slope0p1']
        indexnames2 =['BreakPoint','BreakPoint','BreakPoint','BreakPoint','BreakPoint'] 
        indexnames1=['slope0p1']
        indexnames2 =['BreakPoint'] 
        keyvar = ['soiltype','sand fraction']
        iseg1 = 3
        iseg2 = 4
        y_lim = None
        last_key = next(reversed(xname_labels))
        for xname, ylabel in xname_labels.items():
            if xname in ['wb_soilmean']:
                y_lim = (0.1,0.4)
            elif xname in ['wb_rwc_soilmean']:
                y_lim = (0, 1)
            elif xname in ['rwc_soilmean_recal']:
                y_lim = (0, 1)
            elif xname in ['wb_pos_rwc_soilmean']:
                y_lim = (0, 1)
                
            elif xname in ['wb_psi_soilmean','psi_soilmean']:
                y_lim = (-0.7,0)
            else:
                y_lim = (-2,0)
            for (yvar1, val1), (yvar2, val2) in zip(yvarsdf1, yvarsdf2):
                for isite in sites:
                    for idxname1,idxname2 in zip(indexnames1,indexnames2):
                    
                        
                        #for yvar1, yvar1 in zip(yvarsdf1, yvarsdf2):
                            df1_filtered = df1[(df1["site"] == isite) & (df1["Xname"] == xname) & (df1["yvar"] == yvar1) & (df1["method"] == 'percentile') & (df1["segID"] == iseg1) ]
                            df1_filtered_dd = df1[(df1["site"] == isite) & (df1["Xname"] == xname) & (df1["yvar"] == yvar1) & (df1["method"] == 'drydown') & (df1["segID"] == iseg2) ]
                            df2_filtered = df2[(df2["site"] == isite) & (df2["Xname"] == xname) & (df2["yvar"] == yvar2) & (df2["method"] == 'percentile') & (df2["segID"] == iseg1) ]
                            df2_filtered_dd = df2[(df2["site"] == isite) & (df2["Xname"] == xname) & (df2["yvar"] == yvar2) & (df2["method"] == 'drydown') & (df2["segID"] == iseg2) ]
                            df1_filtered=df1_filtered[keyvar + [idxname1]]
                            df1_filtered_dd=df1_filtered_dd[keyvar + [idxname1]]
                            df2_filtered=df2_filtered[keyvar + [idxname2]]
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
                            if xname == last_key:
                                xlab = 'sand fraction'
                            else:
                                xlab= ''
                            iplot=self.plot_new(df_long,'sand fraction','value','dataset',xlab,ylabel,'',iplot,y_lim = y_lim,iflegend=(iplot==0))

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
        figname = plotorstat["plotallBox"]["figname"]
        indexnames=['MaxCur',]#'MaxCurChange','MinCurChange','best']
        idxname1 = 'MaxCur'
        idxname2 = 'MaxCur'
        self.nrow=6
        colors = ["#34119c", "#090df4", "#3172cb", "#54d3f3","#31d5a7","#0aa16f","#96a014","#e970b4","#920a50","#E70C0C" ]  # blue, orange, green
        colors = [c for c in colors for _ in range(2)]
        colors = colors[::-1]
        colors = ["#0aa16f","#e970b4",]
        mrks = ['o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^']
        plotname0 = self.plotname
        keyvar = ['site','soiltype','sand fraction','method','PET_P']
        for xname, xlabel in xname_labels.items():
            if xname in ['wb_soilmean']:
                y_lim = (0.1,0.4)
            elif xname in ['wb_rwc_soilmean']:
                y_lim = (0, 1)
            elif xname in ['rwc_soilmean_recal']:
                y_lim = (0, 1)
            elif xname in ['wb_psi_soilmean']:
                y_lim = (-0.5,0)
            else:
                y_lim = (-2,0)
            for (yvar1, val1), (yvar2, val2) in zip(yvarsdf1, yvarsdf2):
                df1_filtered = df1[(df1["segID"] == 3) & (df1["method"] == 'percentile') & (df1["Xname"] == xname) & (df1["yvar"] == yvar1)]
                df2_filtered_dd = df2[(df2["segID"] == 4) & (df2["method"] == 'drydown') & (df2["Xname"] == xname) & (df2["yvar"] == yvar2)]
                df1_filtered=df1_filtered[keyvar + [idxname1]]
                #df1_filtered_dd=df1_filtered_dd[keyvar + [idxname1]]
                #df2_filtered=df2_filtered[keyvar + [idxname2]]
                df2_filtered_dd=df2_filtered_dd[keyvar + [idxname2]]  
                df1_filtered = df1_filtered.rename(columns={idxname1: 'crit'})    
                df2_filtered_dd = df2_filtered_dd.rename(columns={idxname2: 'crit'})    
                df_merged = pd.concat([df1_filtered, df2_filtered_dd], axis=0)   
                method_mapping = {
                    'percentile': 'NTD',
                    'drydown': 'EF'
                }
                # Apply mapping to the 'method' column
                df_merged['method'] = df_merged['method'].map(method_mapping)
                #df_merged = df1_filtered
                #self.filename = f'{plotname0}_{idxname}_{ylabel}'
                #iplot=self.plot_new(df_merged,'PET_P','crit','soiltype','',xlabel,'',iplot,huevar='method',colors=colors,plottype='box2',y_lim = y_lim,iflegend=(iplot==0),ncol=2)
                iplot=self.plot_new(df_merged,'sand fraction','crit','soiltype','',xlabel,'',iplot,huevar='method',colors=colors,plottype='box2',y_lim = y_lim,iflegend=(iplot==0),ncol=2)
                # df_filtered = df1[(df1["segID"] == 4) & (df1["method"] == 'drydown') & (df1["Xname"] == xname) & (df1["yvar"] == yvar1)]
                # iplot=self.plot_new(df_filtered,'PET_P',indexname,'soiltype','',xlabel,'',iplot,colors=colors,plottype='box',y_lim = y_lim)
                # df_filtered = df2[(df2["segID"] == 3) & (df2["method"] == 'percentile') & (df2["Xname"] == xname) & (df2["yvar"] == yvar2)]
                # iplot=self.plot_new(df_filtered,'PET_P',indexname,'soiltype','',xlabel,'',iplot,colors=colors,plottype='box',y_lim = y_lim)
                #df_filtered = df2[(df2["segID"] == 4) & (df2["method"] == 'drydown') & (df2["Xname"] == xname) & (df2["yvar"] == yvar2)]
                #iplot=self.plot_new(df_filtered,'PET_P',indexname,'soiltype','',xlabel,'',iplot,colors=colors,plottype='box',y_lim = y_lim)
                #plot=self.plot_new(df_merged,'PET_P','crit','soiltype','',xlabel,'',iplot,colors=colors,plottype='box',y_lim = y_lim)
    if plotorstat["ifanova"]["plot"]:
        outname = plotorstat["ifanova"]["figname"]
        df_new = df1[df1['method'] == 'percentile'].copy()
        final_results,final_results2 = anova_func(df_new)
        df_new = df2[df2['method'] == 'drydown'].copy()
        final_resultsdd,final_results2dd = anova_func(df_new)
        if not os.path.exists(outname):
            # Create a new file
            with pd.ExcelWriter(outname, mode="w", engine="openpyxl") as writer:
                final_results.to_excel(writer, sheet_name="Raw_ANOVA_logi", index=False)
                final_results2.to_excel(writer, sheet_name="Relative_SS_logi", index=False)
                final_resultsdd.to_excel(writer, sheet_name="Raw_ANOVA_LP_drydown", index=False)
                final_results2dd.to_excel(writer, sheet_name="Relative_SS_LP_drydown", index=False)
        else:
            with pd.ExcelWriter(outname, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                final_results.to_excel(writer, sheet_name="Raw_ANOVA_logi", index=False)
                final_results2.to_excel(writer, sheet_name="Relative_SS_logi", index=False)
                final_resultsdd.to_excel(writer, sheet_name="Raw_ANOVA_LP_drydown", index=False)
                final_results2dd.to_excel(writer, sheet_name="Relative_SS_LP_drydown", index=False)

        # df_new = df1[df1['method'] == 'drydown'].copy()
        # final_results,final_results2 = anova_func(df_new)
        # with pd.ExcelWriter(outname, mode="a", if_sheet_exists="replace") as writer:
        #     final_results.to_excel(writer, sheet_name="Raw_ANOVA_logi_drydown", index=False)
        #     final_results2.to_excel(writer, sheet_name="Relative_SS_logi_drydown", index=False)

        # df_new = df2[df2['method'] == 'percentile'].copy()
        # final_results,final_results2 = anova_func(df_new)
  
        # with pd.ExcelWriter(outname, mode="a", if_sheet_exists="replace") as writer:
        #     final_results.to_excel(writer, sheet_name="Raw_ANOVA_LP", index=False)
        #     final_results2.to_excel(writer, sheet_name="Relative_SS_LP", index=False)

        # write to Excel with two sheets
    if plotorstat["stat1"]["plot"]:
        outname = plotorstat["stat1"]["figname"]
        #sites = ['FR-LBr']
        indexnames1=['best']
        indexnames2 =['best'] 
        indexnames1=['slope0p1']
        indexnames2 =['BreakPoint'] 
        keyvar = ['site','soiltype','sand fraction','method','PET_P']
        iseg1 = 3
        iseg2 = 4
        results = []
        results1 = []
        for xname, xlabel in xname_labels.items():

            methods = {'NTD':['percentile'],'EF':['drydown'],'all':['percentile','drydown']}
            for (yvar1, val1), (yvar2, val2) in zip(yvarsdf1, yvarsdf2):
      
                for isite in sites:
                    for idxname1,idxname2 in zip(indexnames1,indexnames2):
                            if isinstance(isite, list):
                                df1_filtered = df1[(df1["site"].isin(isite)) & (df1["Xname"] == xname) & (df1["yvar"] == yvar1) & (df1["method"] == 'percentile') & (df1["segID"] == iseg1) ]
                                #df1_filtered_dd = df1[(df1["site"] == isite) & (df1["Xname"] == xname) & (df1["yvar"] == yvar1) & (df1["method"] == 'drydown') & (df1["segID"] == iseg2) ]
                                #df2_filtered = df2[(df2["site"] == isite) & (df2["Xname"] == xname) & (df2["yvar"] == yvar2) & (df2["method"] == 'percentile') & (df2["segID"] == iseg1) ]
                                df2_filtered_dd = df2[(df2["site"].isin(isite)) & (df2["Xname"] == xname) & (df2["yvar"] == yvar2) & (df2["method"] == 'drydown') & (df2["segID"] == iseg2) ]
                                site_label = "all_sites"
                            else:
                                df1_filtered = df1[(df1["site"]== isite) & (df1["Xname"] == xname) & (df1["yvar"] == yvar1) & (df1["method"] == 'percentile') & (df1["segID"] == iseg1) ]
                                #df1_filtered_dd = df1[(df1["site"] == isite) & (df1["Xname"] == xname) & (df1["yvar"] == yvar1) & (df1["method"] == 'drydown') & (df1["segID"] == iseg2) ]
                                #df2_filtered = df2[(df2["site"] == isite) & (df2["Xname"] == xname) & (df2["yvar"] == yvar2) & (df2["method"] == 'percentile') & (df2["segID"] == iseg1) ]
                                df2_filtered_dd = df2[(df2["site"]== isite) & (df2["Xname"] == xname) & (df2["yvar"] == yvar2) & (df2["method"] == 'drydown') & (df2["segID"] == iseg2) ]
                                site_label = isite
                            df1_filtered=df1_filtered[keyvar + [idxname1]]
                            #df1_filtered_dd=df1_filtered_dd[keyvar + [idxname1]]
                            #df2_filtered=df2_filtered[keyvar + [idxname2]]
                            df2_filtered_dd=df2_filtered_dd[keyvar + [idxname2]]  
                            df1_filtered = df1_filtered.rename(columns={idxname1: 'crit'})    
                            df2_filtered_dd = df2_filtered_dd.rename(columns={idxname2: 'crit'})    
                            df_merged = pd.concat([df1_filtered, df2_filtered_dd], axis=0,ignore_index=True)                     
                            for methodtag,imethod in methods.items():
                                dftmp = df_merged[df_merged["method"].isin(imethod)]
                                dftmp = dftmp.dropna(subset=["crit"])
                                n_valid  = dftmp["crit"].count()
                                min_val = max_val = mean_val = std_val = r = p = r1 = p1 = np.nan
                                min_soiltype = max_soiltype = min_site = max_site = None
                                if n_valid>0:
                                    min_val = dftmp["crit"].min()
                                    max_val = dftmp["crit"].max()
                                    mean_val = dftmp["crit"].mean()
                                    std_val  = dftmp["crit"].std()
                                    
                                    # find corresponding soiltypes
                                    min_row = dftmp.loc[dftmp["crit"].idxmin()]
                                    max_row = dftmp.loc[dftmp["crit"].idxmax()]
                                    min_soiltype = min_row["soiltype"]
                                    max_soiltype = max_row["soiltype"]
                                    min_site = min_row["site"]
                                    max_site = max_row["site"]
                                    if n_valid>5:
                                        dftmptmp = dftmp.copy()
                                        dftmptmp = dftmp[dftmp['soiltype'] != 'sand']
                                        if 'psi' in xname:
                                            r, p = pearsonr(np.log10(-dftmptmp["crit"]), dftmptmp["sand fraction"])
                                        else:
                                            r, p = pearsonr(dftmptmp["crit"], dftmptmp["sand fraction"])
                                        if isinstance(isite, list):
                                            r1, p1 = pearsonr(dftmp["crit"], dftmp["PET_P"])

                                results.append({
                                "xname":xname,
                                "site": site_label,
                                "method": methodtag,
                                'Nvalid':n_valid,
                                "mean": mean_val,
                                "std": std_val,                                
                                "min": min_val,
                                "min_soiltype": min_soiltype,
                                "min_site": min_site,
                                "max": max_val,
                                "max_soiltype": max_soiltype,
                                "max_site": max_site,
                                "r_sand": r,
                                "p_sand": p,
                                "r_PET_P": r1,
                                "p_PET_P": p1,  
                            })
                            df_pivot = df_merged.pivot_table(index=['site','soiltype','sand fraction','PET_P'], 
                            columns="method", 
                            values="crit").reset_index()
                            min_val = max_val = mean_val = std_val = r = p = r1 = p1 = np.nan
                            min_soiltype = max_soiltype = min_site = max_site = None
                            if ('percentile' in df_pivot.columns) and ('drydown' in df_pivot.columns):
                                # ensure both methods exist in each row
                                df_pivot = df_pivot.dropna(subset=["percentile", "drydown"])

                                # calculate RD between the two methods
                                df_pivot["RD"] = np.abs(df_pivot["percentile"] - df_pivot["drydown"]) / (
                                    (df_pivot["percentile"] + df_pivot["drydown"]) / 2
                                )
                                n_valid  = df_pivot["RD"].count()
                                if n_valid>0:
                                    min_val = df_pivot["RD"].min()
                                    max_val = df_pivot["RD"].max()
                                    mean_val = df_pivot["RD"].mean()
                                    std_val  = df_pivot["RD"].std()
                                    
                                    # find corresponding soiltypes
                                    min_row = df_pivot.loc[df_pivot["RD"].idxmin()]
                                    max_row = df_pivot.loc[df_pivot["RD"].idxmax()]
                                    min_soiltype = min_row["soiltype"]
                                    max_soiltype = max_row["soiltype"]
                                    min_site = min_row["site"]
                                    max_site = max_row["site"]
                                    if n_valid>5:
                                        r, p = pearsonr(df_pivot["RD"], df_pivot["sand fraction"])
                                        if isinstance(isite, list):
                                            r1, p1 = pearsonr(df_pivot["RD"], df_pivot["PET_P"])
                                
                            results1.append({
                            "xname":xname,
                            "site": site_label,
                            "method": 'RD of two methods',
                            'Nvalid':n_valid,
                            "mean": mean_val,
                            "std": std_val,                                
                            "min": min_val,
                            "min_soiltype": min_soiltype,
                            "min_site": min_site,
                            "max": max_val,
                            "max_soiltype": max_soiltype,
                            "max_site": max_site,
                            "r_sand": r,
                            "p_sand": p,
                            "r_PET_P": r1,
                            "p_PET_P": p1,                            
                        })
        results = pd.DataFrame(results)
        results1 = pd.DataFrame(results1)
        if not os.path.exists(f'{fout}{outname}'):
            # Create a new file
            with pd.ExcelWriter(f'{fout}{outname}', mode="w", engine="openpyxl") as writer:
                results.to_excel(writer, sheet_name="crit", index=False)
                results1.to_excel(writer, sheet_name="RelativeDifference_crit", index=False)
        else:
            with pd.ExcelWriter(f'{fout}{outname}', mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                results.to_excel(writer, sheet_name="crit", index=False)
                results1.to_excel(writer, sheet_name="RelativeDifference_crit", index=False)
    if plotorstat["multicomp"]["plot"]:
        outname = plotorstat["multicomp"]["figname"]
        indexnames1=['best']
        indexnames2 =['best'] 
        indexnames1=['MaxCur']
        indexnames2 =['MaxCur'] 
        keyvar = ['site','soiltype','sand fraction','method','PET_P']
        iseg1 = 3
        iseg2 = 4

        for xname, xlabel in xname_labels.items():
            anova_results = []
            tukey_results = []
            methods = {'NTD':['percentile'],'EF':['drydown'],'all':['percentile','drydown']}
            for (yvar1, val1), (yvar2, val2) in zip(yvarsdf1, yvarsdf2):
                for idxname1,idxname2 in zip(indexnames1,indexnames2):
                    df1_filtered = df1[ (df1["Xname"] == xname) & (df1["yvar"] == yvar1) & (df1["method"] == 'percentile') & (df1["segID"] == iseg1) ]
                    #df1_filtered_dd = df1[(df1["site"] == isite) & (df1["Xname"] == xname) & (df1["yvar"] == yvar1) & (df1["method"] == 'drydown') & (df1["segID"] == iseg2) ]
                    #df2_filtered = df2[(df2["site"] == isite) & (df2["Xname"] == xname) & (df2["yvar"] == yvar2) & (df2["method"] == 'percentile') & (df2["segID"] == iseg1) ]
                    df2_filtered_dd = df2[ (df2["Xname"] == xname) & (df2["yvar"] == yvar2) & (df2["method"] == 'drydown') & (df2["segID"] == iseg2) ]

                    df1_filtered=df1_filtered[keyvar + [idxname1]]
                    #df1_filtered_dd=df1_filtered_dd[keyvar + [idxname1]]
                    #df2_filtered=df2_filtered[keyvar + [idxname2]]
                    df2_filtered_dd=df2_filtered_dd[keyvar + [idxname2]]  
                    df1_filtered = df1_filtered.rename(columns={idxname1: 'crit'})    
                    df2_filtered_dd = df2_filtered_dd.rename(columns={idxname2: 'crit'})    
                    df_merged = pd.concat([df1_filtered, df2_filtered_dd], axis=0)                     
                    for methodtag,imethod in methods.items():
                        dftmp = df_merged[df_merged["method"].isin(imethod)]
                        dfm = dftmp.dropna(subset=["crit"])
                        # ANOVA for site
                        model_site = ols('crit ~ C(site)', data=dfm).fit()
                        anova_site = sm.stats.anova_lm(model_site, typ=2)
                        anova_site['factor'] = 'site'
                        anova_site['method'] = methodtag
                        anova_results.append(anova_site)
                        
                        # ANOVA for soiltype
                        model_soil = ols('crit ~ C(soiltype)', data=dfm).fit()
                        anova_soil = sm.stats.anova_lm(model_soil, typ=2)
                        anova_soil['factor'] = 'soiltype'
                        anova_soil['method'] = methodtag
                        anova_results.append(anova_soil)
                        
                        # Tukey HSD for site
                        tukey_site = pairwise_tukeyhsd(endog=dfm['crit'], groups=dfm['site'], alpha=0.05)
                        tukey_df_site = pd.DataFrame(data=tukey_site._results_table.data[1:], columns=tukey_site._results_table.data[0])
                        tukey_df_site['method'] = methodtag
                        tukey_df_site['factor'] = 'site'
                        tukey_results.append(tukey_df_site)

                        # Tukey HSD for soiltype
                        tukey_soil = pairwise_tukeyhsd(endog=dfm['crit'], groups=dfm['soiltype'], alpha=0.05)
                        tukey_df_soil = pd.DataFrame(data=tukey_soil._results_table.data[1:], columns=tukey_soil._results_table.data[0])
                        tukey_df_soil['method'] = methodtag
                        tukey_df_soil['factor'] = 'soiltype'
                        tukey_results.append(tukey_df_soil)
                    df_anova = pd.concat(anova_results)
                    df_tukey = pd.concat(tukey_results)
            if not os.path.exists(outname_multicomp):
                # Create a new file
                with pd.ExcelWriter(outname_multicomp, mode="w", engine="openpyxl") as writer:
                    df_anova.to_excel(writer, sheet_name=f"ANOVA_{xname}", index=False)
                    df_tukey.to_excel(writer, sheet_name=f"Tukey_HSD_{xname}", index=False)
            else:
                with pd.ExcelWriter(outname_multicomp, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                    df_anova.to_excel(writer, sheet_name=f"ANOVA_{xname}", index=False)
                    df_tukey.to_excel(writer, sheet_name=f"Tukey_HSD_{xname}", index=False)
    if iplot>0:
        self.plot_save(self.fig)
        plt.savefig(f"{fout}{figname}{self.ifig}.png", dpi=300, bbox_inches="tight")
        plt.show(block=False)  # Show figure without blocking
        plt.pause(0.1)  # Small pause to ensure figure is displayed

    self.close()

    t2    = ptime.time()
    strin = ( '[m]: {:.1f}'.format((t2 - t1) / 60.)
              if (t2 - t1) > 60.
              else '[s]: {:d}'.format(int(t2 - t1)) )
    print('    Time elapsed', strin)