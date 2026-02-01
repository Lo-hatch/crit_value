
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
import statsmodels.formula.api as smf
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
    "wb_soilmean_rwc": fr'$REW_{{crit}}^{{\theta}}$',
    "rwc_soilmean_recal":fr'$REW_{{crit}}$',
    "wb_soilmean_psi": fr'$\psi_{{crit}}^{{\theta}}$',
    "psi_soilmean": fr'$\psi_{{crit}}$',
}  
    indexnames=['MaxCur','MaxCurChange','cons0p05','slope0p1','BreakPoint']
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
                            model = ols(f"{idxname} ~ C(soiltype) + C(ID)", data=subset).fit()
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
                            rel_sites = anova_res.loc["C(ID)", "sum_sq"] / total_ss if "C(ID)" in anova_res.index else None
                            
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
def anova2_func(df_new):
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
    indexnames=['MaxCur','MaxCurChange','BreakPoint','best']
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
                            
                            # model = ols(f'{idxname} ~ C(soiltype) * C(site) * C(ID)', data=subset).fit()
                            # model = ols(f'{idxname} ~ C(soiltype) * C(site) + C(soiltype) * C(ID) + C(site) * C(ID)', data=subset).fit()
                            model = smf.ols(f"{idxname} ~ C(soiltype) + C(site) + C(ID)", data=subset).fit()

                            anova_res = sm.stats.anova_lm(model, typ=2)
                            res_df = anova_res.reset_index().rename(columns={"index": "Source"})
                            # compute relative variance contributions
                            total_ss = anova_res["sum_sq"].sum()

                            rel_soil = anova_res.loc["C(soiltype)", "sum_sq"] / total_ss if "C(soiltype)" in anova_res.index else 0
                            rel_site = anova_res.loc["C(site)", "sum_sq"] / total_ss if "C(site)" in anova_res.index else 0
                            rel_hydraulic = anova_res.loc["C(ID)", "sum_sq"] / total_ss if "C(ID)" in anova_res.index else 0
                            rel_resid = anova_res.loc["Residual", "sum_sq"] / total_ss
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

                            rel_df = pd.DataFrame([{
                                "Relative_soiltype": rel_soil,
                                "Relative_site": rel_site,
                                "Relative_hydraulic": rel_hydraulic,
                                "Relative_residual": rel_resid,
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
def mix_func(df_new):
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
    indexnames=['MaxCur','MaxCurChange','BreakPoint','best']
    results_list = []  # to collect all results
    for iseg in [0,1,2,3,4]:
        if iseg<4:
            seglab = f'seg{iseg+1}'
        else:
            seglab = 'whole'
        for xname, ylabel in xname_labels.items():
            for yvar, yvarlabel in yvar_labels.items():
                for idxname in indexnames:
                    # subset rows for this combination
                    if idxname in df_new.columns:
                        df = df_new[(df_new["segID"] == iseg) & (df_new["Xname"] == xname) & (df_new["yvar"] == yvar)]
                    else:
                        continue
                    df = df.dropna(subset=[idxname, "site", "soiltype", "ID"])
                    if df.shape[0] > 350:  # only if data exists
                        
                        models = [
                            smf.mixedlm(f"{idxname} ~ C(site) + C(soiltype)", data=df, groups=df["ID"]),
                            smf.mixedlm(f"{idxname} ~ C(site) + C(soiltype)", data=df, groups=df["ID"], re_formula="~C(soiltype)+C(site)")
                        ]
                        model_names = ["random_intercept_only", "random_intercept_slopes"]
                        for name, model in zip(model_names, models):
                            result = model.fit()


                            result = model.fit()
                            # print(result1.aic, result1.bic)
                            # print(result2.aic, result2.bic)
                            # LR = 2 * (result2.llf - result1.llf)

                            # # degrees of freedom difference
                            # df_diff = result2.df_modelwc - result1.df_modelwc

                            # # p-value from chi-square distribution
                            # p_value = chi2.sf(LR, df_diff)

                            # print(f"Likelihood Ratio = {LR:.3f}, df = {df_diff}, p = {p_value:.4f}")
                
                            var_random = sum(result.cov_re.values.diagonal()) # random intercept variance (strategy)
                            var_resid = result.scale                 # residual variance
                            # Based on Nakagawa & Schielzeth (2013)
                            var_fix = np.var(result.fittedvalues)
                            fitted = result.fittedvalues  # predicted crit including random effects
                            resid = df[idxname] - fitted
                            rmse = np.sqrt(np.mean(resid**2))
                            var_total = (var_fix + var_random + var_resid)
                            r2_marginal = var_fix / var_total
                            r2_conditional = (var_fix + var_random) / var_total
                            r2_random = r2_conditional - r2_marginal
                            #print(f"R2 of fix and random = {r2_marginal:.3f}, {r2_random:.3f}")
                            # Fixed effects contribution for soiltype only
                            import patsy
                            X_soil = patsy.dmatrix("C(soiltype)", df, return_type='dataframe')

                            # Select only the coefficients for soiltype from the fitted model
                            soil_cols = [col for col in result.fe_params.index if 'soiltype' in col]

                            # Compute contribution of soiltype to fitted values
                            fitted_soiltype = X_soil[soil_cols] @ result.fe_params[soil_cols]

                            # Variance explained by soiltype
                            var_soil = np.var(fitted_soiltype, ddof=1)
                            X_site = patsy.dmatrix("C(site)", df, return_type='dataframe')
                            site_cols = [col for col in result.fe_params.index if 'site' in col]
                            fitted_site = X_site[site_cols] @ result.fe_params[site_cols]

                            var_site = np.var(fitted_site, ddof=1)


                            var_fix2 = var_soil + var_site 
                            # print("var fix:", var_fix)
                            # print("var fix2:", var_fix2)
                            # print("Soiltype proportion:", var_soil/var_total)
                            # print("Site proportion:", var_site/var_total)
                            rel_df = pd.DataFrame([{
                                "Relative_soiltype": var_soil/var_fix2*var_fix/var_total,
                                "Relative_site": var_site/var_fix2*var_fix/var_total,
                                "relative_random":r2_random,
                                'relative_resid': var_resid/var_total,
                                'R2_marginal': r2_marginal,
                                'R2_conditional': r2_conditional,
                                'AIC': result.aic,
                                'BIC': result.bic,
                                'model':name,
                                "segID":seglab,
                                "yvar": yvar,
                                "xname": xname,
                                "indexnames": idxname
                            }])
                            results_list.append(rel_df)          
    results_list = pd.concat(results_list, ignore_index=True)
    return results_list    
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
        df.loc[df["R2_LP"] < df["R2_PLP"], "BreakPoint"] = df["BreakPoint_PLP"]
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
        self.nrow     = 5     # # of rows of subplots per figure
        self.ncol     = 2    # # of columns of subplots per figure
        self.hspace   = 0.07  # x-space between subplots
        self.vspace   = 0.08  # y-space between subplots
        self.textsize = 10    # standard text size
        self.dxabc    = 0.02  # % of (max-min) shift to the right
                              # of left y-axis for a,b,c,... labels
        self.dyabc    = 1.06  # % of (max-min) shift up from lower x-axis
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

        plt.rcParams['axes.labelsize'] = 8  # Font size for x and y axis labels
        plt.rcParams['xtick.labelsize'] = 7  # Set x-axis tick label size
        plt.rcParams['ytick.labelsize'] = 7  # Set y-axis tick label size
        plt.rcParams['axes.titlesize'] = 9  # Font size for plot title
        #plt.rcParams['font.family'] = 'Helvetica'
        #plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
        #plt.rcParams.update({
        #    "text.usetex": True,
        #})

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
        # Custom color list (3 colors for 3 yvar values)
        if colors is None:
            colors = ["#34119c", "#ff7f0e", "#2ca02c"]  # blue, orange, green
        if plottype=='scatter':
            sns.scatterplot(
                data=df_merge,
                x=xvar,
                y=yvar,
                hue=clcvar,
                style=clcvar,  
                alpha=0.6,
                palette=colors,  # List of 3 colors
                ax=ax1,
                s=100,legend=iflegend)
        elif plottype=='box1':
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
                palette=colors,
            )
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
        if iflegend:
            ll = ax1.legend(frameon=self.frameon, ncol=ncol,
                    labelspacing=self.llrspace,
                    handletextpad=self.llhtextpad,
                    handlelength=self.llhlength,
                    loc='upper left',
                    bbox_to_anchor=(self.llxbbox,self.llybbox),
                    scatterpoints=1, numpoints=1,fontsize=5.5)
      
        plt.tight_layout()
        if y_lim is not None:
            plt.setp(ax1, ylim=y_lim) 
        if (iplot == nplotperpage):
        # save one pdf page, zihanlu
            self.plot_save(fig)
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

        file_path1 = '/mnt/d/project/hydraulic/test/plot/f05_test_LHS/result_BASEdc98fe39_LHS_simple_new.xlsx'
        #file_path2 = '/mnt/d/project/hydraulic/test/plot/f05_test_LHS/result_drydown_BASEdc98fe39_LAILBr.xlsx'
        #file_path3 = '/mnt/d/project/hydraulic/test/plot/f05_test_LHS/result_LP_BASEdc98fe39_LAILBr.xlsx'
        file_path4 = '/mnt/d/project/hydraulic/test/plot/f05_test_LHS/result_BASEdc98fe39_LHS_simple_new.xlsx'
        fout = '/mnt/d/project/hydraulic/test/plot/fig_final/v20251111/'
        
    else:
        fp1='/home/zlu/project/test1105/'
        fpobs='/home/mcuntz/projects/coco2/obs4/'
        fmet = '/home/mcuntz/projects/coco2/cable-pop/input/'

        mapping = {3: 1, 6: 2, 2: 3, 5:4, 7:5, 4:6, 1:7}
    plotorstat = {
        'plotsingle':{'plot':False,'figname':'z03_SI_diffwb'},
         'plotall':{'plot':False,'figname':'z03_theta_soiltype_climate_ALLplotV2'},
          'plotallBox':{'plot':False,'figname':'z05_theta_soil_parameter'},
          'stat1':{'plot':False,'figname':'stat_slope0p1_log10_nosand.xlsx'},
          'ifanova':{'plot':True,'figname':'anova_soil_ID.xlsx'},
          'ifanova2':{'plot':False,'figname':'z03_theta_soiltype_climate_ALLplotV2'},
          'mix':{'plot':False,'figname':'z03_theta_soiltype_climate_ALLplotV2'},
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
    sites = ['US-SRM','FR-LBr','GF-Guy']
    sites = ['UIFC']
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
    df1 = df1[~df1['ID'].isin([1,3,5,9,21])]
    #df1dd = logi_excle(file_path2,True)
    #df1 = pd.concat([df1, df1dd], axis=0)
    #df2 = LP_excle(file_path3,False)
    df2 = LP_excle(file_path4,'_EF')
    df2 = df2[~df2['ID'].isin([1,3,5,9,21])]
    #df2 = pd.concat([df2, df2dd], axis=0)
    xname_labels = {
        "wb_soilmean": fr'$\theta_{{crit}}$',
        "wb_soilmean_rwc": fr'$REW_{{crit}}^{{\theta}}$',
       "rwc_soilmean_recal":fr'$REW_{{crit}}$',
       "wb_soilmean_psi": fr'$\psi_{{crit}}^{{\theta}}$',
      "psi_soilmean": fr'$\psi_{{crit}}$',
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
        figname = plotorstat["plotsingle"]["figname"]
        self.llxbbox    = 0.75  # x-anchor legend bounding box
        self.llybbox    = 0.99  # y-anchor legend bounding box
        iplot=0
        
        indexnames1=['MaxCur','MaxCur','MaxCurChange','MaxCur','MaxCurChange',]
        indexnames=['MaxCur']
        indexnames2 =['MaxCur','MaxCurChange','MaxCurChange','BreakPoint','BreakPoint'] 
        keyvar = ['soiltype','sand fraction']
        iseg1 = 3
        iseg2 = 4
        y_lim = None
        for xname, xlabel in xname_labels.items():
            if xname in ['wb_soilmean']:
                y_lim = (0.1,0.4)
            elif xname in ['wb_soilmean_rwc']:
                y_lim = (0, 1)
            elif xname in ['rwc_soilmean_recal']:
                y_lim = (0, 1)
            elif xname in ['wb_soilmean_psi']:
                y_lim = (-0.5,0)
            else:
                y_lim = (-2,0)
            for (yvar1, val1), (yvar2, val2) in zip(yvarsdf1, yvarsdf2):
                for idxname1,idxname2 in zip(indexnames1,indexnames2):
                    
                        for isite in sites:
                        #for yvar1, yvar1 in zip(yvarsdf1, yvarsdf2):
                            df1_filtered = df1[(df1["site"] == isite) & (df1["Xname"] == xname) & (df1["yvar"] == yvar1) & (df1["method"] == 'percentile') & (df1["segID"] == iseg1) ]
                            df1_filtered_dd = df1[(df1["site"] == isite) & (df1["Xname"] == xname) & (df1["yvar"] == yvar1) & (df1["method"] == 'drydown') & (df1["segID"] == iseg2) ]
                            df2_filtered = df2[(df2["site"] == isite) & (df2["Xname"] == xname) & (df2["yvar"] == yvar2) & (df2["method"] == 'percentile') & (df2["segID"] == iseg1) ]
                            df2_filtered_dd = df2[(df2["site"] == isite) & (df2["Xname"] == xname) & (df2["yvar"] == yvar2) & (df2["method"] == 'drydown') & (df2["segID"] == iseg2) ]
                            df1_filtered=df1_filtered[keyvar + [idxname1]]
                            df1_filtered_dd=df1_filtered_dd[keyvar + [idxname1]]
                            df2_filtered=df2_filtered[keyvar + [idxname2]]
                            df2_filtered_dd=df2_filtered_dd[keyvar + [idxname2]]

                            df1_filtered_dd = df1_filtered_dd.rename(columns={idxname1: f'DD_{idxname1}'})
                            df2_filtered = df2_filtered.rename(columns={idxname2: f'{val2} {idxname2}'})
                            df2_filtered_dd = df2_filtered_dd.rename(columns={idxname2: f'{val2} DD_{idxname2}'})
                            df_merged = df1_filtered.merge(df1_filtered_dd, on=keyvar).merge(df2_filtered, on=keyvar).merge(df2_filtered_dd, on=keyvar)
                            df_long = df_merged.melt(id_vars=keyvar, var_name='dataset', value_name='value')
                            iplot=self.plot_new(df_long,'sand fraction','value','dataset','',xlabel,'',iplot,y_lim = y_lim,iflegend=True)

    if plotorstat["plotall"]["plot"]:
        figname = plotorstat["plotall"]["figname"]
        self.llxbbox    = 0.02  # x-anchor legend bounding box
        self.llybbox    = 1.5  # y-anchor legend bounding box

        self.filename =''
        indexnames=['MaxCur',]#'MaxCurChange','MinCurChange','best']

        colors = ["#34119c", "#090df4", "#3172cb", "#54d3f3","#31d5a7","#0aa16f","#96a014","#e970b4","#920a50","#E70C0C" ]  # blue, orange, green
        colors = [c for c in colors for _ in range(2)]
        colors = colors[::-1]
        mrks = ['o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^']
        plotname0 = self.plotname
        indexname = 'MaxCurChange'
        for xname, xlabel in xname_labels.items():
            if xname in ['wb_soilmean']:
                y_lim = (0.1,0.4)
            elif xname in ['wb_soilmean_rwc']:
                y_lim = (0, 1)
            elif xname in ['rwc_soilmean_recal']:
                y_lim = (0, 1)
            elif xname in ['wb_soilmean_psi']:
                y_lim = (-0.5,0)
            else:
                y_lim = (-2,0)
            for (yvar1, val1), (yvar2, val2) in zip(yvarsdf1, yvarsdf2):
                   
                        #self.filename = f'{plotname0}_{idxname}_{ylabel}'
                        df_filtered = df1[(df1["segID"] == 3) & (df1["method"] == 'percentile') & (df1["Xname"] == xname) & (df1["yvar"] == yvar1)]
                        iplot=self.plot_new(df_filtered,'soiltype',indexname,'site','',xlabel,'',iplot,colors=colors,iflegend=True,ncol=10,y_lim = y_lim)
                        df_filtered = df1[(df1["segID"] == 4) & (df1["method"] == 'drydown') & (df1["Xname"] == xname) & (df1["yvar"] == yvar1)]
                        iplot=self.plot_new(df_filtered,'soiltype',indexname,'site','',xlabel,'',iplot,colors=colors,y_lim = y_lim)
                        df_filtered = df2[(df2["segID"] == 3) & (df2["method"] == 'percentile') & (df2["Xname"] == xname) & (df2["yvar"] == yvar2)]
                        iplot=self.plot_new(df_filtered,'soiltype',indexname,'site','',xlabel,'',iplot,colors=colors,y_lim = y_lim)
                        df_filtered = df2[(df2["segID"] == 4) & (df2["method"] == 'drydown') & (df2["Xname"] == xname) & (df2["yvar"] == yvar2)]
                        iplot=self.plot_new(df_filtered,'soiltype',indexname,'site','',xlabel,'',iplot,colors=colors,y_lim = y_lim)
                        iplot=self.plot_new(df_filtered,'soiltype','BreakPoint','site','',xlabel,'',iplot,colors=colors,y_lim = y_lim)
    if plotorstat["plotallBox"]["plot"]:
        sites = ['UIFC']
        figname = plotorstat["plotallBox"]["figname"]
     
        indexnames=['MaxCur',]#'MaxCurChange','MinCurChange','best']
        indexname1 = 'slope0p1'
        indexname2 = 'BreakPoint'
        colors = ["#34119c", "#090df4", "#3172cb", "#54d3f3","#31d5a7","#0aa16f","#96a014","#e970b4","#920a50","#E70C0C" ]  # blue, orange, green
        colors = [c for c in colors for _ in range(2)]
        colors = colors[::-1]
        colors = ["#3172cb","#E70C0C"]
        mrks = ['o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^','o','^']
        plotname0 = self.plotname
        
        for xname, xlabel in xname_labels.items():
            if xname in ['wb_soilmean']:
                y_lim = (0.1,0.4)
            elif xname in ['wb_soilmean_rwc']:
                y_lim = (0, 1)
            elif xname in ['rwc_soilmean_recal']:
                y_lim = (0, 1)
            elif xname in ['wb_soilmean_psi']:
                y_lim = (-5,0)
            else:
                y_lim = (-5,0)
            for (yvar1, val1), (yvar2, val2) in zip(yvarsdf1, yvarsdf2):
                df_filtered = df1[(df1["segID"] == 3) & (df1["Xname"] == xname) & (df1["yvar"] == yvar1)]
                #self.filename = f'{plotname0}_{idxname}_{ylabel}'
                iplot=self.plot_new(df_filtered,'soiltype',indexname1,'ID','',xlabel,'',iplot,colors=colors,plottype='box2',y_lim = y_lim,huevar='site')

                # df_filtered = df2[(df2["segID"] == 4) & (df2["Xname"] == xname) & (df2["yvar"] == yvar2)]
                # iplot=self.plot_new(df_filtered,'soiltype',indexname2,'ID','',xlabel,'',iplot,colors=colors,plottype='box2',y_lim = y_lim,huevar='site')
                #iplot=self.plot_new(df_filtered,'PET_P','BreakPoint','soiltype','',xlabel,'',iplot,colors=colors,plottype='box',y_lim = y_lim)
    if plotorstat["ifanova"]["plot"]:
        outname = plotorstat["ifanova"]["figname"]
        
        df_new = df1.copy()
        final_results,final_results2 = anova_func(df_new)
        if not os.path.exists(f'{fout}{outname}'):
            with pd.ExcelWriter(f'{fout}{outname}', mode="w") as writer:
                final_results.to_excel(writer, sheet_name="Raw_ANOVA_NTD", index=False)
                final_results2.to_excel(writer, sheet_name="Relative_SS_NTD", index=False)
        else:
            with pd.ExcelWriter(f'{fout}{outname}', mode="a", if_sheet_exists="replace") as writer:
                final_results.to_excel(writer, sheet_name="Raw_ANOVA_NTD", index=False)
                final_results2.to_excel(writer, sheet_name="Relative_SS_NTD", index=False)
        df_new = df2.copy()
        final_results,final_results2 = anova_func(df_new)
        if not os.path.exists(f'{fout}{outname}'):
            with pd.ExcelWriter(f'{fout}{outname}', mode="w") as writer:
                final_results.to_excel(writer, sheet_name="Raw_ANOVA_EF", index=False)
                final_results2.to_excel(writer, sheet_name="Relative_SS_EF", index=False)
        else:
            with pd.ExcelWriter(f'{fout}{outname}', mode="a", if_sheet_exists="replace") as writer:
                final_results.to_excel(writer, sheet_name="Raw_ANOVA_EF", index=False)
                final_results2.to_excel(writer, sheet_name="Relative_SS_EF", index=False)


    if plotorstat["ifanova2"]["plot"]:
        df_new = df1[df1['method'] == 'percentile'].copy()
        final_results,final_results2 = anova2_func(df_new)
        with pd.ExcelWriter(outname, mode="a", if_sheet_exists="replace") as writer:
            final_results.to_excel(writer, sheet_name="Raw_ANOVA_logi", index=False)
            final_results2.to_excel(writer, sheet_name="Relative_SS_logi", index=False)
        df_new = df2[df2['method'] == 'drydown'].copy()
        final_results,final_results2 = anova2_func(df_new)
        with pd.ExcelWriter(outname, mode="a", if_sheet_exists="replace") as writer:
            final_results.to_excel(writer, sheet_name="Raw_ANOVA_LP_drydown", index=False)
            final_results2.to_excel(writer, sheet_name="Relative_SS_LP_drydown", index=False)
    if plotorstat["mix"]["plot"]:
        import statsmodels.formula.api as smf
        from scipy.stats import chi2
        df_new = df1[df1['method'] == 'percentile'].copy()
        final_results = mix_func(df_new)
        df_new = df2[df2['method'] == 'drydown'].copy()
        final_results2 = mix_func(df_new)
        if not os.path.exists(outname_mix):
            # Create a new file
            with pd.ExcelWriter(outname_mix, mode="w", engine="openpyxl") as writer:
                final_results.to_excel(writer, sheet_name="logi", index=False)
                final_results2.to_excel(writer, sheet_name="LP", index=False)
        else:
            with pd.ExcelWriter(outname_mix, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                final_results.to_excel(writer, sheet_name="logi", index=False)
                final_results2.to_excel(writer, sheet_name="LP", index=False)



    if iplot>0:
        self.plot_save(self.fig)
        plt.savefig(f"{fout}{figname}{self.ifig}.png", dpi=300, bbox_inches="tight")


    self.close()


    t2    = ptime.time()
    strin = ( '[m]: {:.1f}'.format((t2 - t1) / 60.)
              if (t2 - t1) > 60.
              else '[s]: {:d}'.format(int(t2 - t1)) )
    print('    Time elapsed', strin)