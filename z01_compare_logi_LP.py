
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
if platform.node()=='zlhp':
    file_path = "/mnt/d/project/hydraulic/test/code/python_public_function/"  # Change this to the actual directory
else:
    file_path = "/home/zlu/python_public_function/"
sys.path.append(file_path)
import read_functions as my_rf
import plot_functions as my_pf
import analyse_functions as my_af



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
        self.llxbbox    = 0.04  # x-anchor legend bounding box
        self.llybbox    = 1.5  # y-anchor legend bounding box
        self.llrspace   = 0.    # spacing between rows in legend
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
    def plot_new(self,df_merge,xvar,yvar,clcvar,xlab,ylab,tit,iplot,colors=None,y_lim=None):
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

        sns.scatterplot(
            data=df_merge,
            x=xvar,
            y=yvar,
            hue=clcvar,
            style=clcvar,  
            alpha=0.6,
            palette=colors,  # List of 3 colors
            ax=ax1,
            s=100,
            legend=(iplot==1)  
        )
        #plt.xlabel('Soil type')
        plt.ylabel(ylab)
       
        plt.title(tit,fontsize=6)
        plt.grid(True, color='gray', linewidth=0.4)  
        if iplot==1:
            ll = ax1.legend(frameon=self.frameon, ncol=4,
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

    sites = ['GF-Guy', 'BR-Sa3', 'FR-Bil', 'FR-Hes','AU-Cum',  'FR-Pue', 'FI-Hyy',  'US-SRM' ]
    sites = ['GF-Guy', 'BR-Sa3', 'FR-Hes','AU-Cum',  'FI-Hyy',  'US-SRM' ]
    sites=['FR-Hes']
    sites=['AU-Cum','FR-LBr','FR-Pue']
    sites=['FR-LBr']
    suffmod = '/outputs/site_out_cable_*.nc'
    if platform.node()=='zlhp':
        
        fp1='/mnt/d/project/hydraulic/test/'
        fpobs = '/mnt/d/project/hydraulic/test/obs4/'
        fmet = '/mnt/d/project/hydraulic/test/input/met/'
        file_path1 = '/mnt/d/project/hydraulic/test/plot/f03_test_fitting/result.xlsx'
        file_path1 = '/mnt/d/project/hydraulic/test/plot/f05_test_newIter/result_newRNumtag.xlsx'
        file_path2 = '/mnt/d/project/hydraulic/test/plot/f03b_test_liear_plateau/result_LP.xlsx'
        file_path2 = '/mnt/d/project/hydraulic/test/plot/f05_test_newIter/result_LP_newRNumtag.xlsx'
        file_path1 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_BASEdc98fe39.xlsx'
        file_path2 = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/result_LP_BASEdc98fe39.xlsx'

    else:
        fp1='/home/zlu/project/test1105/'
        fpobs='/home/mcuntz/projects/coco2/obs4/'
        fmet = '/home/mcuntz/projects/coco2/cable-pop/input/'

        mapping = {3: 1, 6: 2, 2: 3, 5:4, 7:5, 4:6, 1:7}
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
    for isite in sites:
        df = pd.read_excel(file_path1, sheet_name=isite)
        ##################################################### add a new column "best"
        conditions = [
            df["Xname"].isin(["wb_soilmean", "wb_rwc_soilmean"]),
            df["Xname"] == "rwc_soilmean_recal"
        ]

        choices = [
            df["MaxCurChange"],
            df["MinCurChange"]
        ]
        df["best"] = np.select(conditions, choices, default=np.nan)

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

        df_wb['MaxCur'] = df_wb.apply(lambda row: normalize_val(row['MaxCur'], row['swilt'], row['sfc']), axis=1)
        df_wb['MaxCurChange'] = df_wb.apply(lambda row: normalize_val(row['MaxCurChange'], row['swilt'], row['sfc']), axis=1)
        df_wb['MinCurChange'] = df_wb.apply(lambda row: normalize_val(row['MinCurChange'], row['swilt'], row['sfc']), axis=1)
        df_wb['best'] = df_wb.apply(lambda row: normalize_val(row['best'], row['swilt'], row['sfc']), axis=1)
        # 5. Drop the helper columns if you want
        df_wb = df_wb.drop(columns=['swilt', 'sfc'])

        # 6. Append new rows to original dataframe
        df_new = pd.concat([df, df_wb], ignore_index=True)
        ############################################### convert wb to psi
        df_wb = df[df['Xname'] == 'wb_soilmean'].copy()
        df_wb['Xname'] = 'wb_psi_soilmean'
        df_wb = df_wb.merge(soil_df[['soiltype', 'ssat', 'sucs','bch']], on='soiltype', how='left')
        # 4. Recalculate MaxCur and MaxCurChange with the formula:
        def theta2psi(val, ssat,sucs, bch):
            if val < 0:
                return np.nan
            psi_sat = sucs * 1000 * 9.81  / 1e6
            return psi_sat * (val/ssat)**(-bch)
        df_wb['MaxCur'] = df_wb.apply(lambda row: theta2psi(row['MaxCur'], row['ssat'], row['sucs'],row['bch']), axis=1)
        df_wb['MaxCurChange'] = df_wb.apply(lambda row: theta2psi(row['MaxCurChange'], row['ssat'], row['sucs'],row['bch']), axis=1)
        df_wb['MinCurChange'] = df_wb.apply(lambda row: theta2psi(row['MinCurChange'],row['ssat'], row['sucs'],row['bch']), axis=1)
        df_wb['best'] = df_wb.apply(lambda row: theta2psi(row['best'], row['ssat'], row['sucs'],row['bch']), axis=1)

        df_wb = df_wb.drop(columns=[ 'ssat', 'sucs','bch'])
        df_new = pd.concat([df_new, df_wb], ignore_index=True)
        df_new = df_new.merge(soil_df[['soiltype', 'sand fraction']], on='soiltype', how='left')
        df1 = df_new


    ###################################################################################################################################################
        df = pd.read_excel(file_path2, sheet_name=isite)
        df["BreakPoint"] = df["BreakPoint"].where((df["BreakPoint"] > 0) & (df["BreakPoint"] < 1))
        ############################## convert wb to rwc as append as new rows where xname='wb_rwc_soilmean'
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

        df_wb['MaxCur'] = df_wb.apply(lambda row: normalize_val(row['MaxCur'], row['swilt'], row['sfc']), axis=1)
        df_wb['MaxCurChange'] = df_wb.apply(lambda row: normalize_val(row['MaxCurChange'], row['swilt'], row['sfc']), axis=1)
        df_wb['MinCurChange'] = df_wb.apply(lambda row: normalize_val(row['MinCurChange'], row['swilt'], row['sfc']), axis=1)
        df_wb['BreakPoint'] = df_wb.apply(lambda row: normalize_val(row['BreakPoint'], row['swilt'], row['sfc']), axis=1)
        # 5. Drop the helper columns if you want
        df_wb = df_wb.drop(columns=['swilt', 'sfc'])

        # 6. Append new rows to original dataframe
        df_new = pd.concat([df, df_wb], ignore_index=True)
        df_new["BreakPoint"] = df_new["BreakPoint"].where((df_new["BreakPoint"] > 0) & (df_new["BreakPoint"] < 1))
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
        df_wb['MaxCur'] = df_wb.apply(lambda row: theta2psi(row['MaxCur'], row['ssat'], row['sucs'],row['bch']), axis=1)
        df_wb['MaxCurChange'] = df_wb.apply(lambda row: theta2psi(row['MaxCurChange'], row['ssat'], row['sucs'],row['bch']), axis=1)
        df_wb['MinCurChange'] = df_wb.apply(lambda row: theta2psi(row['MinCurChange'],row['ssat'], row['sucs'],row['bch']), axis=1)
        df_wb['BreakPoint'] = df_wb.apply(lambda row: theta2psi(row['BreakPoint'], row['ssat'], row['sucs'],row['bch']), axis=1)
        df_wb = df_wb.drop(columns=[ 'ssat', 'sucs','bch'])
        df_new = pd.concat([df_new, df_wb], ignore_index=True)
        df_new = df_new.merge(soil_df[['soiltype', 'sand fraction']], on='soiltype', how='left')
  
        df2 = df_new

        ###########################################################################################################

        xname_labels = {
            "wb_soilmean": fr'$\theta_{{crit}}$',
            "wb_rwc_soilmean": fr'$\theta-REW_{{crit}}$',
            #"rwc_soilmean_recal":fr'$REW_{{crit}}$',
            "wb_psi_soilmean": fr'$\theta-\psi_{{crit}}$',
        }  
        yvarsdf1 = [
            ("(Edemand - TVeg)/Edemand", r'$\frac{(E_{sat}-E)}{E_{sat}}$'),
            ("(GPP_wb - GPP)/GPP_wb", r'$\frac{(GPP_{sat}-GPP)}{GPP_{sat}}$'),
            ("(gsw_sw - gs)/gsw_sw", r'$\frac{(gs_{sat}-gs)}{gs_{sat}}$'),
            ("(Edemand - TVeg)/Edemand", r'$\frac{(E_{sat}-E)}{E_{sat}}$'),
            ("(GPP_wb - GPP)/GPP_wb", r'$\frac{(GPP_{sat}-GPP)}{GPP_{sat}}$'),
            ("(gsw_sw - gs)/gsw_sw", r'$\frac{(gs_{sat}-gs)}{gs_{sat}}$')
        ]

        yvarsdf2 = [
            ("Transpiration",'E'),
            ("GPP","GPP"),
            ("gsw","gsw"),
            ('latent heat fraction','latent heat fraction'),
            ('latent heat fraction','latent heat fraction'),
            ('latent heat fraction','latent heat fraction')
        ]
        imethod = 'percentile'
        iplot=0
      
        indexnames=['MaxCur','MaxCurChange','MinCurChange']
        indexnames=['MaxCur']
        idxname2 = 'BreakPoint'
        keyvar = 'soiltype'
        for iseg in [3]:
            for xname, xlabel in xname_labels.items():
                if xname != 'wb_psi_soilmean':
                    continue
                for idxname1 in indexnames:
                    for (yvar1, val1), (yvar2, val2) in zip(yvarsdf1, yvarsdf2):
                    #for yvar1, yvar1 in zip(yvarsdf1, yvarsdf2):
                        df1_filtered = df1[(df1["Xname"] == xname) & (df1["yvar"] == yvar1) & (df1["method"] == imethod) & (df1["segID"] == iseg) ]
                        df2_filtered = df2[(df2["Xname"] == xname) & (df2["yvar"] == yvar2) & (df2["method"] == imethod) & (df2["segID"] == iseg) ]
                        merged_df = pd.merge(df1_filtered[[keyvar,idxname1]], df2_filtered[[keyvar,idxname2]], on=keyvar, how="inner")
                        if iseg==4:
                            seglab='whole'
                        else:
                            seglab=iseg+1
                        xlab = fr'{xlabel}: {val1}'
                        ylab = fr'{xlabel}: {val2}'
                        if iseg==4:
                            tit = fr'{isite}:{xlabel} (index={idxname1}, segID=whole)'
                        else:
                            tit = fr'{isite}:{xlabel} (index={idxname1}, segID={iseg+1})'
                        cc = merged_df[keyvar]
                        unique_values = list(dict.fromkeys(cc))
                        soil_to_int = {soil: i for i, soil in enumerate(unique_values)}
                        colors_int = np.array([soil_to_int[s] for s in cc])
                        cmap = plt.get_cmap('RdYlBu_r', len(unique_values))
                        bounds = np.arange(-0.5, len(unique_values), 1)
                        clcpro_default = {
                            "clab": None,
                            "ctick": range(len(unique_values)),
                            "cticklab": unique_values,
                            'cbound': bounds,
                        }
                        if iplot==0:
                            iplot,ax1 = my_pf.plot_scatter_cmap(merged_df[idxname1],merged_df[idxname2],colors_int,xlab,ylab,tit=tit,
                                                            clcpro=clcpro_default,cmap='RdYlBu_r',self=self,iplot=iplot,oneline=True,clcbarloc='top',clcbarpad=0.05,ifcor=True)
                                                                                                           
                        else:
                            iplot,ax1 = my_pf.plot_scatter_cmap(merged_df[idxname1],merged_df[idxname2],colors_int,xlab,ylab,tit=tit,
                                                            clcpro=clcpro_default,cmap='RdYlBu_r',self=self,iplot=iplot,oneline=True,ifcolorbar=False,ifcor=True)     
                        title_obj = ax1.title  
                        title_obj.set_fontsize(6.5)  
        # isoil = 'fine clay'
        # keyvar = 'segID'
        # for xname, xlabel in xname_labels.items():
        #     for yvar1, yvar2 in zip(yvarsdf1, yvarsdf2):
        #         df1_filtered = df1[(df1["Xname"] == xname) & (df1["yvar"] == yvar1) & (df1["method"] == imethod) & (df1["soiltype"] == isoil) ]
        #         df2_filtered = df2[(df2["Xname"] == xname) & (df2["yvar"] == yvar2) & (df2["method"] == imethod) & (df2["soiltype"] == isoil) ]
        #         merged_df = pd.merge(df1_filtered[[keyvar,idxname1]], df2_filtered[[keyvar,idxname2]], on=keyvar, how="inner")
        #         if iseg==4:
        #             seglab='whole'
        #         else:
        #             seglab=iseg+1
        #         xlab = fr'{xlabel} {yvar1}'
        #         ylab = fr'{xlabel} {yvar2}'
                
        #         tit = fr'{xlabel} (method={imethod}, soiltype={isoil})'

        #         cc = merged_df[keyvar]
        #         unique_values = list(dict.fromkeys(cc))
        #         soil_to_int = {soil: i for i, soil in enumerate(unique_values)}
        #         colors_int = np.array([soil_to_int[s] for s in cc])
        #         cmap = plt.get_cmap('RdYlBu_r', len(unique_values))
        #         bounds = np.arange(-0.5, len(unique_values), 1)
        #         clcpro_default = {
        #             "clab": None,
        #             "ctick": range(len(unique_values)),
        #             "cticklab": unique_values,
        #             'cbound': bounds,
        #         }
        #         if iplot==0:
        #             iplot = my_pf.plot_scatter_cmap(merged_df[idxname1],merged_df[idxname2],colors_int,xlab,ylab,tit=tit,
        #                                             clcpro=clcpro_default,cmap='RdYlBu_r',self=self,iplot=iplot,oneline=True,clcbarloc='top',clcbarpad=0.9,ifcor=True)
        #         else:
        #             iplot = my_pf.plot_scatter_cmap(merged_df[idxname1],merged_df[idxname2],colors_int,xlab,ylab,tit=tit,
        #                                             clcpro=clcpro_default,cmap='RdYlBu_r',self=self,iplot=iplot,oneline=True,ifcolorbar=False,ifcor=True)     


    if iplot>0:
        self.plot_save(self.fig)


    self.close()


    t2    = ptime.time()
    strin = ( '[m]: {:.1f}'.format((t2 - t1) / 60.)
              if (t2 - t1) > 60.
              else '[s]: {:d}'.format(int(t2 - t1)) )
    print('    Time elapsed', strin)