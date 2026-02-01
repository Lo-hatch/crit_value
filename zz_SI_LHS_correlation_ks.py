import pandas as pd
from mcplot import mcPlot
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import pearsonr
import matplotlib.ticker as mtick
import platform
import sys
from scipy.stats import linregress
from mcplot import position, str2tex, abc2plot
if platform.node()=='zlhp':
    file_path = "/mnt/d/project/hydraulic/test/code/python_public_function/"  # Change this to the actual directory
else:
    file_path = "/home/zlu/python_public_function/"
sys.path.append(file_path)
import read_functions as my_rf
import plot_functions as my_pf
import analyse_functions as my_af
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
      
        self.left     = 0.08  # left space on page
        self.right    = 0.95    # right space on page
        self.bottom   = 0.1   # lower space on page
        self.top      = 0.95   # upper space on page

        self.nrow     = 7     # # of rows of subplots per figure
        self.ncol     = 5    # # of columns of subplots per figure
        self.hspace   = 0.06  # x-space between subplots
        self.vspace   = 0.05  # y-space between subplots
        self.textsize = 10    # standard text size
        self.dxabc    = 0  # % of (max-min) shift to the right
                              # of left y-axis for a,b,c,... labels
        self.dyabc    = 1.04  # % of (max-min) shift up from lower x-axis
                              # for a,b,c,... labels
        # legend
        self.llxbbox    = 0.72  # x-anchor legend bounding box
        self.llybbox    = 0.5  # y-anchor legend bounding box
        self.llrspace   = 0.2    # spacing between rows in legend
        self.llcspace   = 1.0   # spacing between columns in legend
        self.llhtextpad = 0.4   # pad between the legend handle and text
        self.llhlength  = 1.5   # length of the legend handles

        self.rasterized = True
        self.dpi = 150

        self.set_matplotlib_rcparams()
        my_pf.set_plot_style()
    def plot_scatter(self,df,iplot):        
        nplotperpage = self.nrow * self.ncol
        colors = plt.cm.tab10.colors[:2]
        clc = colors[0]
        if iplot == 0:
            self.ifig += 1
            fig = plt.figure(self.ifig)
            self.fig = fig
        else:
            fig = self.fig

        iplot += 1
        pos  = position(self.nrow, self.ncol, iplot,right = 0.8,
                        hspace=self.hspace, vspace=self.vspace)

        ax1 = fig.add_axes(pos)
        ax1.tick_params(axis='y', colors='k')
        ax1.yaxis.label.set_color('k')
        colors = {"NTD": "tab:blue", "EF": "tab:orange"}
        markers = {"NTD": "o", "EF": "s"}


        # ax1.scatter( df_sorted[col], df_sorted[indexname],s=30, alpha=0.7)
        # ax1.set_ylabel(indexname)
        # ax1.set_xlabel(col)  
if __name__ == '__main__':
    # one model in one figure
    # three figures for each value of the tested variable: GPP,Qle, psi_can
    # show 2015-2016 period, dry and wet year of FR-Hes

    import time as ptime
    import glob
    t1 = ptime.time()
    
    critname = 'result_BASEdc98fe39_LHS_simple_new.xlsx'
    csvname = 'lhs_samples_50new.csv'
    cols_label = {'g1tuzet':'g1', 'psi_50_leaf':r'$P50_{l}$', 'kmax':'Ks', 'P50': r'$P50_{sx}$', 'P88dP50': r'$\Delta P_{88-50}$'}

    critname = 'result_BASEdc98fe39_LHS_param6.xlsx'
    csvname = 'lhs_samples_50_param6.csv'
    cols_label = {'g1tuzet':'g1', 'psi_50_leaf':r'$P50_{l}$', 'kmax':'Ks', 'P50': r'$P50_{sx}$', 'P88dP50': r'$\Delta P_{88-50}$','root_shoot':r'$B_{root}/B_{leaf}$'}
    

    critname = 'result_BASEdc98fe39_LHS_g1tuzet.xlsx'
    csvname = 'g1tuzet_sample.csv'
    cols_label = {'g1tuzet':'g1'}
    savetag = 'g1tuzet'

    critname = 'result_BASEdc98fe39_LHS_kmax.xlsx'
    csvname = 'ks_sample.csv'
    cols_label = {'kmax':'Ks'}
    savetag = 'kmax'

    critname = 'result_BASEdc98fe39_LHS_psi_50_leaf.xlsx'
    csvname = 'psi_50_leaf_sample.csv'
    cols_label = {'psi_50_leaf':r'$P50_{l}$'}
    savetag = 'psi_50_leaf'

    indexname1 = ['slope0p1']
    plotorstat = {
        'plotcorr':{'plot':True,'figname':f'z05_LHS_correlation_scatter_{savetag}_{indexname1[0]}'},
         'calcorr':{'plot':False,'figname':f'correlation_{savetag}_{indexname1[0]}'},
               }

    self = PlotIt('Scatter Plot')

    figout = '/mnt/d/project/hydraulic/test/plot/fig_final/v20251217/'
    sites = ['UIFC']
    file_path = '/mnt/d/project/hydraulic/test/plot/f05_test_LHS/'
    out_dir=file_path 
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
    tag = '_NTD'
    soilvar = 'wb_soilmean_rwc'
    if tag == '_EF':
        indexname = "BreakPoint"
    elif tag == '_NTD':
        indexname = "slope0p1"

        
    # 1. Read Excel files'
    commonkeys  = ['ID','Xname','soiltype','sand fraction']

    dfparams = pd.read_csv(f"{file_path}{csvname}")

    df2 = LP_excle(f"{file_path}{critname}",'_EF')

    df2 = df2[commonkeys+['BreakPoint']]
    df2 = df2.rename(columns={"BreakPoint": "EF"})
    df1 = logi_excle(f"{file_path}{critname}",'_NTD')

    df1 = df1[commonkeys+indexname1]
    #df1 = df1.rename(columns={'slope0p1': "NTD-slope10%",'MaxCur':"NTD-MaxCur"})
    df1 = df1.rename(columns={'slope0p1': "NTD"})
    df_merged = df1.merge(
        df2,
        on=commonkeys,
        how="inner"
    )
    df_merged = df_merged.merge(
        dfparams,
        on="ID",
        how="left"
        )
    if '50new' in csvname:
        df_merged = df_merged[~df_merged['ID'].isin([1,3,5,9,21])]
    #df_merged = df_merged[~df_merged['ID'].isin([1])]
    if 'g1tuzet' in csvname:
        df_merged['NTD-NL'] = df_merged['NTD']
        df_merged.loc[df_merged['ID'] == 1, 'NTD-NL'] = np.nan
        df_merged['EF-NL'] = df_merged['EF']
        df_merged.loc[df_merged['ID'] == 1, 'EF-NL'] = np.nan
    ################################ delete extreme combinations ##########################
    # mask = (df_merged['kmax'] <= 4) & (df_merged['P50'] >= -7)
    # df_merged = df_merged[mask]
    correlation_results = []
    xname_labels = {
        "wb_soilmean": fr'$\theta_{{crit}}$',
       # "wb_soilmean_rwc": fr'$REW_{{crit}}^{{\theta}}$',
        #"rwc_soilmean_recal":fr'$REW_{{crit}}$',
       # "wb_soilmean_psi": fr'$\psi_{{crit}}^{{\theta}}$',
        #"psi_soilmean": fr'$\psi_{{crit}}$',
    }  
    
    soil_types = df2["soiltype"].unique()

    if plotorstat["plotcorr"]["plot"]:
        
        figname = plotorstat["plotcorr"]["figname"]
        if 'param6' in csvname:
            self.ncol = 6
        self.ncol = 4

        iplot=0
        for xname, ylabel in xname_labels.items():
            for isoil in soil_types:
                for col, label in cols_label.items():
                    print(col, label)
                    df = df_merged[(df_merged["Xname"] == xname) & (df_merged["soiltype"] == isoil)]
                    if iplot % self.ncol == 0:
                        ylab = f'{ylabel}'
                    else:
                        ylab = ''
                    if iplot >= 4:#(self.nrow - 1) * self.ncol:
                        xlab = label
                    else:
                        xlab = ''
                    colors = [
                   
                    '#1b9e77',
                     "#1D4137",
                    '#d95f02',
                    "#d902a7",
                ]
                    
                    #yy = df[["NTD-MaxCur", "NTD-slope10%","EF"]].to_dict(orient="list")
                    if 'g1tuzet' in csvname:
                        color0 = colors
                        yy = df[["NTD","NTD-NL", "EF","EF-NL"]].to_dict(orient="list")
                    else:
                        yy = df[["NTD", "EF"]].to_dict(orient="list")
                        color0=None
                    if iplot==6:
                        iflegend = True
                    else:
                        iflegend = False
                    iplot=my_pf.plot_scatter_simple(
                        xx=df[col].values,
                        yy=yy,
                        xlab=xlab,
                        ylab=ylab,
                        tit = isoil,
                        iplot=iplot,
                        txtpos=[0.98, 0.9],
                        self = self,
                        colors = color0,
                        iflegend = iflegend
                    )
                    plt.savefig(f"{figout}{figname}{self.ifig}.png", dpi=300, bbox_inches="tight")

        if iplot>0:
            self.plot_save(self.fig)
        
    if plotorstat["calcorr"]["plot"]:
        figname = plotorstat["calcorr"]["figname"]+'.xlsx'
        results = []
        for xname, ylabel in xname_labels.items():
            for isoil in soil_types:
                df = df_merged[(df_merged["Xname"] == xname) & (df_merged["soiltype"] == isoil)]
                for col, label in cols_label.items():
                    for method in ['NTD','EF']:
                        
                        sub = df[[col, method]].dropna()  
                        res = linregress(sub[col], sub[method])
                        slope = res.slope
                        r = res.rvalue
                        p = res.pvalue
                        results.append({
                            'xname': xname,
                            'isoil': isoil,
                            'col': col,
                            'method': method,
                            'slope':slope,
                            'r': r,
                            'p': p
                        })
        df_out = pd.DataFrame(results)
        if not os.path.exists(f'{figout}{figname}'):
            # Create a new file
            with pd.ExcelWriter(f'{figout}{figname}', mode="w", engine="openpyxl") as writer:
                df_out.to_excel(writer, index=False)
        else:
            with pd.ExcelWriter(f'{figout}{figname}', mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                df_out.to_excel(writer, index=False)


    plt.show()

    self.close()

 