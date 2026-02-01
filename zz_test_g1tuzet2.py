
from mcplot import mcPlot
import numpy as np
import pandas as pd
import matplotlib
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
from scipy.stats import ttest_ind
from scipy.stats import ttest_1samp
from itertools import combinations

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
      

        self.nrow     = 4     # # of rows of subplots per figure
        self.ncol     = 2    # # of columns of subplots per figure
        self.right = 0.95
        self.left = 0.08
        self.hspace   = 0.04  # x-space between subplots
        self.vspace   = 0.03  # y-space between subplots
        self.textsize = 10    # standard text size
        self.dxabc    = 0  # % of (max-min) shift to the right
                              # of left y-axis for a,b,c,... labels
        self.dyabc    = 1.04  # % of (max-min) shift up from lower x-axis
                              # for a,b,c,... labels
        # legend
        self.llxbbox    = -0.1  # x-anchor legend bounding box
        self.llybbox    = 1.2  # y-anchor legend bounding box
        self.llrspace   = 0.    # spacing between rows in legend
        self.llcspace   = 1.0   # spacing between columns in legend
        self.llhtextpad = 0.4   # pad between the legend handle and text
        self.llhlength  = 1.5   # length of the legend handles

        self.rasterized = True
        self.dpi = 150

        self.set_matplotlib_rcparams()

        my_pf.set_plot_style()


    def read_data_notCommand(self,fullnames,shortnames,vars):
        fns = dict(zip(shortnames, fullnames))
        self.fns = fns
        print('Reading input')
        for ifn in fns:
            fn = fns[ifn]
            if ifn != "obs":
                #modify_nc_time(fn)
                print(f'    Reading 1D model results from file {fn}')
                a = my_rf.read_md(fn,vars)
                setattr(self, ifn, a)
             
            else:

                print(f'    Reading obs from file {fn}')
                obs = my_rf.read_obs(fn)
                setattr(self, ifn, obs)
    def find_drydown_period(self,isite,df,x_lim = None,min_drydown_days: int = 10, rain_thresh: float = 0.005):
        # because in daily.nc, rainf is average, in order to get the total `rain` in one day, need to multiply by 86400
        if ('AU' in isite) or ('ZM' in isite):
            months = monthss#[1,2,3,4,11,12]
        else:
            months = monthnn#[5,6,7,8,9]
        df = df.sel(time=df['time'].dt.month.isin(months))
        rain = df['Rainf']#.resample(time="1D").sum()
        soilwater = df['SoilMoistPFT']#.resample(time="1D").mean()
        if x_lim is not None:
            rain = rain.sel(time=slice(x_lim[0],x_lim[1]))
            soilwater = soilwater.sel(time=slice(x_lim[0],x_lim[1]))
        years = np.unique(soilwater.time.dt.year)
        all_drydowns = []
        for year in years:
            # Select data for growing season of this year
            mask = (soilwater.time.dt.year == year)
            swsub = soilwater.sel(time=mask)

            rainsub = rain.sel(time=mask)
        
            time = swsub.time

            rain_event = (rainsub > rain_thresh)
            #sw_diff = swsub.diff(dim='time', label='upper')
            sw_diff = swsub.diff('time') / swsub.shift(time=1)
            pad = xr.DataArray([True], coords={ 'time':  [swsub.time.values[0]] }, dims='time')
            #decreasing = xr.concat([pad, sw_diff < 0], dim='time')
            decreasing = xr.concat([pad, sw_diff < 0.001], dim='time')
            decreasing['time'] = time
            i = 0
            while i < len(time) - min_drydown_days:
                if (rain_event[i]) or (i==0):
                    if rain_event[i]:
                        j = i + 1
                    else:
                        j = i
                    while j < len(time) and decreasing[j] and (not rain_event[j]) :
                        j += 1
                    if j - i - 1 >= min_drydown_days:
                        all_drydowns.extend(time[i+1:j].values)
                    i = j
                else:
                    i += 1
        # all_drydowns = all_drydowns.astype('datetime64[D]')

        return np.array(all_drydowns)

    def calc_xx_yy_cc(self,isite,xx,yy,clcvar,method,iseg,x_lim = None):
        if (clcvar is None):
            print("!!!!! clcvar or clcname is lack for color scatter")
            return  # Exit the function early

        if ('AU' in isite) or ('ZM' in isite):
            months = monthss#[1,2,3,4,11,12]
        else:
            months = monthnn#[5,6,7,8,9]
        if isinstance(xx, xr.DataArray):
            xx,yy = my_af.cal_common_timestamp_dA(xx,yy)

            if x_lim is None:
                xx = xx.sel(time=xx['time'].dt.month.isin(months))
                yy = yy.sel(time=yy['time'].dt.month.isin(months))
                clcvar = clcvar.sel(time=clcvar['time'].dt.month.isin(months))
         
            elif len(x_lim)==2:
                xx = xx.sel(time=slice(x_lim[0],x_lim[1]))
                yy = yy.sel(time=slice(x_lim[0],x_lim[1]))
                clcvar = clcvar.sel(time=slice(x_lim[0],x_lim[1]))
    
                xx = xx.sel(time=xx['time'].dt.month.isin(months) )
                yy = yy.sel(time=yy['time'].dt.month.isin(months) )
                clcvar = clcvar.sel(time=clcvar['time'].dt.month.isin(months) )

     
            else:
                xx = xx.sel(time=x_lim)
                yy = yy.sel(time=x_lim)
                clcvar = clcvar.sel(time=x_lim)
                xx = xx.sel(time=xx['time'].dt.month.isin(months))
                yy = yy.sel(time=yy['time'].dt.month.isin(months))
                clcvar = clcvar.sel(time=clcvar['time'].dt.month.isin(months))
     

        if isinstance(clcvar, xr.DataArray):
            xx1,cc = my_af.cal_common_timestamp_dA(xx,clcvar)
            cc = cc.values
            xx = xx.values
            yy = yy.values
        else:
            cc = clcvar
        bounds = None
        unique_values = sorted(np.unique(cc))
        if len(unique_values)<=10:
            cmap = plt.get_cmap('RdYlBu_r', len(unique_values))
            bounds = [x - 0.5 for x in unique_values]
            bounds.append(unique_values[-1]+0.5)
        else:
            N = 4
            if method =='Equal':
                vmin = np.quantile(cc,0.01)
                vmax = np.quantile(cc,0.99)
                bounds = np.linspace(vmin, vmax, N+1)
                l_bounds = bounds[0:-1]
                # l_bounds = np.append(l_bounds, bounds[1])
                # l_bounds = np.append(l_bounds, np.nanmin(cc))

                u_bounds = bounds[1:]
                # u_bounds = np.append(u_bounds, np.nanmax(cc))
                # u_bounds = np.append(u_bounds, np.nanmax(cc))
                if iseg<4:
                    mask = (cc > l_bounds[iseg]) & (cc <= u_bounds[iseg])
                    xx = xx[mask]
                    yy = yy[mask]
                    cc = cc[mask]
            elif method == 'percentile':
                bounds = [np.nanmin(cc), np.quantile(cc,0.25), np.quantile(cc,0.50),np.quantile(cc,0.75),np.nanmax(cc)]
                if iseg<4:
                    mask = (cc > bounds[iseg]) & (cc <= bounds[iseg+1])
                    xx = xx[mask]
                    yy = yy[mask]
                    cc = cc[mask]
        clcpro_default = {
            "clab": None,
            "ctick": None,
            "cticklab": None,
        }
        clcpro_default['cbound'] = bounds
        return xx,yy,cc, clcpro_default
    def plot_boxplot(self,df,iplot,ylab=None,tit = None,sep_bin= None):
        import seaborn as sns
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
        pos  = position(self.nrow, self.ncol, iplot,right = 0.95, left = 0.1,
                        hspace=self.hspace, vspace=self.vspace)

        ax1 = fig.add_axes(pos)
        ax1.tick_params(axis='y', colors='k')

        ax1.yaxis.label.set_color('k')
        min_count = 3
        counts = df.groupby(["x_bin", "source"]).size().reset_index(name="n")

        # Keep only bins where ALL sources have >= min_count
        valid_bins = counts.groupby("x_bin")["n"].min()
        valid_bins = valid_bins[valid_bins >= min_count].index

        # Filter df_all
        df = df[df["x_bin"].isin(valid_bins)]
        df["x_bin"] = df["x_bin"].cat.remove_unused_categories()
        sns.boxplot(
            x="x_bin",
            y="y",
            hue="source",        # separate color by dataframe
            data=df,
            palette=["#1f77b4", "#ff7f0e", "#2ca02c"],  # optional custom colors,
            showfliers=False,  # <-- hide outliers,
            boxprops={"edgecolor": "none"} , # remove box edges,
        )
        for i, b in enumerate(df["x_bin"].cat.categories):
            bin_str = str(b)
            if pvals.get(bin_str, 1) >= 0.05:  # not significant
                # max NTD in this bin across all sources
                max_y = np.percentile(df.loc[df["x_bin"] == b, "y"],90)
                ax1.text(
                    i,                # x-position = bin index
                    max_y + 0.02,     # y-position slightly above the box
                    "n.a.",           # text
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    color='red'
                )
        if tag == 'NTD':
            ax1.axhline(0.05, color='k', linestyle='--', linewidth=0.8)
        plt.ylabel(ylab)
        plt.xticks(rotation=45) 
        plt.xlabel(fr'$\psi\ bin$')
        if sep_bin is not None:
            # find category index
            categories = list(df["x_bin"].cat.categories)

            if sep_bin in categories:
                i = categories.index(sep_bin)

                ax1.axvline(
                    i - 0.5,
                    color="k",
                    linestyle="--",
                    linewidth=0.8
                )
                ax1.axvline(
                    i + 0.5,
                    color="k",
                    linestyle="--",
                    linewidth=0.8
                )
        abc2plot(ax1, self.dxabc, self.dyabc,
                (self.ifig - 1) * nplotperpage + iplot  ,
                lower=True, bold=True, usetex=self.usetex, mathrm=True,large=True)
        ax1.legend_.remove()
        plt.title(tit)
        #plt.legend( frameon=False)
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

    ifttest2 = True
    ifttest0p05 = False
    drydown = False
    self = PlotIt('Scatter Plot')

    sites = ['GF-Guy', 'BR-Sa3', 'FR-Bil', 'FR-Hes','AU-Cum',  'FR-Pue', 'FI-Hyy',  'US-SRM' ]
    sites = ['GF-Guy', 'BR-Sa3', 'FR-Hes','AU-Cum',  'FI-Hyy',  'US-SRM' ]
    sites=['FR-Hes']
    sites=['AU-Cum','FR-LBr','FR-Pue']
    
    sites=['AU-Gin','ZM-Mon','AU-ASM','IT-Cp2','IT-SRo','FR-LBr','IT-Ro1','FR-Pue','CN-Qia','AU-How',]
    sites=['IT-Ro1','FR-Pue']

    sites=[ "AU-ASM",  "US-SRM" ,"AU-Gin" ,"US-Me6", "AU-How", "FR-Pue", "IT-Cp2" ,"ZM-Mon", "IT-Ro1" ,"IT-SRo",
 "CN-Qia", "FR-LBr","GF-Guy", "MY-PSO", "CA-Qfo", "CA-SF2", "CA-TPD",
 "SE-Nor" ,"CN-Cng" ,"CN-Dan"]
    sites = [#"US-SRM" ,
        "IT-Cp2" ,
    "FR-LBr", 
    #"CN-Qia",
            ]

    suffmod = '/outputs/site_out_cable_*.nc'
    if platform.node()=='zlhp':
        
        fp1='/mnt/e/biocomp/test1105/'
        fpobs = '/mnt/d/project/hydraulic/test/obs4/'
        fmet = '/mnt/d/project/hydraulic/test/input/met/'
        fout = '/mnt/d/project/hydraulic/test/plot/fig_final/ztmp/'

    else:
        fp1='/home/zlu/project/test1105/'
        fpobs='/home/mcuntz/projects/coco2/obs4/'
        fmet = '/home/mcuntz/projects/coco2/cable-pop/input/'
        fout = '/home/zlu/project/test1105/code/'
    
    if drydown:
        segs = [4]
        outfile = f"{fout}ttest_wb_g1tuzet.xlsx"
        monthss = [1,2,3,12]
        monthnn = [6,7,8,9]
        tag = 'EF'
    else:
        segs = [0,1,2,3,4]
        segs = [3]
        outfile = f"{fout}ttest_wb_g1tuzet.xlsx"
        monthss = [1,2,3,4,11,12]
        monthnn = [5,6,7,8,9]
        tag = 'NTD'
    varlab0,sn=['tuzet_LWP_Haverd2013/','tuzet']
 
    if tag == 'NTD':
        ylab = r'$\frac{(E_{sat}-E)}{E_{sat}}$'
        
    else:
        ylab =  r'normalized f$_{LE}$'
    
    snd = {
       'fine clay': 'Stype3_Vtype1',
        'silty clay': 'Stype6_Vtype1',
       'clay loam': 'Stype2_Vtype1',
        'sandy clay': 'Stype5_Vtype1',
        'sandy clay loam':'Stype7_Vtype1',
        'Sandy loam': 'Stype4_Vtype1',
        'sand':'Stype1_Vtype1',
            }
    sep_bins = {
       'fine clay': "[0.32, 0.34)",
        'silty clay': "[0.30, 0.32)",
       'clay loam': "[0.24, 0.26)",
        'sandy clay': "[0.24, 0.26)",
        'sandy clay loam':"[0.20, 0.22)",
        'Sandy loam':"[0.16, 0.18)",
        'sand':"[0.08, 0.10)",
            }
    snd = {k: v + "_g1tuzet" for k, v in snd.items()}

    # snd = {
    #    'fine clay': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype3_Vtype1',
    #     'silty clay': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype6_Vtype1',
    #     'clay loam': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype2_Vtype1',
    #     'sandy clay': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype5_Vtype1',
    #     'sandy clay loam':'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype7_Vtype1',
    #     'Sandy loam': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype4_Vtype1',
    #     'sand':'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype1_Vtype1',
    #         }
    # tag = 'LBrLAI'
    
    vars=['GPP','Qle']     
    varread = ['Rainf','GPP','TVeg','Qle','Qh','gsw_sl',
                'SoilMoistPFT',
                'psi_soilmean','wb_soilmean','rwc_soilmean_recal',
                'wb_30','rwc_30','psi_30',
                'wb_fr_rootzone','rwc_fr_rootzone','psi_fr_rootzone',
                'wb_depth_rootzone','rwc_depth_rootzone','psi_depth_rootzone',
                'epotcan3',
                'gsw_epotcan3_sl','GPP_epotcan3']
    suffobs = '_flux.nc'
    varsoil = {
    'wb_soilmean': 'mean soil water content (m m$^{-1}$)',
    #'rwc_soilmean': 'relative water content',
    #'rwc_soilmean_recal': 'relative water content (recal)',
    #'psi_soilmean': 'mean soil water potential (MPa)',
    #'wb_30': 'mean soil water content in 30 cm (m m$^{-1}$)',
    #'rwc_30': 'relative water content in 30 cm (m m$^{-1}$)',
    #'psi_30':'mean soil water potential in 30 cm (Mpa)',
    #'wb_fr_rootzone':'mean soil water content in rootzone weighted by root fraction(m m$^{-1}$)',
    #'rwc_fr_rootzone':'relative water content in rootzone weighted by root fraction(m m$^{-1}$)',
    #'psi_fr_rootzone':'mean soil water potential in rootzone weighted by root fraction(Mpa)',
    #'wb_depth_rootzone':'mean soil water content in rootzone weighted by layer depth(m m$^{-1}$)',
    #'rwc_depth_rootzone':'relative water content in rootzone weighted by layer dept(m m$^{-1}$)',
    #'psi_depth_rootzone':'mean soil water potential in rootzone weighted by layer dept(Mpa)',
    }
    

    n = 0
    frq=''
    kmax = 0.6
    psi_ref = 0.8
    g2 = 1.0
    light_grey = (0.8, 0.8, 0.8)  # Light grey
    dark_grey = (0.4, 0.4, 0.4)   # Dark grey
    black = (0.0, 0.0, 0.0)       # Black
    iplot = 0


# Extract the two 'xxxx' parts before the .nc extension
    fitnames = ['logi','tanh','Dtanh','expo','Sexpo']
    fitnames = ['logi','Sexpo']
    methods = ['None','DE','DA','SH']
    iplot = 0
    calc_methods = ['percentile']
    bins = [-1.2,-1.1,-1.0,-0.9, -0.8,-0.7, -0.6, -0.5,-0.4,-0.3, -0.2, -0.1, -0.08,-0.06,-0.04,-0.02,0.0]
    labels = [
        "[-1.2, -1.1)",
        "[-1.1, -1.0)",
        "[-1.0, -0.9)",
        "[-0.9, -0.8)",
        "[-0.8, -0.7)",
        "[-0.7, -0.6)",
        "[-0.6, -0.5)",
        "[-0.5, -0.4)",
        "[-0.4, -0.3)",
        "[-0.3, -0.2)",
        "[-0.2, -0.1)",
        "[-0.1, -0.08)",
            "[-0.08, -0.06)",
            "[-0.06, -0.04)",
            "[-0.04, -0.02)",
            "[-0.02, 0)",
    ]   
    bins = np.arange(0.1, 0.46, 0.02)
    labels = [
        f"[{bins[i]:.2f}, {bins[i] + 0.02:.2f})"
        for i in range(len(bins)-1)
    ]
    

    for ikey,imod in snd.items():
        dfs = {}
        for itest in [1,2]:#range(3,5,1):
            print("HIT shortnames init, itest =", itest)
            fullnames = []
            shortnames = []
            for isite in sites:
                #fn = glob.glob(f'{fp1}{isite}/{varlab0}/{imod}/outputs/site_out_cable_*_daily.nc')
                #fn_all = glob.glob(f'{fp1}{isite}/{varlab0}/{imod}/outputs/site_out_cable_*.nc')
                fn_all = glob.glob(f'{fp1}{isite}/{varlab0}/{imod}/test{itest}/outputs/site_out_cable_*.nc')
                # fn_all = glob.glob(f'{fp1}{isite}/tuzet_LWP_Haverd2013/{imod}/outputs/site_out_cable_*.nc')
                # print(fn_all)
                fn = [f for f in fn_all if not f.endswith('daily.nc')]
                if not fn:  # same as: if len(fn) == 0
                    continue  # skip this iteration
                match = re.search(r'_([0-9]+)_([0-9]+)', fn[0])
                yr_st = int(match.group(1))
                yr_en = int(match.group(2))
                fullnames += fn
                shortnames += [f'{ikey}-{isite}-test{itest}']
            end1    = ptime.time()
            if not fullnames:
                continue
            print(f"get modname {end1 - t1:.1f} seconds")
            # fullnames+=[fpobs  +isite + suffobs]
            # shortnames+=['obs']
            print(f'   fullname {fullnames}')
            print(f'   shortname {shortnames}')

            self.read_data_notCommand(fullnames,shortnames,varread)
            x_lim = [f"{yr_st+3}-01-01",f"{yr_en}-12-30"]
            # if isite == 'CN-Qia':
            #     x_lim = [f"{yr_st}-01-01",f"{yr_en}-12-30"]

            if isite == 'AU-Cum':
                x_lim = ["2013-01-01","2014-12-30"]
            end2    = ptime.time()
            print(f"read data {end2 - end1:.1f} seconds")
            
            df_all = pd.DataFrame(columns=["x", "y"])
            for isn in shortnames:
                print(f"soil types: {isn}")

                ds = getattr(self,isn)

                # sand= ds['sand'].values.flatten()[0]
                # clay = ds['clay'].values.flatten()[0]
                # silt = ds['silt'].values.flatten()[0]
                tit=f'{isite}: {isn}'
                self.tit = tit
                ############# use hourly data
                # df_filtered = ds.where(ds['leaf_to_air_vpd'] >= 0.6,drop=True)

                # Tveg = df_filtered['TVeg'].resample(time='1D').mean()
                # epotcan3 = df_filtered['epotvpd'].resample(time='1D').mean()
                # # Tveg = ds['TVeg']
                # # epotcan3 = ds['epotvpd']
                # clcvar = ds['epotcan3'].resample(time='1D').mean()
                # dssoil = ds[list(varsoil.keys())].resample(time='1D').mean()

                ###########  directly use daily data
                Tveg = ds['TVeg']
                GPP = ds['GPP']
                GPP_epotcan3 = ds['GPP_epotcan3']
    
                epotcan3 = ds['epotcan3']
                clcvar = ds['epotcan3']
                #clcvar = epotcan3
                dssoil = ds[list(varsoil.keys())]
                gs_epotcan3 = ds['gsw_epotcan3_sl']
            # gs_epotvpd = ds['gsw_epotvpd_sl']
                gs = ds['gsw_sl']
                #gsref = ds['gsw_ref_sl']
                #Rsw = 1-(gsref-gs_epotcan3)/(gsref-gs)
                #Rsw = Rsw.where((Rsw >= 0) & (Rsw <= 1), np.nan)

                #################### different y
                if drydown:
                    plot_config = {

                        # '(Edemand - TVeg)/TVeg': {
                        #     'var': (epotcan3-Tveg)/Tveg,
                        #     'ylim': (-0.55, 0.6)
                        # },
                        'latent heat fraction': {
                            'var': ds['Qle']/(ds['Qle']+ds['Qh']),
                            'ylab': r'latent heat fraction',
                            'ylim': (-0.55, 1.5)
                        },

                    }
                else:
                    plot_config = {

                        # '(Edemand - TVeg)/TVeg': {
                        #     'var': (epotcan3-Tveg)/Tveg,
                        #     'ylim': (-0.55, 0.6)
                        # },
                        '(Edemand - TVeg)/Edemand': {
                            'var': (epotcan3-Tveg)/epotcan3,
                            'ylab': r'$\frac{(E_{wb}-E)}{E_{wb}}$',
                            'ylim': (-0.55, 1.5)
                        },

                    }


                for ylabel, config in plot_config.items():
                    for ix in varsoil:
                        print(f"soil var: {ix}")
                        print(f'yvar: {ylabel}')
                        xvar = dssoil[ix]
                        yvar = config['var']
                        ylim = config['ylim']
                    
                        for imethod in calc_methods:
                            for iseg in segs:
                                if drydown:
                                    ds['Rainf'] =  ds['Rainf'] * 86400
                                    times = self.find_drydown_period(isite,ds,x_lim)
                                    xx,yy,cc,clcpro=self.calc_xx_yy_cc(isite,xvar,yvar,clcvar,method=imethod,iseg=iseg,x_lim = times)
                                else:
                                    xx,yy,cc,clcpro=self.calc_xx_yy_cc(isite,xvar,yvar,clcvar,method=imethod,iseg=iseg,x_lim = x_lim)
                                if tag == 'EF':
                                    #r_lp, rmse_lp, x_fit,y_fit,bk= my_af.linear_plateau_fitting(xx,yy)
                                    #yy = yy/np.max(y_fit)
                                    yy = yy/np.percentile(yy,90)
                                if iseg<4:
                                    tit = f'{isite} {isn} {imethod}:seg{iseg+1}'
                                else:
                                    tit = f'{isite} {isn} {imethod}:whole'
                                if isite=='US-SRM':
                                    ifsigma = True
                                else:
                                    ifsigma = False
                                df = pd.DataFrame({
                                    "x": xx,
                                    "y": yy
                                })
                                df_all = pd.concat([df_all, df], ignore_index=True)
            df_all["x_bin"] = pd.cut(df_all["x"], bins=bins, labels=labels, include_lowest=True)
            dfs[itest] = df_all
 
                        #     iplot,xcmx,xccmx,xccmn,r,rmse,xcons1,xcons2,xcons3,xslope1,xslope2,xslope3,xcons3K = self.plot_scatter(xx,yy,ix,ylabel,'logi',
                        #  iplot,oneline=False,ifcolor=False,iffit = True,clcvar=clcvar,clcpro=clcpro,tit =tit ,fitDirec=True,ifsigma = ifsigma,x_lim=[0.26,0.425])
                   

        if ifttest2:
            results = []
            for (name1, df1), (name2, df2) in combinations(dfs.items(), 2):
                print(f"Comparing {name1} vs {name2}")
                

                alpha = 0.05  # significance level

                for b in labels:
                    ntd1 = df1.loc[df1["x_bin"] == b, "y"].dropna()
                    ntd2 = df2.loc[df2["x_bin"] == b, "y"].dropna()

                    if len(ntd1) >= 3 and len(ntd2) >= 3:
                        stat, pval = ttest_ind(ntd1, ntd2, equal_var=False)

                        mean1 = ntd1.mean()
                        mean2 = ntd2.mean()

                        if pval < alpha:
                            larger = "df1" if mean1 > mean2 else "df2"
                            significance = "significant"
                        else:
                            larger = "NA"
                            significance = "not significant"
                    else:
                        pval = np.nan
                        mean1 = np.nan
                        mean2 = np.nan
                        larger = "NA"
                        significance = "insufficient data"

                    results.append({
                        "soil_type_1": name1,
                        "soil_type_2": name2,
                        "psi_bin": b,
                        "mean_NTD_df1": mean1,
                        "mean_NTD_df2": mean2,
                        "p_value": pval,
                        "result": significance,
                        "larger_NTD": larger
                    })
            results_df = pd.DataFrame(results)
            # with pd.ExcelWriter(outfile, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            #     results_df.to_excel(writer, sheet_name=f'{isite}_{tag}_ttest', index=False)
        if ifttest0p05:
            constant = 0.05
            alpha = 0.05

            results = []

            for name, df in dfs.items():
                for b in df["x_bin"].cat.categories:
                    values = df.loc[df["x_bin"] == b, "y"].dropna()

                    if len(values) >= 3:
                        stat, pval = ttest_1samp(values, constant)
                        mean_val = values.mean()

                        if pval < alpha:
                            direction = "greater" if mean_val > constant else "smaller"
                            significance = "significant"
                        else:
                            direction = "NA"
                            significance = "not significant"
                    else:
                        pval = np.nan
                        mean_val = np.nan
                        direction = "NA"
                        significance = "insufficient data"

                    results.append({
                        "dataset": name,
                        "x_bin": str(b),
                        "mean_NTD": mean_val,
                        "p_value": pval,
                        "result": significance,
                        "vs_0.05": direction
                    })

            results_df = pd.DataFrame(results)
            with pd.ExcelWriter(outfile, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                results_df.to_excel(writer, sheet_name=f'{isite}_{tag}_ttest0.05', index=False)
        
        pvals = dict(zip(results_df["psi_bin"].astype(str), results_df["p_value"]))
        df_all = pd.concat(
            [df.assign(source=name) for name, df in dfs.items()],
            ignore_index=True
        )
        if iplot % self.ncol == 0:
            ylab0 = ylab
        else:
            ylab0 = ''
        iplot = self.plot_boxplot(df_all,iplot,ylab=ylab0,tit = ikey,sep_bin = sep_bins[ikey])
    plt.savefig(f"{fout}boxplot_g1tuzet_wbbin{tag}.png", dpi=300, bbox_inches="tight", transparent=True)
    # if iplot>0:
    #     self.plot_save(self.fig)


    self.close()


    t2    = ptime.time()
    strin = ( '[m]: {:.1f}'.format((t2 - t1) / 60.)
              if (t2 - t1) > 60.
              else '[s]: {:d}'.format(int(t2 - t1)) )
    print('    Time elapsed', strin)
