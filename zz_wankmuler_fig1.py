
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
      

        self.nrow     = 2     # # of rows of subplots per figure
        self.ncol     = 1    # # of columns of subplots per figure
        self.hspace   = 0.04  # x-space between subplots
        self.vspace   = 0.07  # y-space between subplots
        self.textsize = 10    # standard text size
        self.dxabc    = 0.02  # % of (max-min) shift to the right
                              # of left y-axis for a,b,c,... labels
        self.dyabc    = 1.06  # % of (max-min) shift up from lower x-axis
                              # for a,b,c,... labels
        # legend
        self.llxbbox    = 0.02  # x-anchor legend bounding box
        self.llybbox    = 1.7  # y-anchor legend bounding box
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
    
    def calc_xx_yy_cc(self,isite,Y,clcvar=None,x_lim = None):
       
        if ('AU' in isite) or ('ZM' in isite):
            months = monthss#[1,2,3,4,11,12]
        else:
            months = monthnn#[5,6,7,8,9]

        if x_lim is None:
            Y_new = Y.sel(time=Y.time.dt.month.isin(months))
            if clcvar is not None:
                clcvar = clcvar.sel(time=clcvar['time'].dt.month.isin(months))
        
        elif len(x_lim)==2:
            Y_new1 = Y.sel(time=slice(x_lim[0],x_lim[1]))
            Y_new = Y_new1.sel(time=Y_new1.time.dt.month.isin(months))
            if clcvar is not None:
                clcvar = clcvar.sel(time=slice(x_lim[0],x_lim[1]))
                clcvar = clcvar.sel(time=clcvar['time'].dt.month.isin(months))

    
        else:
            Y_new1 = Y.sel(time=x_lim)
            Y_new = Y_new1.sel(time=Y_new1.time.dt.month.isin(months))
            if clcvar is not None:
                clcvar = clcvar.sel(time=x_lim)
                clcvar = clcvar.sel(time=clcvar['time'].dt.month.isin(months))
        if clcvar is not None:
            cc = clcvar.values.flatten()
            bounds = [np.nanmin(cc), np.quantile(cc,0.25), np.quantile(cc,0.50),np.quantile(cc,0.75),np.nanmax(cc)]
            iseg=3
            if iseg<4:
                mask = (cc > bounds[iseg]) & (cc <= bounds[iseg+1])
                Y_new = Y_new.where(mask)
     
        return Y_new

    def plot_scatter(self, xx,Y, iplot, oneline=False, ifcolor=True, iffit = False, x_lim=None,y_lim=None, tit=None,xlog=False,ylog=False):

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
        colors = ['C0', 'C1', 'C2', 'C3']
        for var, c in zip(target, colors):
            y = np.asarray(Y[var])

            if y.ndim == 1:
                ax1.scatter(
                    xx.values,
                    y,
                    label=var,
                    color=c,
                    alpha=0.7
                )

            elif y.ndim == 2:
                for i in range(y.shape[1]):
                    ax1.scatter(xx.values, y[:, i], label=f'{var} dim {i}')
            
        if x_lim is not None:
            ax1.set_xlim(x_lim) 
        if y_lim is not None:
            ax1.set_ylim(y_lim) 
        if xlog:
            ax1.set_xscale('log')
            ax1.set_xlabel('x (log10)')
        if ylog:
            ax1.set_yscale('log')
            ax1.set_ylabel('Y (log10)')

        ax1.axvline(x=0.4, color=dark_grey, linestyle=':')
        ax1.axvline(x=0.6, color=dark_grey, linestyle=':')
        #plt.setp(ax1, xlabel=xlab,ylabel=ylab)
        
        #ax2.set_yticklabels([])
        # plt.setp(ax1,x_lim=[0,0.5],y_lim=[-0.01,0.7])
        # abc2plot(ax1, self.dxabc, self.dyabc,
        #         (self.ifig - 1) * nplotperpage + iplot,
        #         lower=True, bold=True, usetex=self.usetex, mathrm=True)

        ll = ax1.legend(frameon=self.frameon, ncol=1,
                        labelspacing=self.llrspace,
                        handletextpad=self.llhtextpad,
                        handlelength=self.llhlength,
                        loc='upper right',
                        #bbox_to_anchor=(self.llxbbox, self.llybbox),
                        scatterpoints=1, numpoints=1)
        plt.setp(ll.get_texts(), fontsize='small')
            #fig.suptitle(self.tit)

        # if tit is not None:
        #     tit = str2tex(tit, usetex=self.usetex)
        ax1.set_title(f'{tit}')

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
    
    sites=[ "AU-ASM",  "US-SRM" ,"AU-Gin" ,"US-Me6", "AU-How", "FR-Pue", "IT-Cp2" ,"ZM-Mon", "IT-Ro1" ,"IT-SRo",
 "CN-Qia", "FR-LBr","GF-Guy", "MY-PSO", "CA-Qfo", "CA-SF2", "CA-TPD",
 "SE-Nor" ,"CN-Cng" ,"CN-Dan"]
    #sites = [ "AU-ASM","AU-Gin" , "AU-How","ZM-Mon"]
    sites = ["US-SRM" , # "CA-SF2","IT-Cp2" ,
        "FR-LBr", #"CN-Qia",
            ]
    drydown = True

    suffmod = '/outputs/site_out_cable_*.nc'
    if platform.node()=='zlhp':
        
        fp1='/mnt/e/biocomp/test1105/'
        fpobs = '/mnt/d/project/hydraulic/test/obs4/'
        fmet = '/mnt/d/project/hydraulic/test/input/met/'
        fout = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/'

        modnames = ['psix_StypeOri_wbpsdo','psix_Stype2_wbpsdo','psix_Stype3_wbpsdo']
        sn = ['clay loam Ori wbpsdo','clay loam wbpsdo','clay wbpsdo']


    else:
        fp1='/home/zlu/project/test1105/'
        fpobs='/home/mcuntz/projects/coco2/obs4/'
        fmet = '/home/mcuntz/projects/coco2/cable-pop/input/'
        fout='/home/zlu/project/test1105/code/'
    if drydown:
        segs = [4]
        #outfile = f"{fout}result_LP_drydown_BASEdc98fe39.xlsx"
        monthss = [1,2,3,12]
        monthnn = [6,7,8,9]
       
    else:
        segs = [3]
        #outfile = f"{fout}result_LP_BASEdc98fe39_AUsites.xlsx"
        monthss = [1,2,3,4,11,12]
        monthnn = [5,6,7,8,9]
     
    
    varlab0,sn=['tuzet_LWP_Haverd2013/','tuzet']
    outfile = f"{fout}result_BASEdc98fe39.xlsx"

    snd = {
       'fine clay': 'BASEdc98fe39_Stype3_Vtype1',
    #     'silty clay': 'BASEdc98fe39_Stype6_Vtype1',
    #    'clay loam': 'BASEdc98fe39_Stype2_Vtype1',
    #     'sandy clay': 'BASEdc98fe39_Stype5_Vtype1',
    #     'sandy clay loam':'BASEdc98fe39_Stype7_Vtype1',
    #     'Sandy loam': 'BASEdc98fe39_Stype4_Vtype1',
         'sand':'BASEdc98fe39_Stype1_Vtype1',
            }
    tag = 'EF'



    
    vars=['GPP','Qle']     
    varread = ['Rainf','GPP','TVeg','Qle','Qh','gsw_sl',
                'SoilMoistPFT',
                'psi_soilmean','wb_soilmean','rwc_soilmean_recal',
                'wb_30','rwc_30','psi_30',
                'wb_fr_rootzone','rwc_fr_rootzone','psi_fr_rootzone',
                'wb_depth_rootzone','rwc_depth_rootzone','psi_depth_rootzone',
                'epotcan3','ksoilmean','krootmean','kplant','kbelowmean',
                'gsw_epotcan3_sl','GPP_epotcan3','ksoil','kroot']
    suffobs = '_flux.nc'
    varsoil = {
    #'wb_soilmean': 'mean soil water content (m m$^{-1}$)',
    #'rwc_soilmean': 'relative water content',
    #'rwc_soilmean_recal': 'relative water content (recal)',
    'psi_soilmean': 'mean soil water potential (MPa)',
    #'wb_30': 'mean soil water content in 30 cm (m m$^{-1}$)',
    #'rwc_30': 'relative water content in 30 cm (m m$^{-1}$)',
    # 'psi_30':'mean soil water potential in 30 cm (Mpa)',
    # 'wb_fr_rootzone':'mean soil water content in rootzone weighted by root fraction(m m$^{-1}$)',
    # 'rwc_fr_rootzone':'relative water content in rootzone weighted by root fraction(m m$^{-1}$)',
    # 'psi_fr_rootzone':'mean soil water potential in rootzone weighted by root fraction(Mpa)',
    # 'wb_depth_rootzone':'mean soil water content in rootzone weighted by layer depth(m m$^{-1}$)',
    # 'rwc_depth_rootzone':'relative water content in rootzone weighted by layer dept(m m$^{-1}$)',
    # 'psi_depth_rootzone':'mean soil water potential in rootzone weighted by layer dept(Mpa)',
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
    fitnames = ['linear-plateau']
    methods = ['None','DE','DA','SH']

    calc_methods = ['percentile']
    
    for isite in sites:
        print(f"site name: {isite}")
        fullnames = []
        shortnames = []
        for ikey,imod in snd.items():
            #fn = glob.glob(f'{fp1}{isite}/{varlab0}/{imod}/outputs/site_out_cable_*_daily.nc')
            fn_all = glob.glob(f'{fp1}{isite}/{varlab0}/{imod}/outputs/site_out_cable_*.nc')
            fn = [f for f in fn_all if not f.endswith('daily.nc')]
            # fn_all = glob.glob(f'{fp1}{isite}/tuzet_LWP_Haverd2013/{imod}/outputs/site_out_cable_*.nc')
            # print(fn_all)
            # fn = [f for f in fn_all if not f.endswith('daily.nc')]
            if not fn:  # same as: if len(fn) == 0
                continue  # skip this iteration
            match = re.search(r'_([0-9]+)_([0-9]+)', fn[0])
            yr_st = int(match.group(1))
            yr_en = int(match.group(2))
            fullnames += fn
            shortnames += [ikey]
        end1    = ptime.time()
        if not fullnames:
            continue
        print(f"get modname {end1 - t1:.1f} seconds")
        # fullnames+=[fpobs  +isite + suffobs]
        # shortnames+=['obs']
        # print(f'   fullname {fullnames}')
        # print(f'   shortname {shortnames}')

        self.read_data_notCommand(fullnames,shortnames,varread)
        x_lim = [f"{yr_st+3}-01-01",f"{yr_en}-12-30"]
        x_lim = [f"{yr_st}-01-01",f"{yr_en}-12-30"]
        if isite == 'AU-Cum':
            x_lim = ["2013-01-01","2014-12-30"]
        if isite == 'CN-Qia':
            x_lim = [f"{yr_st}-01-01",f"{yr_en}-12-30"]

        end2    = ptime.time()
        print(f"read data {end2 - end1:.1f} seconds")
        result = []

        for isn in shortnames:
            print(f"soil types: {isn}")

            ds = getattr(self,isn)

            # sand= ds['sand'].values.flatten()[0]
            # clay = ds['clay'].values.flatten()[0]
            # silt = ds['silt'].values.flatten()[0]
            tit=f'{isite}: {isn}'
            self.tit = tit

            ds['Ktotal'] = 1/(1/ds['kbelowmean']+1/ds['kplant'])
            ds['Ratio_ksoil_kroot'] = ds['ksoilmean']/ds['krootmean']
            ds['Ratio_ksoil_kroot2D'] = ds['ksoil']/ds['kroot']
            target = ['kbelowmean','kplant','Ktotal']
            target = ['Ratio_ksoil_kroot2D']

           
            for ix in varsoil:
                Y = ds[target + [ix]]
                print(f"soil var: {ix}")
                
                for imethod in calc_methods:
                    for iseg in segs:
                        print(f'segID: {iseg}')
                        if drydown:
                            ds['Rainf'] =  ds['Rainf'] * 86400
                            times = self.find_drydown_period(isite,ds,x_lim)
                            Y_new=self.calc_xx_yy_cc(isite,Y,x_lim = times)
                        else:
                            Y_new=self.calc_xx_yy_cc(isite,Y,clcvar=ds['epotcan3'],x_lim = x_lim)
                        x_lim1 = [0.001, 2]
                        y_lim1 = [1e-5,1e-3]
                        y_lim1 = [0,20]

                        
                        # iplot = self.plot_scatter(-Y_new[ix],Y_new[target],iplot,oneline=False,ifcolor=False,iffit = True,tit =tit,
                        #                           y_lim =y_lim1,x_lim=x_lim1,xlog=True,ylog=True )
                        iplot = self.plot_scatter(-Y_new[ix],Y_new[target],iplot,oneline=False,ifcolor=False,iffit = True,tit =tit,y_lim =y_lim1,x_lim=x_lim1,)
    if iplot>0:
        self.plot_save(self.fig)



    self.close()


    t2    = ptime.time()
    strin = ( '[m]: {:.1f}'.format((t2 - t1) / 60.)
              if (t2 - t1) > 60.
              else '[s]: {:d}'.format(int(t2 - t1)) )
    print('    Time elapsed', strin)