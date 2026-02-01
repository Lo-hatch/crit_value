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
import seaborn as sns
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
      

        self.nrow     = 3     # # of rows of subplots per figure
        self.ncol     = 1    # # of columns of subplots per figure
        self.hspace   = 0.04  # x-space between subplots
        self.vspace   = 0.02  # y-space between subplots
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
        
        # Increase font sizes
        plt.rcParams.update({
            "font.size": 14,            # Default global font size
            "axes.labelsize": 16,       # Axis label size
            "axes.titlesize": 16,       # Axis title size
            "xtick.labelsize": 14,      # X-axis tick label size
            "ytick.labelsize": 14,      # Y-axis tick label size
            "legend.fontsize": 14,      # Legend text size
        })



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

    def calc_xx_yy_cc(self,isite,xx,yy,clcvar,method,iseg,x_lim = None):
        if (clcvar is None):
            print("!!!!! clcvar or clcname is lack for color scatter")
            return  # Exit the function early

        if ('AU' in isite) or ('ZM' in isite):
            months = [1,2,3,4,11,12]
        else:
            months = [5,6,7,8,9]
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
    def plot_scatter(self, xx,yy,xlab,ylab,fitname, iplot, oneline=False, ifcolor=True, iffit = False,clcvar=None, clcpro = None, x_lim=None,y_lim=None, tit=None, fitDirec=True, GM_method=None,ifsigma=False):

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
       
        xcmx,xccmx1,xccmn1,r,rmse,xcons1,xcons2,xcons3,xcons3K,xslope1,xslope2,xslope3 = [np.nan] * 12
        if ifcolor:
            if clcvar is None:
                print("Error: Input clcvar cannot be None.", file=sys.stderr)
                sys.exit(1)
            if clcpro is None:
                print("Error: Input clcpro cannot be None.", file=sys.stderr)
                sys.exit(1)
            
            my_pf.plot_scatter_cmap(xx,yy,clcvar,xlab,ylab,clcpro = clcpro, sz = 8,clcbarpad = 0.06,clcbarloc='top',ax1=ax1,fig = fig,ifcolorbar=False)
        else:
            cmap = plt.get_cmap('RdYlBu_r', 4)
            cmap = cmap(np.linspace(0, 1, 4))  # Now this is a (4, 4) array of RGBA
            cmap = ListedColormap(cmap)
            colors = sns.color_palette("Set2", 4)
            cmap = ListedColormap(colors)
            scatter_handles = []
            for i, (xx, yy) in enumerate(zip(X, Y)):
                lab = sites[i]
                sc = ax1.scatter(xx,yy, facecolors='none',edgecolors=cmap(i), s=9, label=lab)
                scatter_handles.append(sc)
        xpos = 0.57
        xinte = 0.12
        if iffit:

                #r_squared1, rmse1, x_fit1,y_fit1,xcmx1,xcmn1, xccmx1,xccmn1= my_af.logistic_fitting(xx[mask].values,yy[mask].values)
            
            for i, (xx, yy) in enumerate(zip(X, Y)):
                lab = sites[i]
                if lab in ['US-SRM','IT-Cp2']:
                    ifsigma=True
                else:
                    ifsigma=False
                if fitname=='logi':
                    popt,r, rmse, x_fit,y_fit,slopes,best,y2,bestylab,y_3d,x_3d,y_c,xcmx,xcmn1,y_cc, xccmx1,xccmn1,dycmx,dyccmx,dyccmn,xcons1,xcons2,xcons3,xslope1,xslope2,xslope3= my_af.logistic_fitting(xx,yy,fitDirec=fitDirec,GM_method=GM_method,ifsigma=ifsigma)
                if fitname =='gomp':
                    r, rmse, x_fit,y_fit,best,y2,bestylab,y_3d,x_3d,y_c,xcmx,xcmn1,y_cc, xccmx1,xccmn1,dycmx,dyccmx,dyccmn,xcons1,xcons2,xcons3,xslope1,xslope2,xslope3= my_af.gompertz_fitting(xx,yy,fitDirec=fitDirec,GM_method=GM_method)
                if fitname == 'Dsig':
                    r, rmse, x_fit,y_fit,y_c,slopes,xcmxl,xccmx1,xcmx, dycmx,dyccmx,xcons1,xcons2,xcons3,xslope1,xslope2,xslope3= my_af.double_sigmoid_fitting(xx,yy,fitDirec=fitDirec,GM_method=GM_method)
                if fitname == 'Sexpo':
                    r, rmse,x_fit,y_fit,y_3d,x_3d,y_c,xcmx,xcmn1,y_cc,xccmx1,xccmn1,xcons1,xcons2,xcons3,xslope1,xslope2,xslope3 = my_af.exponential_strench_decrease_fitting(xx,yy,fitDirec=fitDirec,GM_method=GM_method)
                y_fit_kernel,xcons3K =my_af.kernelRegression(xx,yy)
                value = xslope2
                line1,=ax1.plot(x_fit, y_fit, color=cmap(i),linestyle='-',label='fitting line',linewidth=1.5)
                line2=ax1.axvline(x=value, color=cmap(i), linestyle=':',label='maximum curvature')
                if r<0.01:
                    txt = '0.0'
                else:
                    txt = f'{my_pf.numstrFormat(r,2)}'
                txt =  f'{lab}: {my_pf.numstrFormat(value,2)}, R2={txt}, RMSE={my_pf.numstrFormat(rmse,2)}'
                txt =  f'{lab}: {my_pf.numstrFormat(value,2)}'
                
                #txt =  f'{lab}: Max Curvature Change={my_pf.numstrFormat(xccmx1,3)}'
                ax1.text(xpos,0.9-i*xinte,txt ,verticalalignment='bottom', horizontalalignment='left',
        transform=ax1.transAxes,color=cmap(i), fontsize=14)
            xx = np.concatenate(X)
            yy = np.concatenate(Y)
            ifsigma = False
            if fitname=='logi':
                popt,r, rmse, x_fit,y_fit,slopes,best,y2,bestylab,y_3d,x_3d,y_c,xcmx,xcmn1,y_cc, xccmx1,xccmn1,dycmx,dyccmx,dyccmn,xcons1,xcons2,xcons3,xslope1,xslope2,xslope3= my_af.logistic_fitting(xx,yy,fitDirec=fitDirec,GM_method=GM_method,ifsigma=ifsigma)
            if fitname =='gomp':
                r, rmse, x_fit,y_fit,best,y2,bestylab,y_3d,x_3d,y_c,xcmx,xcmn1,y_cc, xccmx1,xccmn1,dycmx,dyccmx,dyccmn,xcons1,xcons2,xcons3,xslope1,xslope2,xslope3= my_af.gompertz_fitting(xx,yy,fitDirec=fitDirec,GM_method=GM_method)
            if fitname == 'Dsig':
                r, rmse, x_fit,y_fit,y_c,slopes,xcmxl,xccmx1,xcmx, dycmx,dyccmx,xcons1,xcons2,xcons3,xslope1,xslope2,xslope3= my_af.double_sigmoid_fitting(xx,yy,fitDirec=fitDirec,GM_method=GM_method)
            if fitname == 'Sexpo':
                r, rmse,x_fit,y_fit,y_3d,x_3d,y_c,xcmx,xcmn1,y_cc,xccmx1,xccmn1,xcons1,xcons2,xcons3,xslope1,xslope2,xslope3 = my_af.exponential_strench_decrease_fitting(xx,yy,fitDirec=fitDirec,GM_method=GM_method)
            y_fit_kernel,xcons3K =my_af.kernelRegression(xx,yy)
            value = xslope2
            line1,=ax1.plot(x_fit, y_fit, color='k',linestyle='-',label='fitting line',linewidth=1.5)
            line2=ax1.axvline(x=value, color='k', linestyle=':',label='maximum curvature')
            #line2a=ax1.axhline(y = 0.05, color=light_grey, linestyle='--')
            if r<0.01:
                txt = '0.0'
            else:
                txt = f'{my_pf.numstrFormat(r,2)}'
            txt =  f'whole: {my_pf.numstrFormat(value,2)}, R2={txt}, RMSE={my_pf.numstrFormat(rmse,2)} '
            txt =  f'whole: {my_pf.numstrFormat(value,2)}'
           

            ax1.text(xpos,0.9-(i+1)*xinte,txt ,verticalalignment='bottom', horizontalalignment='left',
    transform=ax1.transAxes,color='k', fontsize=14)
            # elif fitname=='tanh':
            #     r, rmse, x_fit,y_fit,y_3d,x_3d,xcmx = my_af.tangent_fitting(xx,yy)
            # elif fitname=='Dtanh':
            #     r, rmse, x_fit,y_fit,y_3d,x_3d,xcmx = my_af.tangent_double_fitting(xx,yy)
            # elif fitname=='expo':
            #     r, rmse, x_fit,y_fit,y_3d,x_3d,xcmx,xcmn2,xccmx2,xccmn2 = my_af.exponential_fitting(xx,yy) 
            # elif fitname=='Sexpo':
            #     r, rmse,x_fit,y_fit,y_3d,x_3d,y_c,xcmx,xcmn3,xccmx3,xccmn3 = my_af.exponential_strench_decrease_fitting(xx,yy,fitDirec=fitDirec,GM_method=GM_method)
            

                # txt = f'{my_pf.numstrFormat(r_squared1,2)}, {my_pf.numstrFormat(rmse1,2)} Vs {my_pf.numstrFormat(r_squared2,2)}, {my_pf.numstrFormat(rmse2,2)}'
            



            # txt = f'{my_pf.numstrFormat(r_squared1,2)} Vs {my_pf.numstrFormat(r_squared2,2)} Vs {my_pf.numstrFormat(r_squared3,2)}'            
            # ax1.text(0.02,0.9-i*0.1, txt ,verticalalignment='bottom', horizontalalignment='left',
            #         transform=ax1.transAxes,color=cmap(i), fontsize=7)
            # best = xccmx1
            # y2 = y_cc

           # line3=ax1.axvline(x=xccmx1, color=light_grey, linestyle='--',label='maximum curvature change')
           # line4=ax1.axvline(x=xccmn1, color=light_grey, linestyle=':',label='minimum curvature change')
            #ax1.axvline(x=x_3d, color=dark_grey, linestyle=':')
            # # ax1.text(xc, plt.ylim()[0], f'{my_pf.numstrFormat(xc,2)}', color=cmap(i), ha='center', va='top')
            # ax1.text(best, plt.ylim()[0], f'{my_pf.numstrFormat(best,2)}', color=cmap(i), ha='center', va='top')

            # ax1.text(0.02+0*0.16+0.02,0.9-2*0.1, f'Max curvature change={my_pf.numstrFormat(xccmx1,2)}' ,verticalalignment='bottom', horizontalalignment='left',
            # transform=ax1.transAxes,color=clc, fontsize=6)
            # ax1.text(0.02+0*0.16+0.02,0.9-3*0.1, f'Min curvature change={my_pf.numstrFormat(xccmn1,2)}' ,verticalalignment='bottom', horizontalalignment='left',
            # transform=ax1.transAxes,color=clc, fontsize=6)


                
        # else:
        #     if isinstance(xx, xr.DataArray):
        #         xx = xx.values
        #         yy = yy.values
        #     my_pf.plot_scatter(ax1,xx,yy,xlab,ylab,oneline=oneline)
        if x_lim is not None:
            ax1.set_xlim(x_lim) 
        if y_lim is not None:
            ax1.set_ylim(y_lim) 
        #ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
        # xx0 = xx[np.abs(yy) < 0.01]
        # yy0 = yy[np.abs(yy) < 0.01]

        # if len(xx0)>0:
        #     # 2. Plot those data points on top of the boxplot (as red dots)
        #     ax1.scatter(xx0,yy0, facecolors='red',edgecolors='none', s=9)
        #     # 3. Calculate mean and median
        #     near_mean = xx0.mean()
        #     near_median = np.median(xx0)

        #     # 4. Annotate on the plot
        #     text = f"mean={my_pf.numstrFormat(near_mean,2)}\nmedian={my_pf.numstrFormat(near_median,2)}"
        #     ax1.text(0.99, 0.95, text, transform=ax1.transAxes, ha='right', va='top', fontsize=8, bbox=dict(facecolor='white', edgecolor='gray'))
           
        # if fitname!='logi':
        #     ax1.set_yticklabels([])
        
        plt.setp(ax1, xlabel=xlab,ylabel=ylab)
        
        #ax2.set_yticklabels([])
        # plt.setp(ax1,x_lim=[0,0.5],y_lim=[-0.01,0.7])
        abc2plot(ax1, self.dxabc, self.dyabc,
                (self.ifig - 1) * nplotperpage + iplot,
                lower=True, bold=True, usetex=self.usetex, mathrm=True,large=True)
        # if iplot==1:
        #     lines = [line1, line2]
        #     labels = [line.get_label() for line in lines]
        # ll = ax1.legend(handles=scatter_handles,frameon=self.frameon, ncol=1,
        #                 labelspacing=self.llrspace,
        #                 handletextpad=self.llhtextpad,
        #                 handlelength=self.llhlength,
        #                 loc='upper left',
        #                 bbox_to_anchor=(self.llxbbox, self.llybbox),
        #                 scatterpoints=1, numpoints=1,fontsize =7)
        #     plt.setp(ll.get_texts(), fontsize='small')
            #fig.suptitle(self.tit)
        # abc2plot(ax1, self.dxabc, self.dyabc,
        #         (self.ifig - 1) * nplotperpage + iplot,
        #         lower=True, bold=True, usetex=self.usetex, mathrm=True)
        # if tit is not None:
        #     tit = str2tex(tit, usetex=self.usetex)
        plt.title(f'{tit}')
        if ylab == 'Rel_limitation_wb':
            idx = np.argmin(np.abs(y_fit - 0.5))
            # Get corresponding x
            xcmx = x_fit[idx]
            xccmx1 = xcmx
        if (iplot == nplotperpage):
        # save one pdf page, zihanlu
            self.plot_save(fig)
            iplot = 0
        #return iplot,xcmx,xccmx1,xccmn1,r,rmse,dycmx,dyccmx,dyccmn
        return iplot,xcmx,xccmx1,xccmn1,r,rmse,xcons1,xcons2,xcons3,xslope1,xslope2,xslope3,xcons3K
    
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
    
    sites=['AU-Gin','ZM-Mon','AU-ASM','IT-Cp2','IT-SRo','FR-LBr','IT-Ro1','FR-Pue','CN-Qia','AU-How',]
    sites=['IT-Ro1','FR-Pue']
    sites=[ "CA-Qfo","CN-Dan","AU-How", "FR-Pue", "IT-Cp2" ,"ZM-Mon", "IT-Ro1" ,"IT-SRo",
 "CN-Qia", "FR-LBr","GF-Guy", "MY-PSO", "CA-SF2", "CA-TPD",
 "SE-Nor" ,"CN-Cng" ]
    sites=["AU-ASM", "US-SRM" ,"CA-SF2",
           "AU-Gin" ,
           "US-Me6","CN-Cng" ,"ZM-Mon","IT-Cp2" ,
           "IT-SRo","IT-Ro1" ,"MY-PSO", 
           "AU-How", "FR-LBr","FR-Pue","CA-TPD", "CN-Qia", "SE-Nor","GF-Guy", ]
    sites=[ 'AU-ASM',"CN-Cng" ,'MY-PSO',"FR-Pue"]
    sites=[ 
        'US-SRM',
         #'CA-SF2',
          'IT-Cp2',
       "FR-LBr" ,'CN-Qia'
           ]
    sitetag = 'UIFC'
    #sites=["FR-Pue","AU-ASM","GF-Guy","CN-Cng" ,]
    #sites=["AU-ASM","FR-LBr","MY-PSO","SE-Nor"]
    # IT-Cp2, AU-Gin FR-LBr-sand,SE-Nor GF-Guy MY-PSO CA-Qfo CA-SF2 CA-TPD CN-Cng
    suffmod = '/outputs/site_out_cable_*.nc'
    if platform.node()=='zlhp':
        
        fp1='/mnt/e/biocomp/test1105/'
        fpobs = '/mnt/d/project/hydraulic/test/obs4/'
        fmet = '/mnt/d/project/hydraulic/test/input/met/'
        fout = '/mnt/d/project/hydraulic/test/plot/f05_test_Ratio/'
        ffigout = '/mnt/d/project/hydraulic/test/plot/fig_final/ppt0129/'


    else:
        fp1='/home/zlu/project/test1105/'
        fpobs='/home/mcuntz/projects/coco2/obs4/'
        fmet = '/home/mcuntz/projects/coco2/cable-pop/input/'
        fout = '/home/zlu/project/test1105/plot/f03_test_fitting/'

    varlab0,sn=['tuzet_LWP_Haverd2013/','tuzet']
    outfile = f"{fout}result_BASEdc98fe39.xlsx"
    snd = {
    #  'fine clay': 'BASEdc98fe39_Stype3_Vtype1',
    #  'silty clay': 'BASEdc98fe39_Stype6_Vtype1',
      'clay loam': 'BASEdc98fe39_Stype2_Vtype1',
    #   'sandy clay': 'BASEdc98fe39_Stype5_Vtype1',
    #   'sandy clay loam':'BASEdc98fe39_Stype7_Vtype1',
    #   'Sandy loam': 'BASEdc98fe39_Stype4_Vtype1',
    #   'sand':'BASEdc98fe39_Stype1_Vtype1',
            }
    tag = f'{sitetag}_NTD'
    # snd = {
    # 'fine clay': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype3_Vtype1',
    # 'silty clay': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype6_Vtype1',
    # 'clay loam': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype2_Vtype1',
    # 'sandy clay': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype5_Vtype1',
    # 'sandy clay loam':'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype7_Vtype1',
    # 'Sandy loam': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype4_Vtype1',
    # 'sand':'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype1_Vtype1',
    #    }
    # tag = f'{sitetag}LBrLAI'
    # snd = {
    #     'fine clay': 'BASEdc98fe39_Stype3_Vtype1',
    # 'fine clay LBr': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype3_Vtype1',
    # 'silty clay': 'BASEdc98fe39_Stype6_Vtype1',
    # 'silty clay LBr': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype6_Vtype1',
    # 'clay loam': 'BASEdc98fe39_Stype2_Vtype1',
    # 'clay loam LBr': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype2_Vtype1',
    # 'sandy clay': 'BASEdc98fe39_Stype5_Vtype1',
    # 'sandy clay LBr': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype5_Vtype1',
    # 'sandy clay loam':'BASEdc98fe39_Stype7_Vtype1',
    # 'sandy clay loam LBr':'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype7_Vtype1',
    #  'Sandy loam': 'BASEdc98fe39_Stype4_Vtype1',
    # 'Sandy loam LBr': 'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype4_Vtype1',
    # 'sand':'BASEdc98fe39_Stype1_Vtype1',
    # 'sand LBr':'Ratio_maximum_Bfr2_fwpsimk5_LAImax_LBr_Stype1_Vtype1',
    #    }
    #snd = {k: v + "_1990_2021" for k, v in snd.items()}
    
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
    'wb_soilmean':  r'$\theta$ (m$^3$ m$^{-3}$)',
    #'rwc_soilmean': 'relative water content',
    # 'rwc_soilmean_recal': 'relative water content (recal)',
    # 'psi_soilmean': 'mean soil water potential (MPa)',
    # 'wb_30': 'mean soil water content in 30 cm (m m$^{-1}$)',
    # 'rwc_30': 'relative water content in 30 cm (m m$^{-1}$)',
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
    dark_grey = "#4F4E4D60"  # Dark grey
    black = (0.0, 0.0, 0.0)       # Black
    iplot = 0


# Extract the two 'xxxx' parts before the .nc extension
    fitnames = ['logi','tanh','Dtanh','expo','Sexpo']
    fitnames = ['logi','Sexpo']
    methods = ['None','DE','DA','SH']

    calc_methods = ['percentile']

    plot_config = {

        # '(Edemand - TVeg)/TVeg': {
        #     'var': (epotcan3-Tveg)/Tveg,
        #     'ylim': (-0.55, 0.6)
        # },
        '(Edemand - TVeg)/Edemand': {
            #'var': (epotcan3-Tveg)/epotcan3,
            'ylab': r'$\frac{(E_{sat}-E)}{E_{sat}}$',
            'ylim': (-0.55, 1.5)
        },
        # '(GPP_wb - GPP)/GPP_wb': {
        #    'var': (GPP_epotcan3-GPP)/GPP_epotcan3,
        #    'ylab': r'$\frac{(GPP_{sat}-GPP)}{GPP_{sat}}$',
        #    'ylim': (-0.55, 1.5)
        # },
        # 'Edemand - TVeg': {
        #     'var': (epotcan3-Tveg),
        #     'ylim': None
        # },
        # '(gsw_sw - gs)/gsw_sw': {
        #    'var': (gs_epotcan3-gs)/gs_epotcan3,
        #    'ylab': r'$\frac{(gs_{sat}-gs)}{gs_{sat}}$',
        #    'ylim': (-0.55, 0.6)
        # },      
        # '(Evpd - TVeg)/TVeg': {
        #     'var': (epotvpd-Tveg)/Tveg,
        #     'ylim': (-0.55, 0.6)
        # },
        # '(Evpd - TVeg)/Evpd': {
        #     'var': (epotvpd-Tveg)/epotvpd,
        #     'ylim': (-0.55, 1.5)
        # },
        # 'Evpd - TVeg': {
        #     'var': (epotvpd-Tveg),
        #     'ylim': None
        # },          
        # '(gsw_vpd - gs)/gsw_vpd': {
        #     'var': (gs_epotvpd-gs)/gs_epotvpd,
        #     'ylim': (-0.55, 0.6)
        # },
        # '(gsw_sw - gs)/gsw_sw - (gsw_vpd - gs)/gsw_vpd': {
        #     'var': (gs_epotcan3-gs)/gs_epotcan3 - (gs_epotvpd-gs)/gs_epotvpd,
        #     'ylim': (-0.55, 0.6)
        # },   
        # 'Rel_limitation_wb': {
        #     'var': Rsw,
        #     'ylim': (-0.55, 0.6)
        # },  

    }
    result = []
    for ylabel, config in plot_config.items():


        print(f'yvar: {ylabel}')

        for imethod in calc_methods:

            for iseg in [3]:
                for ikey,imod in snd.items():
                    fullnames = []
                    shortnames = []
                    for isite in sites:

                        fn_all = glob.glob(f'{fp1}{isite}/{varlab0}/{imod}/outputs/site_out_cable_*.nc')
                        fn = [f for f in fn_all if not f.endswith('daily.nc')]
                        #fn = glob.glob(f'{fp1}{isite}/{varlab0}/{imod}/outputs/daily_1990_2021.nc')
                        # fn_all = glob.glob(f'{fp1}{isite}/tuzet_LWP_Haverd2013/{imod}/outputs/site_out_cable_*.nc')
                        print(f'{fp1}{isite}/{varlab0}/{imod}/outputs/site_out_cable_*.nc')
                        # fn = [f for f in fn_all if not f.endswith('daily.nc')]
                        
                        if not fn:  # same as: if len(fn) == 0
                            print(f'filename: {fn}')
                            continue  # skip this iteration
                        print(f'filename: {fn}')
                        if isinstance(fn, str):
                            match = re.search(r'_([0-9]+)_([0-9]+)', fn)
                            fullnames += [fn]
                        else:
                            match = re.search(r'_([0-9]+)_([0-9]+)', fn[0])
                            fullnames += fn
                        yr_st = int(match.group(1))
                        yr_en = int(match.group(2))
                        shortnames += [isite]


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
                    # if isite == 'CN-Qia':
                    #     x_lim = [f"{yr_st}-01-01",f"{yr_en}-12-30"]

        
                    end2    = ptime.time()
                    print(f"read data {end2 - end1:.1f} seconds")
                    

                    for ix, xlab in varsoil.items():
                        X = []
                        Y = []
                        print(f"soil var: {ix}")
                        for isn in shortnames:
                            print(f"site: {isn}")

                            ds = getattr(self,isn)
                            clcvar = ds['epotcan3']
                            xvar = ds[ix]
                            ylab = config['ylab']
                            if 'Edemand' in ylabel:
                                yvar =(ds['epotcan3'] -  ds['TVeg'])/ds['epotcan3']
                            elif 'GPP_wb' in ylabel:
                                yvar =(ds['GPP_epotcan3'] -  ds['GPP'])/ds['GPP_epotcan3']
                            elif 'gsw_wb' in ylabel:
                                yvar =(ds['gsw_epotcan3_sl'] -  ds['gsw_sl'])/ds['gsw_epotcan3_sl']

                            xx,yy,cc,clcpro=self.calc_xx_yy_cc(isn,xvar,yvar,clcvar,method=imethod,iseg=iseg,x_lim = x_lim)
                            X.append(xx)
                            Y.append(yy)
                        if ix=='wb_soilmean':
                            if  ikey in ['fine clay']:
                                x_lim1=(0.18,0.46)
                            elif ikey == 'sandy clay':
                                x_lim1=(0.2,0.4)
                            else:
                                x_lim1=(0,0.35)
                        elif 'rwc_' in ix:
                            x_lim1 = (-0.2,1)   
                        x_lim1 = None    
                        if iseg<4:
                            tit = f'{ikey}: seg{iseg+1}'
                        else:
                            tit = f'{ikey}: {imethod}:whole'       
                        tit = '' 
               
                        iplot,xcmx,xccmx,xccmn,r,rmse,xcons1,xcons2,xcons3,xslope1,xslope2,xslope3,xcons3K = self.plot_scatter(X,Y,'',ylab,'logi',
                            iplot,oneline=False,ifcolor=False,iffit = True,clcvar=clcvar,clcpro=clcpro,tit =tit ,y_lim=(0,1),x_lim = x_lim1)
             
                        result.append([ix,ikey,ylabel,imethod,iseg,xcmx,xccmx,xccmn,r,rmse,xcons1,xcons2,xcons3,xslope1,xslope2,xslope3,xcons3K])
    # df = pd.DataFrame(result, columns=["Xname", "soiltype","yvar","method",'segID',"MaxCur_logi","MaxCurChange_logi","MinCurChange_logi","MaxCur_gomp","MaxCurChange_logi","MinCurChange_logi",
    #                                     "MaxCur_dsig","MaxCurR_dsig"])
    # df = pd.DataFrame(result, columns=["Xname", "soiltype","yvar","method",'segID',"MaxCur","MaxCurChange","MinCurChange","R2","RMSE","cons0p03","cons0p04","cons0p05",'slope0p05','slope0p1','slope0p15',"cons0p05Kernel"])
    
    # if not os.path.exists(outfile):
    #     # Create a new file
    #     with pd.ExcelWriter(outfile, mode="w", engine="openpyxl") as writer:
    #         df.to_excel(writer, sheet_name=f"{tag}", index=False)
    # else:
    #     with pd.ExcelWriter(outfile, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    #         df.to_excel(writer, sheet_name=f'{tag}', index=False)
            #gsref = ds['gsw_ref_sl']
            #Rsw = 1-(gsref-gs_epotcan3)/(gsref-gs)
            #Rsw = Rsw.where((Rsw >= 0) & (Rsw <= 1), np.nan)

            #################### different y


            # for iyr in range(yr_st+3, yr_en + 1):
            #     x_lim = [f"{iyr}-01-01",f"{iyr}-12-30"]
            #     iplot,xx,yy = self.plot_scatter(isite,xlablist[xindex],'epotcan3 - TVeg (Kg m-2 s-1)',xvarlist[xindex],delta,frq,iplot,False,True,clcvar,clcname,x_lim=x_lim,tit=f'{iyr}:May-Sep')
            #     #iplot,xx,yy = self.plot_scatter(isite,'SWdown (W m-2)','epotcan3 - TVeg (Kg m-2 s-1)',Rs,delta,frq,iplot,False,x_lim=x_lim,tit=f'{iyr}:May-Sep')
            #     Rstmp = Rs.sel(time=slice(x_lim[0],x_lim[1]))
            #     x_lim = Rstmp.time[Rstmp>250]
            #     iplot,xx,yy = self.plot_scatter(isite,xlablist[xindex],'epotcan3 - TVeg (Kg m-2 s-1)',xvarlist[xindex],delta,frq,iplot,False,True,clcvar,clcname,x_lim=x_lim,tit=f'{iyr}:May-Sep (Rs>250)')
            

            ######################### test over colored scatter#########################
            # self.nrow     = 6
            # for ylabel, config in plot_config.items():
            #     for ix in varsoil:
            #         print(f"soil var: {ix}")
            #         print(f'yvar: {ylabel}')
            #         xvar = dssoil[ix]
            #         yvar = config['var']
            #         ylim = config['ylim']
            #         for imethod in calc_methods:
            #             xx,yy,cc,clcpro=self.calc_xx_yy_cc(isite,xvar,yvar,clcvar,method=imethod,iseg=4)
            #             clcpro['clab'] = r'Edemand (kg $m^{-2}$ $s^{-1}$)'
            #             iplot,xcmx,xccmx,xccmn = self.plot_scatter(xx,yy,ix,config['ylab'],'logi',iplot,oneline=False,ifcolor=True,iffit = True,clcvar=cc,clcpro=clcpro,tit =tit ,fitDirec=True)


            ################################### test over fitting function #################################
            # xx,yy=self.calc_xx_yy(isite,xvar,yvar,clcvar,method='Equal',iseg=3)
            # for i,imethod in enumerate(methods):
            #     for ifit in fitnames:
            #     # iplot,xx,yy = self.plot_scatter(isite,xlablist[xindex],'epotcan3 - TVeg (Kg m-2 s-1)',xvarlist[xindex],delta,frq,iplot,False,True,clcvar,clcname,x_lim=x_lim,tit=f'{isite}:{yr_st+3}-{yr_en}:GS')
            #         if ifit != 'logi':
            #             tit0 = None
            #         else:
            #             tit0 = tit
            #         #iplot,xx,yy = self.plot_scatter(isite,ix,'Edemand - TVeg',dssoil[ix],delta,frq,ifit,iplot,oneline=False,ifcolor=True,iffit = True,clcvar=clcvar,clcpro=clcpro,x_lim=x_lim,tit = tit0)
            #         if i==0:
            #             iplot = self.plot_scatter(xx,yy,ix,ylab,ifit,iplot,oneline=False,ifcolor=True,iffit = True,clcvar=clcvar,clcpro=clcpro,x_lim=x_lim,tit = f'{ifit}:curvefit',fitDirec=True)
            #         else:
            #             iplot = self.plot_scatter(xx,yy,ix,ylab,ifit,iplot,
            #             oneline=False,ifcolor=True,iffit = True,clcvar=clcvar,clcpro=clcpro,x_lim=x_lim,tit = f'{ifit}:{imethod}',fitDirec=False,GM_method=imethod)
            # #iplot,xx,yy = self.plot_scatter(isite,ix,'TVeg',dssoil[ix],Tveg,frq,iplot,oneline=False,ifcolor=True,iffit = True,clcvar=clcvar,clcpro=clcpro,x_lim=x_lim,tit=tit)
            ############################### test over different groups for one function ################################
            #ylabel = '(Edemand - TVeg)/Edemand'
            
            # ylabel = '(Evpd - TVeg)/TVeg'
            # ylabel = 'Evpd - TVeg'


              

    if (iplot>0) and (iplot<3):
        plt.savefig(f"{ffigout}z00_NTDfit_multisite.png", dpi=300, bbox_inches="tight", transparent=True)
    # elif iplot>3:
    if iplot>0:
        self.plot_save(self.fig)


    self.close()


    t2    = ptime.time()
    strin = ( '[m]: {:.1f}'.format((t2 - t1) / 60.)
              if (t2 - t1) > 60.
              else '[s]: {:d}'.format(int(t2 - t1)) )
    print('    Time elapsed', strin)
