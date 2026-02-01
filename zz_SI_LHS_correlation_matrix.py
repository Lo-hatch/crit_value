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
        df = pd.read_excel(file_path, sheet_name=f'{isite}{tag}')
        df['method'] = 'NTD'
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

        df_new = df.copy()
        wb_to_rwc_list = ['wb_soilmean','wb_30','wb_fr_rootzone','wb_depth_rootzone']
        for iwb in wb_to_rwc_list:
            df_wb = df[df['Xname'] == iwb].copy()
            df_wb['Xname'] = f'{iwb}_rwc'
            df_wb = df_wb.merge(soil_df[['soiltype', 'swilt', 'sfc']], on='soiltype', how='left')
            def normalize_val(val, swilt, sfc):
                if pd.isna(val) or pd.isna(swilt) or pd.isna(sfc) or sfc == swilt:
                    return float('nan')
                return (val - swilt) / (sfc - swilt)
            for c in cols:
                df_wb[c] = df_wb.apply(lambda row: normalize_val(row[c], row['swilt'], row['sfc']), axis=1)
            df_wb = df_wb.drop(columns=['swilt', 'sfc'])
            df_new = pd.concat([df_new, df_wb], ignore_index=True)

            df_wb = df[df['Xname'] == iwb].copy()
            df_wb['Xname'] = f'{iwb}_psi'
            df_wb = df_wb.merge(soil_df[['soiltype', 'ssat', 'sucs','bch']], on='soiltype', how='left')
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
            df_wb = df[df['Xname'] == iwb].copy()
            df_wb['Xname'] = f'{iwb}_rwc'
            cols = ['MaxCur', 'MaxCurChange', 'MinCurChange','best','BreakPoint']
            df_wb = df_wb.merge(soil_df[['soiltype', 'swilt', 'sfc']], on='soiltype', how='left')
            def normalize_val(val, swilt, sfc):
                if pd.isna(val) or pd.isna(swilt) or pd.isna(sfc) or sfc == swilt:
                    return float('nan')
                return (val - swilt) / (sfc - swilt)
            for c in cols:
                df_wb[c] = df_wb.apply(lambda row: normalize_val(row[c], row['swilt'], row['sfc']), axis=1)
            df_wb = df_wb.drop(columns=['swilt', 'sfc'])
            df_new = pd.concat([df_new, df_wb], ignore_index=True)

            df_wb = df[df['Xname'] == iwb].copy()
            df_wb['Xname'] = f'{iwb}_psi'
            df_wb = df_wb.merge(soil_df[['soiltype', 'ssat', 'sucs','bch']], on='soiltype', how='left')
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
    def __init__(self, *args, **kwargs):
        """ initialisation """
        super().__init__(*args, **kwargs)
        self.left     = 0.08
        self.right    = 0.95
        self.bottom   = 0.1
        self.top      = 0.95
        self.hspace   = 0.06
        self.vspace   = 0.05
        self.textsize = 10
        self.dxabc    = 0
        self.dyabc    = 1.04
        self.llxbbox    = 0.72
        self.llybbox    = 0.5
        self.llrspace   = 0.2
        self.llcspace   = 1.0
        self.llhtextpad = 0.4
        self.llhlength  = 1.5
        self.rasterized = True
        self.dpi = 150
        self.set_matplotlib_rcparams()
        my_pf.set_plot_style()

if __name__ == '__main__':
    import time as ptime
    import glob
    t1 = ptime.time()
    
    # Define the 4 parameters we want to plot with their individual files
    param_configs = {
        'g1tuzet': {
            'critname': 'result_BASEdc98fe39_LHS_g1tuzet.xlsx',
            'csvname': 'g1tuzet_sample.csv',
            'label': 'g1'
        },
        'kmax': {
            'critname': 'result_BASEdc98fe39_LHS_kmax.xlsx',
            'csvname': 'ks_sample.csv',
            'label': r'$K_s$'
        },

        'P50': {
            'critname': 'result_BASEdc98fe39_LHS_P50.xlsx',  # Assuming this exists, adjust if needed
            'csvname': 'P50_sample.csv',  # Assuming this exists, adjust if needed
            'label': r'$P50_{sx}$'
        },
            'psi_50_leaf': {
            'critname': 'result_BASEdc98fe39_LHS_psi_50_leaf.xlsx',
            'csvname': 'psi_50_leaf_sample.csv',
            'label': r'$P50_{l}$'
        },
    }
    
    param_list = list(param_configs.keys())
    cols_label = {k: v['label'] for k, v in param_configs.items()}
    
    indexname1 = ['slope0p1']
    plotorstat = {
        'plotcorr':{'plot':True,'figname':f'z05_LHS_correlation_matrix_{indexname1[0]}'},
        'calcorr':{'plot':False,'figname':f'correlation_matrix_{indexname1[0]}'},
    }

    self = PlotIt('Scatter Plot Matrix')

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
    if tag == '_EF':
        indexname = "BreakPoint"
    elif tag == '_NTD':
        indexname = "slope0p1"

    # 1. Read Excel files for each parameter separately
    # Keep each parameter's data separate so each uses its own NTD/EF values
    commonkeys  = ['ID','Xname','soiltype','sand fraction']
    
    # Load data for each parameter separately - store in a dictionary
    df_by_param = {}
    for param in param_list:
        config = param_configs[param]
        critname = config['critname']
        csvname = config['csvname']
        
        # Read parameter values
        dfparams = pd.read_csv(f"{file_path}{csvname}")
        
        # Read EF and NTD results
        df2 = LP_excle(f"{file_path}{critname}",'_EF')
        df2 = df2[commonkeys+['BreakPoint']]
        df2 = df2.rename(columns={"BreakPoint": "EF"})
        df1 = logi_excle(f"{file_path}{critname}",'_NTD')
        df1 = df1[commonkeys+indexname1]
        df1 = df1.rename(columns={'slope0p1': "NTD"})
        
        # Merge EF and NTD
        df_param = df1.merge(
            df2,
            on=commonkeys,
            how="inner"
        )
        
        # Merge with parameter values
        param_cols = ['ID', param] if param in dfparams.columns else ['ID']
        dfparams_param = dfparams[param_cols]
        df_param = df_param.merge(
            dfparams_param,
            on="ID",
            how="left"
        )
        
        # Handle special cases
        if 'g1tuzet' in csvname:
            df_param['NTD-NL'] = df_param['NTD']
            df_param.loc[df_param['ID'] == 1, 'NTD-NL'] = np.nan
            df_param['EF-NL'] = df_param['EF']
            df_param.loc[df_param['ID'] == 1, 'EF-NL'] = np.nan
        
        # Store this parameter's complete dataframe
        df_by_param[param] = df_param

    xname_labels = {
        "wb_soilmean": fr'$\theta_{{crit}}$',
        "wb_soilmean_rwc": fr'$REW_{{crit}}^{{\theta}}$',
        #"rwc_soilmean_recal":fr'$REW_{{crit}}$',
        "wb_soilmean_psi": fr'$\psi_{{crit}}^{{\theta}}$',
        #"psi_soilmean": fr'$\psi_{{crit}}$',
    }  
    
    # Get soil types in the order defined in soil_data (fine clay first, sand last)
    soil_type_order = [soil_data[i]['soiltype'] for i in sorted(soil_data.keys())]
    # Get available soil types from the first parameter's dataframe
    if len(df_by_param) > 0:
        first_param = list(df_by_param.keys())[0]
        available_soil_types = df_by_param[first_param]["soiltype"].unique()
    else:
        available_soil_types = []
    # Keep only soil types that exist in the data, in the order from soil_data
    soil_types = [st for st in soil_type_order if st in available_soil_types]
    n_soil = len(soil_types)
    n_param = len(param_list)
    
    # Set up plot dimensions: rows = soil types, columns = parameters
    self.nrow = n_soil
    self.ncol = n_param

    if plotorstat["plotcorr"]["plot"]:
        figname = plotorstat["plotcorr"]["figname"]
        
        # Second pass: create plots - one figure per xname
        print(f"DEBUG: xname_labels keys: {list(xname_labels.keys())}")
        
        for xname, ylabel in xname_labels.items():
            print(f"DEBUG: Processing xname: {xname}")
            # First pass: calculate y-axis limits for each soil type (row) for this xname
            ylims_by_soil = {}
            for isoil in soil_types:
                y_min_list = []
                y_max_list = []
                for col in param_list:
                    if col not in df_by_param:
                        continue
                    df = df_by_param[col][(df_by_param[col]["Xname"] == xname) & (df_by_param[col]["soiltype"] == isoil)]
                    if len(df) == 0:
                        continue
                    # Get y values from NTD and EF columns
                    if 'NTD-NL' in df.columns and 'EF-NL' in df.columns:
                        yy = df[["NTD","NTD-NL", "EF","EF-NL"]]
                    else:
                        yy = df[["NTD", "EF"]]
                    # Get all y values (flatten all columns)
                    y_values = []
                    for col_name in yy.columns:
                        y_values.extend(yy[col_name].dropna().tolist())
                    if len(y_values) > 0:
                        # For log scale (wb_soilmean_psi), filter out non-positive values
                        if xname == 'wb_soilmean_psi':
                            y_values = [v for v in y_values if v > 0]
                        if len(y_values) > 0:
                            y_min_list.append(np.nanmin(y_values))
                            y_max_list.append(np.nanmax(y_values))
                if y_min_list and y_max_list:
                    y_min = np.nanmin(y_min_list)
                    y_max = np.nanmax(y_max_list)
                    # For log scale (wb_soilmean_psi), use multiplicative padding
                    if xname == 'wb_soilmean_psi':
                        # Ensure positive values for log scale
                        if y_min > 0 and y_max > 0:
                            # Use multiplicative padding for log scale
                            ylims_by_soil[isoil] = (
                                y_min / 1.1,  # Divide by factor for lower bound
                                y_max * 1.1   # Multiply by factor for upper bound
                            )
                        else:
                            ylims_by_soil[isoil] = (0.01, 1)  # Default positive range
                    else:
                        # Add some padding (5% of range) for linear scale
                        y_range = y_max - y_min
                        if y_range > 0:
                            ylims_by_soil[isoil] = (
                                y_min - 0.05 * y_range,
                                y_max + 0.05 * y_range
                            )
                        else:
                            ylims_by_soil[isoil] = (y_min - 0.1, y_max + 0.1)
                else:
                    if xname == 'wb_soilmean_psi':
                        ylims_by_soil[isoil] = (0.01, 1)  # Default positive range for log scale
                    else:
                        ylims_by_soil[isoil] = (0, 1)
            
            # Reset for each xname - create a new figure
            iplot = 0
            axes_dict = {}  # Store axes by (soil_idx, param_idx) for later ylim setting
            plots_created = False  # Track if any plots were created
            # Store the starting figure number for this xname
            start_fig_num = self.ifig + 1 if hasattr(self, 'ifig') else 1

            for isoil_idx, isoil in enumerate(soil_types):
                for col_idx, col in enumerate(param_list):
        
                    label = cols_label[col]
                    # Use the dataframe for this specific parameter
                    df = df_by_param[col][(df_by_param[col]["Xname"] == xname) & (df_by_param[col]["soiltype"] == isoil)]
                    
                    # Skip if dataframe is empty
                    if len(df) == 0:
                        continue
                    
                    # Set labels: y-label only for first column, x-label only for last row
                    if col_idx == 0:
                        ylab = f'{ylabel}'
                    else:
                        ylab = ''
                    if isoil_idx == n_soil - 1:
                        xlab = label
                    else:
                        xlab = ''
                    
                    colors = [
                        '#1b9e77',
                        "#1D4137",
                        '#d95f02',
                        "#d902a7",
                    ]
                    
                    # Determine which columns to plot
                    if 'NTD-NL' in df.columns and 'EF-NL' in df.columns:
                        color0 = colors
                        yy = df[["NTD","NTD-NL", "EF","EF-NL"]].to_dict(orient="list")
                    else:
                        yy = df[["NTD", "EF"]].to_dict(orient="list")
                        color0 = None
                    
                    # Show legend only for first subplot
                    if iplot == 0:
                        iflegend = True
                    else:
                        iflegend = False
                    
                    # Create plot
                    iplot_before = iplot
                  
                    iplot = my_pf.plot_scatter_simple(
                        xx=df[col].values,
                        yy=yy,
                        xlab=xlab,
                        ylab=ylab,
                        tit=isoil,
                        iplot=iplot,
                        txtpos=[0.98, 0.9],
                        self=self,
                        colors=color0,
                        iflegend=iflegend
                    )
                    
                    # Mark that a plot was created (even if iplot reset to 0 for new page)
                    if iplot > iplot_before or (iplot == 0 and iplot_before > 0):
                        plots_created = True
                    
                    # Get the axes object that was just created and store it
                    # Always try to store the axes, even if iplot didn't increment
                    fig = self.fig
                    axes = fig.get_axes()
                    if len(axes) > 0:
                        # Calculate the expected plot index
                        plot_idx = isoil_idx * n_param + col_idx
                        # Try to get axes by calculated index first
                        if plot_idx < len(axes):
                            ax = axes[plot_idx]
                            axes_dict[(isoil_idx, col_idx)] = ax
                        elif iplot > iplot_before:
                            # If calculated index doesn't work but iplot incremented, use iplot - 1
                            ax_idx = iplot - 1
                            if ax_idx < len(axes):
                                ax = axes[ax_idx]
                                axes_dict[(isoil_idx, col_idx)] = ax
                            else:
                                # Last resort: use the last axes
                                ax = axes[-1]
                                axes_dict[(isoil_idx, col_idx)] = ax
                        else:
                            # If iplot didn't increment, still try to store using last axes
                            # This handles edge cases
                            ax = axes[-1]
                            axes_dict[(isoil_idx, col_idx)] = ax
                        
                        # Set log scale for y-axis when xname is wb_soilmean_psi
                        if xname == 'wb_soilmean_psi':
                            ax.set_yscale('log')
            
            # Final pass: set consistent y-axis limits for each row
            # This ensures all subplots in the same row have the same y-axis limits
            for isoil_idx, isoil in enumerate(soil_types):
                if isoil in ylims_by_soil:
                    ylim = ylims_by_soil[isoil]
                    # DEBUG: Breakpoint here for last row
                    if isoil_idx == len(soil_types) - 1:
                        debug_last_row_first_pass = True  # Set breakpoint here
                        # Debug: Check which axes are in axes_dict for last row
                        last_row_keys = [(isoil_idx, c) for c in range(n_param)]
                        missing_keys = [k for k in last_row_keys if k not in axes_dict]
                      
                        for col_idx in range(n_param):
                            if (isoil_idx, col_idx) in axes_dict:
                                ax = axes_dict[(isoil_idx, col_idx)]
                                # Disable auto-scaling on y-axis
                                ax.set_autoscaley_on(False)
                                # Set log scale for wb_soilmean_psi
                                if xname == 'wb_soilmean_psi':
                                    ax.set_yscale('log')
                                # Set the y-axis limits
                                ax.set_ylim(ylim)
                        else:
                            # Debug: Print if axes is missing
                            if isoil_idx == len(soil_types) - 1:
                                print(f"DEBUG: axes_dict missing key ({isoil_idx}, {col_idx})")
            
            # Check if plots were created (either by iplot > 0 or by checking if figure has axes)
            # This should be OUTSIDE the soil_types loop, at the xname level
            fig = self.fig if hasattr(self, 'fig') and self.fig is not None else None
            if fig is None:
                # Try to get figure from current figure
                fig = plt.gcf()
            
            all_axes = fig.get_axes() if fig is not None else []
            
            print(f"DEBUG: For xname={xname}, plots_created={plots_created}, axes_count={len(all_axes)}, self.ifig={self.ifig if hasattr(self, 'ifig') else 'N/A'}")
            
            if plots_created or len(all_axes) > 0:
                # Get all axes and set limits based on their position
                # This ensures we catch all axes, even if some weren't stored in axes_dict
                
                # Set limits based on calculated position
                for isoil_idx, isoil in enumerate(soil_types):
                    if isoil in ylims_by_soil:
                        ylim = ylims_by_soil[isoil]
                        # DEBUG: Breakpoint here for last row
                        if isoil_idx == len(soil_types) - 1:
                            debug_last_row = True  # Set breakpoint here
                        for col_idx in range(n_param):
                            plot_idx = isoil_idx * n_param + col_idx
                            if plot_idx < len(all_axes):
                                ax = all_axes[plot_idx]
                                ax.set_autoscaley_on(False)
                                # Set log scale for wb_soilmean_psi
                                if xname == 'wb_soilmean_psi':
                                    ax.set_yscale('log')
                                ax.set_ylim(ylim)
                            # Also use stored axes if available (as backup)
                            elif (isoil_idx, col_idx) in axes_dict:
                                ax = axes_dict[(isoil_idx, col_idx)]
                                ax.set_autoscaley_on(False)
                                # Set log scale for wb_soilmean_psi
                                if xname == 'wb_soilmean_psi':
                                    ax.set_yscale('log')
                                ax.set_ylim(ylim)
                
                # Ensure output directory exists
                os.makedirs(figout, exist_ok=True)
                
                # Get the figure - use self.fig which should be set by plot_scatter_simple
                # When iplot=0, plot_scatter_simple creates a new figure and sets self.fig
                save_fig = None
                fig_num = 1
                
                if hasattr(self, 'fig') and self.fig is not None:
                    save_fig = self.fig
                    if hasattr(save_fig, 'number'):
                        fig_num = save_fig.number
                elif hasattr(self, 'ifig'):
                    # If self.fig is None but we have a figure number, get figure by number
                    fig_num = self.ifig
                    save_fig = plt.figure(fig_num)
                else:
                    # Last resort: get current figure
                    save_fig = plt.gcf()
                    if hasattr(save_fig, 'number'):
                        fig_num = save_fig.number
                
                if save_fig is None:
                    print(f"ERROR: Could not get figure for xname={xname}")
                else:
                    # Verify the figure has axes before saving
                    fig_axes = save_fig.get_axes()
                    if len(fig_axes) == 0:
                        print(f"ERROR: Figure for xname={xname} has no axes! Skipping save.")
                    else:
                        print(f"DEBUG: Saving figure for {xname}: {len(fig_axes)} axes, figure number={fig_num}")
                        
                        # Create filename with xname
                        xname_safe = xname.replace('/', '_').replace('\\', '_')  # Make filename safe
                        save_figname = f"{figname}_{xname_safe}"
                        
                        # Save the figure - imitate z05_LHS_correlation_temp.py pattern
                        # First save with plt.savefig, then call plot_save
                        save_path = f"{figout}{save_figname}{fig_num}.png"
                        
                        try:
                            # Save the figure first (like in z05_LHS_correlation_temp.py line 410)
                            plt.savefig(save_path, dpi=300, bbox_inches="tight")
                            print(f"Figure saved to: {save_path} (axes count: {len(all_axes)})")
                            
                            # Then call plot_save (like in z05_LHS_correlation_temp.py line 413)
                            if hasattr(self, 'plot_save'):
                                self.plot_save(save_fig)
                        except Exception as e:
                            print(f"Error saving figure: {e}")
                            import traceback
                            traceback.print_exc()
                            # Try saving with a simpler approach
                            try:
                                save_fig.savefig(save_path, dpi=300, bbox_inches="tight")
                                print(f"Figure saved to: {save_path} (fallback method)")
                            except Exception as e2:
                                print(f"Error saving figure (second attempt): {e2}")
                        
                        # Verify the file was actually saved and has content
                        if os.path.exists(save_path):
                            file_size = os.path.getsize(save_path)
                            if file_size > 1000:  # Check if file has reasonable size (not empty)
                                print(f"  ✓ File saved successfully ({file_size} bytes)")
                            else:
                                print(f"  ⚠ Warning: Saved file is very small ({file_size} bytes), might be empty")
                        else:
                            print(f"  ✗ ERROR: File was not saved!")
                        
                        # Close the figure after saving to free memory
                        # But don't clear self.fig yet - let the next iteration create a new one
                        plt.close(save_fig)
            else:
                print(f"Warning: No plots created for {xname} (iplot = {iplot}, plots_created = {plots_created}, axes_count = {len(all_axes)}), figure not saved")
        
    if plotorstat["calcorr"]["plot"]:
        figname = plotorstat["calcorr"]["figname"]+'.xlsx'
        results = []
        for xname, ylabel in xname_labels.items():
            for isoil in soil_types:
                for col, label in cols_label.items():
                    if col not in df_by_param:
                        continue
                    df = df_by_param[col][(df_by_param[col]["Xname"] == xname) & (df_by_param[col]["soiltype"] == isoil)]
                    for method in ['NTD','EF']:
                        sub = df[[col, method]].dropna()
                        # For wb_soilmean_psi, apply log10 to y values (method column) for regression
                        if xname == 'wb_soilmean_psi':
                            # Filter out non-positive values before taking log10
                            sub = sub[sub[method] > 0].copy()
                        
                        if len(sub) > 0:
                            if xname == 'wb_soilmean_psi':
                                sub_log = sub.copy()
                                sub_log[method] = np.log10(sub_log[method])
                                res = linregress(sub_log[col], sub_log[method])
                            else:
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
            with pd.ExcelWriter(f'{figout}{figname}', mode="w", engine="openpyxl") as writer:
                df_out.to_excel(writer, index=False)
        else:
            with pd.ExcelWriter(f'{figout}{figname}', mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                df_out.to_excel(writer, index=False)

    # Don't show figures interactively since we're saving them all
    # If you want to view figures, uncomment the line below
    # plt.show()
    self.close()
