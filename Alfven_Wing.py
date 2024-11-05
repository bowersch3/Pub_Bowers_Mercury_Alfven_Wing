#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:13:42 2024

@author: bowersch
"""

ICME_List_file='/Users/bowersch/Desktop/DIAS/Work/My Papers/Alfven_Wing/ICME_MESSENGER_Winslow.xlsx'

import pandas as pd
import datetime
import spiceypy
meta_kernel = '/Users/bowersch/Desktop/MESSENGER Data/msgr_spice_kernels/MESSENGER_meta_kernel.txt'

import matplotlib.colors as colors
if spiceypy.ktotal( 'ALL' )==0:
    
    spiceypy.furnsh(meta_kernel)


from os import sys
import matplotlib.colors as colors
#from lobe_id import shade_in_time_series

sys.path.append('/Users/bowersch/Desktop/Python_Code/MESSENGER_Lobe_Analysis/')
from trying3 import check_for_mp_bs_WS,convert_to_datetime,convert_to_date_2,plot_mp_and_bs,\
    plot_MESSENGER_trange_cyl,convert_to_date_2_utc,get_mercury_distance_to_sun,\
        convert_datetime_to_string,plot_MESSENGER_trange_3ax,get_aberration_angle,\
            get_day_of_year,convert_to_date_s, read_in_Weijie_files,\
                convert_to_datetime_K
from generate_all_plot import generate_all_plot
from ANN_work import plot_region_time_series_plots
import kth22_model_for_mercury_v8 as kth   
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,AutoMinorLocator
from Machine_Learning import load_PAD
from FIPS_FOV_Map import get_FIPS_FOV

def event_time_series(i):
    
    '''
    FIGURE 1 in the manuscript (event_time_series(21))
    
    enerate a time series of the ICME event to check for interesting 
    features, i is the ICME number (0-6)
    
    '''
    
    icme=pd.read_excel(ICME_List_file)
    
    time_a = icme[['ICME_A_YEAR','ICME_A_DOY','ICME_A_HH','ICME_A_MM','ICME_A_SS']]\
        .iloc[i]
        
    time_e = icme[['ICME_E_YEAR','ICME_E_DOY','ICME_E_HH','ICME_E_MM','ICME_E_SS']]\
        .iloc[i]
        
    fips_data = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/MESSENGER_Lobe_Analysis/FIPS_Dictionary.pkl')
    
    H_data = fips_data['H_data']
    
    time_FIPS = fips_data['time']
        
    def create_datestring(year, day_of_year, hour, minute, second):
        # Create a datetime object for January 1st of the given year
        date = datetime.datetime(int(year), 1, 1)
    
        # Add the number of days (day_of_year - 1) to get to the desired date
        date += datetime.timedelta(days=float(day_of_year) - 1, hours=float(hour), minutes=float(minute),seconds=float(second))
        
        def convert_datetime_to_string(time):
            dstring=time.strftime("%Y-%m-%d %H:%M:%S")
            return dstring
        
        return convert_datetime_to_string(date), date
    
    date_string_a, date_a = create_datestring(time_a[0], time_a[1], time_a[2], time_a[3], time_a[4])
    
    date_string_e, date_e = create_datestring(time_e[0], time_e[1], time_e[2], time_e[3], time_e[4])
    
    
    print(date_a)
    print(date_e)
    trange_ICME=[date_string_a,date_string_e]
    
    #trange_ICME=['2012-05-25 00:00:00','2012-05-25 23:59:00']
    
    def make_plot(trange):
        
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import numpy as np
        
        from trying3 import load_MESSENGER_into_tplot,read_in_Weijie_files,check_for_mp_bs_WS
        #time,mag=pytplot.get_data("B_mso")
        #time,magamp=pytplot.get_data("B_amp")
        
        #from lobe_id import check_FIPS_file
        
        #fcheck=check_FIPS_file(trange[0][0:10])
        
        fcheck=True
        
        if ((fcheck==True) or (fcheck==False)):
            
            time,mag,magamp,eph=load_MESSENGER_into_tplot(trange[0][0:10],FIPS=False)
            
            if trange[0][0:10] != trange[1][0:10]:
                from create_lobe_time_series import append_for_run_over_FIPS,append_for_run_over
                time,mag,magamp,eph,mp_in_ts,mp_out_ts,bs_in_ts,bs_out_ts=append_for_run_over(trange[0][0:10])
                
            fig, (ax1,ax2,ax3,ax4,ax5,ax6)=plt.subplots(6,sharex=True)
            
            control_param_path = '/Users/bowersch/Desktop/Python_Code/KTH22-model-main/control_params_v8b.json'
            fit_param_path = '/Users/bowersch/Desktop/Python_Code/KTH22-model-main/kth_own_cf_fit_parameters_opt_total_March23.dat'
            
            
            
            def create_KTH_prediction(trange):
                
                date_string_mso=trange[0][0:10]
                
                ephx_mso,ephy_mso,ephz_mso=get_eph_mso(date_string_mso)
                
                x_mso=ephx_mso
                
                y_mso=ephy_mso
                
                z_mso=ephz_mso
                
                r_hel = np.ones(len(x_mso))*get_mercury_distance_to_sun(time[0])
                
                di = np.ones(len(x_mso))*100
                
                B_KTH =  kth.kth22_model_for_mercury_v8(x_mso, y_mso, z_mso, r_hel, di, 
                                                        control_param_path, fit_param_path, 
                                                        dipole=True, neutralsheet=True, 
                                                        ringcurrent=True, internal=True, 
                                                        external=True)
    
                Bx_KTH = B_KTH[0]
                By_KTH = B_KTH[1]
                Bz_KTH = B_KTH[2]
                
                
                magamp_KTH=np.sqrt(Bx_KTH**2+Bz_KTH**2+By_KTH**2)
                
                day_before = convert_datetime_to_string(convert_to_datetime(trange[0])\
                    -datetime.timedelta(days=1))
                
                if trange[0][0:10] != trange[1][0:10]:
                    date_string_mso=trange[1][0:10]
                    
                    ephx_mso,ephy_mso,ephz_mso=get_eph_mso(date_string_mso)
                    
                    x_mso=ephx_mso
                    
                    y_mso=ephy_mso
                    
                    z_mso=ephz_mso
                    
                    r_hel = np.ones(len(x_mso))*get_mercury_distance_to_sun(time[0])
                    
                    di = np.ones(len(x_mso))*100
                    
                    B_KTH =  kth.kth22_model_for_mercury_v8(x_mso, y_mso, z_mso, r_hel, di, 
                                                            control_param_path, fit_param_path, 
                                                            dipole=True, neutralsheet=True, 
                                                            ringcurrent=True, internal=True, 
                                                            external=True)
                    
                    magamp_KTH_2=np.sqrt(B_KTH[0]**2+B_KTH[0]**2+B_KTH[2]**2)
                    
                    Bx_KTH=np.append(Bx_KTH,B_KTH[0])
                    By_KTH=np.append(By_KTH,B_KTH[1])
                    Bz_KTH=np.append(Bz_KTH,B_KTH[2])
                    
                    magamp_KTH=np.append(magamp_KTH,magamp_KTH_2)
                    
                    date_string_mso=day_before[0:10]
                    
                    ephx_mso,ephy_mso,ephz_mso=get_eph_mso(date_string_mso)
                    
                    x_mso=ephx_mso
                    
                    y_mso=ephy_mso
                    
                    z_mso=ephz_mso
                    
                    r_hel = np.ones(len(x_mso))*get_mercury_distance_to_sun(time[0])
                    
                    di = np.ones(len(x_mso))*100
                    
                    B_KTH =  kth.kth22_model_for_mercury_v8(x_mso, y_mso, z_mso, r_hel, di, 
                                                            control_param_path, fit_param_path, 
                                                            dipole=True, neutralsheet=True, 
                                                            ringcurrent=True, internal=True, 
                                                            external=True)
                    
                    magamp_KTH_2=np.sqrt(B_KTH[0]**2+B_KTH[0]**2+B_KTH[2]**2)
                    
                    Bx_KTH=np.append(Bx_KTH,B_KTH[0])
                    By_KTH=np.append(By_KTH,B_KTH[1])
                    Bz_KTH=np.append(Bz_KTH,B_KTH[2])
                    
                    magamp_KTH=np.append(magamp_KTH,magamp_KTH_2)
                    
                return Bx_KTH,By_KTH,Bz_KTH,magamp_KTH
            
            #Bx_KTH,By_KTH,Bz_KTH,magamp_KTH = create_KTH_prediction(trange)
            
            #ax6.set_yscale('log')
                    
            

            
            dates=time
            
            new_resolution = 300  # Number of points to average over

            time=dates
            
            time=np.array([pd.Timestamp(t) for t in time])
            
            
            # Calculate the new length of the arrays after averaging
            new_length = len(time) // new_resolution
            
            # Reshape the arrays to prepare for averaging
            t_reshaped = time[:new_length * new_resolution].reshape(new_length, new_resolution)
            
            def tstamp(x):
                return x.timestamp()
            t_utc=np.vectorize(tstamp)(t_reshaped)
            
            # Calculate the average values
            t_averaged_utc = np.mean(t_utc, axis=1)
            
            t_averaged=np.array([convert_to_datetime(convert_to_date_2_utc(t)) for t in t_averaged_utc])
            
            
            # Create a figure and axis objec
            
            # Plot the data
            
            # Set the x-axis label and format
            
            ax1.axhline(y=0.0, color='black', linestyle='--',linewidth=0.5)
            ax2.axhline(y=0.0, color='black', linestyle='--',linewidth=0.5)
            ax3.axhline(y=0.0, color='black', linestyle='--',linewidth=0.5)
            #ax5.axhline(y=0.0, color='black', linestyle='--',linewidth=0.5)
            
           # ax5.plot(dates,eph,linewidth=.75)
            #ax5.set_ylabel('EPH')
            
            bd = np.where((dates < convert_to_datetime('2011-12-30 19:02:00')) | (dates > convert_to_datetime('2011-12-30 19:05:00')))[0]
            
            gd = np.where((dates > convert_to_datetime(trange_ICME[0])) & (dates < convert_to_datetime(trange_ICME[1])))[0]
            
            dates = dates[bd]
            
            mag = mag[bd,:]
            
            eph = eph[gd,:]
            
            magamp = magamp[bd]
            
            #ax6.plot(t_averaged,fov[:,0],color='blue',label="$X_{MSM'}$")
            #ax6.plot(t_averaged,fov[:,1],color='green',label="$Y_{MSM'}$")
            #ax6.plot(t_averaged,fov[:,2],color='red',label="$Z_{MSM'}$")
            #ax6.legend()
            
            #ax6.plot(fips_data['time'],fips_data['nobs'])
            
            ax1.plot(dates,mag[:,0],'darkorange',linewidth=1.2,label='MESSENGER')

            #ax1.plot(dates,Bx_KTH,'red',linewidth=.75,label='KTH_Model')
            
            
            ax2.plot(dates,mag[:,1],'darkorange',linewidth=1.2,label='MESSENGER')
            #ax2.plot(dates,By_KTH,'red',linewidth=.75,label='KTH_Model')
            
            
            ax3.plot(dates,mag[:,2],'darkorange',linewidth=1.2,label='MESSENGER')
            #ax3.plot(dates,Bz_KTH,'red',linewidth=.75,label='KTH_Model')
            
            
            ax4.plot(dates,magamp,'darkorange',linewidth=1.2,label='MESSENGER')
            #ax4.plot(dates,magamp_KTH,'red',linewidth=.75,label='KTH_Model')
            
            ax_eph = make_mercury()
            
            ax_eph[0].plot(eph[:,0]/2440,eph[:,2]/2440,color = 'black')
            ax_eph[1].plot(eph[:,0]/2440,eph[:,1]/2440,color = 'black')
            ax_eph[2].plot(eph[:,1]/2440,eph[:,2]/2440,color = 'black')
            
            ax_eph[0].scatter(eph[0,0]/2440,eph[0,2]/2440,color='maroon',s=40)
            ax_eph[1].scatter(eph[0,0]/2440,eph[0,1]/2440,color='maroon',s=40)
            ax_eph[2].scatter(eph[0,1]/2440,eph[0,2]/2440,color='maroon',s=40)
            
            ax_eph[0].scatter(eph[-1,0]/2440,eph[-1,2]/2440,color='maroon',s=40)
            ax_eph[1].scatter(eph[-1,0]/2440,eph[-1,1]/2440,color='maroon',s=40)
            ax_eph[2].scatter(eph[-1,1]/2440,eph[-1,2]/2440,color='maroon',s=40)
            
            ax_eph[0].set_ylabel("$Z_{MSM'}$")
            ax_eph[0].set_xlabel("$X_{MSM'}$")
            
            ax_eph[1].set_ylabel("$Y_{MSM'}$")
            ax_eph[1].set_xlabel("$X_{MSM'}$")
            
            ax_eph[2].set_ylabel("$Z_{MSM'}$")
            ax_eph[2].set_xlabel("$Y_{MSM'}$")

            for x in ax_eph:
                
                format_ticks(x,17)
            
            
            
            tail_t=np.load('/Users/bowersch/Desktop/Python_Code/MESSENGER_Lobe_Analysis/tail_t_np.npy',allow_pickle=True)
            
            date_event=convert_to_datetime(trange[0])
            
            diff=date_event-tail_t[:,0]
            
            diff=np.array([np.abs(d.total_seconds()) for d in diff])
            
            gd=np.where(diff==np.min(diff))[0][0]
            
            tail_t=tail_t[gd,:]
            
            dates=np.array(dates)
            
            
            
            
            #ax4.plot(dates[gd],pred,color='darkorange',linewidth=1.1)
            
            #ax4.plot(dates[gd],pred2,color='darkorange',linewidth=.8,linestyle='-')
            
            #ax4.plot(dates[gd],pred3,color='darkorange',linewidth=.8,linestyle='-')
            
            
            ax1.set_ylabel("$BX_{MSM'}$ \n [nT]",fontsize = 14)
            ax2.set_ylabel("$BY_{MSM'}$ \n [nT]",fontsize = 14)
            ax3.set_ylabel("$BZ_{MSM'}$ \n [nT]",fontsize= 14)
            ax4.set_ylabel('$|B|$ \n [nT]',fontsize = 14)
            
            axess=[ax1,ax2,ax3,ax4]
            
            mp = np.load('/Users/bowersch/Desktop/Python_Code/MESSENGER_Lobe_Analysis/mp_total.npz',allow_pickle=True)
            
            bs = np.load('/Users/bowersch/Desktop/Python_Code/MESSENGER_Lobe_Analysis/bs_total.npz',allow_pickle=True)
            
            mp_time = mp['time']
            
            bs_time = bs['time']
            
            gd_bs = np.where((bs_time[:,0] > convert_to_datetime(trange_ICME[0]))  & (bs_time[:,0] < convert_to_datetime(trange_ICME[1])))[0]
            
            bs_time = bs_time[gd_bs,:]
            
            gd_mp = np.where((mp_time[:,0] > convert_to_datetime(trange_ICME[0]))  & (mp_time[:,0] < convert_to_datetime(trange_ICME[1])))[0]
            
            mp_time = mp_time[gd_mp,:]
            
            for x in axess:
                format_ticks(x,14)
                

                #x.legend(loc='upper left')
            
            
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            ds=trange[0][0:10]
            #ax4.set_xlabel(ds)
            
            #print(trange[0])
            
            tstart=datetime.datetime.strptime(trange[0], "%Y-%m-%d %H:%M:%S")
            
            tstop=datetime.datetime.strptime(trange[1], "%Y-%m-%d %H:%M:%S")
            
            ax4.set_xlim(tstart,tstop)
            
            ax1.set_ylim(-150,150)
            ax2.set_ylim(-150,150)
            ax3.set_ylim(-150,150)
            ax4.set_ylim(0,300)
            #ax4.set_yscale('log')

                
            
            import matplotlib.colors as colors
            #figi=ax5.pcolormesh(tF,erange,np.transpose(HF),shading='nearest',\
                          #norm=colors.LogNorm(),cmap='inferno')
            
                
            
# =============================================================================
#             cb_ax = fig.add_axes([.93,.091,.02,.2])
#             fig.colorbar(figi,orientation='vertical',cax=cb_ax,label='DEF (cm$^2$ s sr)$^{-1}$')
#         
#             ax5.set_yscale('log')
#             
#             ax5.set_ylim(.01,20)
#             
#             ax5.set_ylabel('E/q [keV/q]')
# =============================================================================
            N = [np.size(np.where(H_data[d,:] > 0)) for d in range(np.size(H_data[:,0]))]
            
            #ax7.plot(tF,N,color='black')
            
            fips_data = pd.read_pickle('/Users/bowersch/FIPS_Dictionary.pkl')
                
            tF = fips_data['time']
            
            HF = fips_data['H_data']
            
            file = '/Users/bowersch/Desktop/FIPS_R2011364CDR_V3.TAB'
            
            df = np.genfromtxt(file)
            
            


            HF=df[:,195:259]

            df_dates=df[:,0]
            
            cutoff=200411609
            
            #cutoff=cutoff.timestamp()

            if df_dates[-1] > cutoff:
                

                datetime_MET=convert_to_date_s('2004-08-03 06:55:32')
                
            else: 
                
                datetime_MET=convert_to_date_s('2013-01-08 20:13:20')

            df_dates2=df_dates+datetime_MET.timestamp()

            df_datetime=[convert_to_date_2_utc(d) for d in df_dates2]



            df_datetime=[convert_to_datetime(d) for d in df_datetime]

            tF=np.array(df_datetime)
            
            file2 = '/Users/bowersch/Desktop/FIPS_R2011365CDR_V3.TAB'
            
            df = np.genfromtxt(file2)
            
            HF_2=df[:,195:259]

            df_dates=df[:,0]
            
            
            #cutoff=cutoff.timestamp()

            if df_dates[-1] > cutoff:
                

                datetime_MET=convert_to_date_s('2004-08-03 06:55:32')
                
            else: 
                
                datetime_MET=convert_to_date_s('2013-01-08 20:13:20')

            df_dates2=df_dates+datetime_MET.timestamp()

            df_datetime=[convert_to_date_2_utc(d) for d in df_dates2]



            df_datetime=[convert_to_datetime(d) for d in df_datetime]

            tF_2=np.array(df_datetime)
            
            HF = np.vstack((HF,HF_2))
            
            tF = np.hstack((tF,tF_2))
            
            
            
            
            erange = fips_data['erange']
            
            df_ntp = fips_data['df_ntp']
            
            gd_fips = np.where((tF > date_a) & (tF < date_e))[0]
            
            tF = tF[gd_fips]
            
            HF = HF[gd_fips,:]
            
            
            gd_nobs = np.where((df_ntp.time > date_a) & (df_ntp.time < date_e))[0]
            
            df_ntp = df_ntp.iloc[gd_nobs]
            
            add_onto_density_panel(ax6,df_ntp.time,df_ntp.n,'darkorange',19,1,'ICME Orbit')
            
            ax6.scatter(df_ntp.time[(df_ntp.qual==1)],df_ntp.n[(df_ntp.qual==1)],color='red',s=10,zorder=4,marker = '^')
            
            ax6.set_yscale('log')
            ax6.set_ylim(1E-2,100)
            
            
            import matplotlib.colors as colors
            figi=ax5.pcolormesh(tF,erange,np.transpose(HF),shading='nearest',\
                          norm=colors.LogNorm(),cmap='inferno')
            
            pos1 = ax5.get_position()  # [left, bottom, width, height]

            # Adjust the position for the colorbar
            pos2 = [pos1.x1 + 0.01, pos1.y0, 0.02, pos1.height]  # [left, bottom, width, height]

            # Create a new axis for the colorbar
            cax = fig.add_axes(pos2)

            # Add the colorbar to the new axis
            colorbar = fig.colorbar(figi, cax=cax) 
            
            ax5.set_ylabel('E/q \n [kEV/q]',fontsize = 14)
            
            ax6.set_ylabel('n \n [cm$^{-3}$]',fontsize=14)
            

            ax5.set_yscale('log')
            ax5.set_ylim(1E-1,20)
                
                
            
            
        
        
        ax=[ax1,ax2,ax3,ax4,ax5,ax6]
        
        icme_sheath = [convert_to_datetime('2011-12-30 16:27:23'),\
                       convert_to_datetime('2011-12-30 20:54:15')]
            
        bs_1 = bs_time[0,:]
            
        
        
        for axis in ax:
            
            axis.axvline(x = date_a, color='maroon',linewidth=1.2)
        
            axis.axvline(x = date_e, color='maroon',linewidth=1.2)
            
            #shade_in_time_series(axis,icme_sheath[0],icme_sheath[1],color='blue',shade=False,alpha=.05)
            
            
            for i in range(len(bs_time)):
                print(bs_time[i])
                axis.axvline(x = icme_sheath[1],color='blue',linewidth = 1.2)
                axis.axvline(x = bs_time[i,0],color = 'green', linewidth = 1.2)
                axis.axvline(x = mp_time[i,0],color = 'mediumpurple', linewidth = 1.2)
                axis.axvline(x = mp_time[i,1],color = 'mediumpurple', linewidth = 1.2)
                if i ==1:
                    
                    axis.axvline(x = bs_time[i,1],color = 'green', linewidth = 1.2)
            
            format_ticks(axis,17)
        
        print([date_string_a,date_string_e])    
        
        #ax=plot_MESSENGER_trange_3ax(trange)
        
    make_plot(trange_ICME)
    
def multi_orbit_plot(ii,r1=False,r2=False,r3=False,r4=False,ranges=False,conduct=False,ganymede=False):
    
    '''FIGURE 2 in the Manuscript
    
    Example Usage: multi_orbit_plot(21)
    
    '''
    
    alpha_gray= .4
    
    def make_3d_mercury():
        
        fs=18
        
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create a sphere
        phi, theta = np.mgrid[0.0:2.0*np.pi:100j, 0.0:np.pi:50j]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        
        phi, theta = np.mgrid[0.0:1.0*np.pi:100j, 0.0:np.pi:50j]
        y_neg = np.sin(theta) * np.cos(phi)
        x_neg = np.sin(theta) * np.sin(phi)*(-1)
        z_neg = np.cos(theta)
        
        
        # Create a figure and a 3D Axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the sphere
        ax.plot_surface(x, y, z-.19, color='grey', alpha=1)
        
        ax.plot_surface(x_neg, y_neg, z_neg-.19, color='black', alpha=1)
        
        # Set axis labels
        ax.set_xlabel("$X_{MSM'}$",fontsize=fs-2)
        ax.set_ylabel("$Y_{MSM'}$",fontsize=fs-2)
        ax.set_zlabel("$Z_{MSM'}$",fontsize=fs-2)
        
        # Set plot limits to have equal aspect ratio
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-5, 1])
        ax.set_aspect('equal') 
        ax.tick_params(labelsize=fs-5)
        
        return ax
    
        
    
    from trying3 import plot_MESSENGER_trange_cyl
    
    ax_c=plot_MESSENGER_trange_cyl(['2011-09-20 00:00:00','2011-09-20 15:00:00'],plot=False)
    
    def make_Ma_plot():
        fs=19
        
        fig, ax = plt.subplots(2,2)
        
        #ax.set_title('Alfvenic Mach Number Estimates',fontsize=fs)
        
        ax[0,0].set_ylabel('$B_{IMF}$ (nT)', fontsize = fs - 2)
        
        ax[1,0].set_ylabel('$P_{SW}^{dyn}$ (nPa)',fontsize = fs-2)
        
        ax[0,1].set_ylabel('$P_{M}^{mag}$ (nPa)',fontsize = fs-2)
        
        ax[1,1].set_ylabel('\u03B8 (deg)',fontsize = fs-2)
        
        for x in range(2):
            
            for y in range(2):
                ax[x,y].set_xlabel('$M_{A}$',fontsize = fs-2)
                ax[x,y].xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
                ax[x,y].yaxis.set_minor_locator(AutoMinorLocator())
                
                ax[x,y].tick_params(axis='both', which='major', length=8)
                ax[x,y].tick_params(axis='both', which='minor', length=4)

        
        #ax.set_yticks([])
        
        #ax.set_ylim([-1,1])
        
        return ax
    


        
    
    def add_onto_MA_plot(ax,M,B_SW,M_err,y_err,col,siz,alph,labe,ylabe):
        
        if ylabe == '$B_{IMF}$ (nT)':
            axis=(0,0)
            
        if ylabe == '$P_{SW}^{dyn}$ (nPa)':
            axis=(1,0)
            
        if ylabe == '$P_{M}^{mag}$ (nPa)':
            axis=(0,1)
            
        if ylabe == '$\theta$ (deg)':
            axis=(1,1)
            
        ax[axis].scatter(M,B_SW,c=col,s=siz,alpha=alph,label=labe)
        ax[axis].errorbar(M,B_SW,xerr=M_err,yerr=y_err,elinewidth=1.5,linestyle='None',color=col,alpha=alph)
        
        
        
        #ax[axis].set_ylabel(ylabe,fontsize=17)
        handles, labels = ax[axis].get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        
        # Plot legend with only unique labels
        #ax[axis].legend(unique_labels.values(), unique_labels.keys(),loc='upper right')
    
    def add_onto_boundary_plot(ax,df,col,siz,alph,labe,philpott=False):
        
        
        mp = np.load('/Users/bowersch/Desktop/Python_Code/MESSENGER_Lobe_Analysis/mp_total.npz',allow_pickle=True)
        
        bs = np.load('/Users/bowersch/Desktop/Python_Code/MESSENGER_Lobe_Analysis/bs_total.npz',allow_pickle=True)
        
        mp_time = mp['time']
        
        bs_time = bs['time']
        
        if philpott==True:
            
            df_boundaries = pd.read_pickle('df_boundaries_philpott.pkl')
            
            mp_in_1 = df_boundaries[df_boundaries.Cross_Type=='mp_in_1'].time
            mp_in_2 = df_boundaries[df_boundaries.Cross_Type=='mp_in_2'].time
            
            mp_out_1 = df_boundaries[df_boundaries.Cross_Type=='mp_out_1'].time
            mp_out_2 = df_boundaries[df_boundaries.Cross_Type=='mp_out_2'].time
            
            mp_1 = np.sort(np.append(mp_in_1,mp_out_1))
            
            mp_2 = np.sort(np.append(mp_in_2,mp_out_2))
            
            mp_time = np.stack((mp_1,mp_2),axis=1)
            
            mp_time = mp_time.astype('datetime64[s]').astype('datetime64[us]').astype(object)
            
            bs_in_1 = df_boundaries[df_boundaries.Cross_Type=='bs_in_1'].time
            bs_in_2 = df_boundaries[df_boundaries.Cross_Type=='bs_in_2'].time
            
            bs_out_1 = df_boundaries[df_boundaries.Cross_Type=='bs_out_1'].time
            bs_out_2 = df_boundaries[df_boundaries.Cross_Type=='bs_out_2'].time

            
            bs_1 = np.sort(np.append(bs_in_1,bs_out_1))
            
            bs_2 = np.sort(np.append(bs_in_2,bs_out_2))
            
            bs_time = np.stack((bs_1,bs_2),axis=1)
            
            bs_time = bs_time.astype('datetime64[s]').astype('datetime64[us]').astype(object)
            
        
        time = df.time
        
        gd_mp = np.where((mp_time[:,0] > time.iloc[0]) & (mp_time[:,1] < time.iloc[-1]))[0]
        
        t_mp = mp_time[gd_mp,:]
        

        gd_bs = np.where((bs_time[:,0] > time.iloc[0]) & (bs_time[:,1] < time.iloc[-1]))[0]
        
        t_bs = bs_time[gd_bs,:]
        
        
        x_mp = np.zeros((0,2))
        
        r_mp = np.zeros((0,2))
        
        x_bs = np.zeros((0,2))
        
        r_bs = np.zeros((0,2))
        
        for l in range(len(t_mp)):
            
            df_0 = df[(df.time > t_mp[l,0]) & (df.time < t_mp[l,0]+datetime.timedelta(seconds=1.5))]
        
            df_1 = df[(df.time > t_mp[l,1]) & (df.time < t_mp[l,1]+datetime.timedelta(seconds=1.5))]
        
            x_mp0 = np.mean(df_0.ephx)
            
            r_mp0 = np.mean(np.sqrt(df_0.ephy**2+df_0.ephz**2))
            
            x_mp1 = np.mean(df_1.ephx)
            
            r_mp1 = np.mean(np.sqrt(df_1.ephy**2+df_1.ephz**2))
            
            x_mp = np.append(x_mp,[[x_mp0,x_mp1]],axis=0)
            
            r_mp = np.append(r_mp, [[r_mp0,r_mp1]],axis=0)
            
        for l in range(len(t_bs)):
        
            df_0 = df[(df.time > t_bs[l,0]) & (df.time < t_bs[l,0]+datetime.timedelta(seconds=1.5))]
        
            df_1 = df[(df.time > t_bs[l,1]) & (df.time < t_bs[l,1]+datetime.timedelta(seconds=1.5))]
        
            x_bs0 = np.mean(df_0.ephx)
            
            r_bs0 = np.mean(np.sqrt(df_0.ephy**2+df_0.ephz**2))
            
            x_bs1 = np.mean(df_1.ephx)
            
            r_bs1 = np.mean(np.sqrt(df_1.ephy**2+df_1.ephz**2))
            
            x_bs = np.append(x_bs,[[x_bs0,x_bs1]],axis=0)
            
            r_bs = np.append(r_bs, [[r_bs0,r_bs1]],axis=0)
            

        #eph_mp = mp['eph']
        

        
        r_mp_range = np.abs((r_mp[:,1]-r_mp[:,0])/2)
        
        r_mp = np.mean(r_mp,axis=1)
        
        x_mp_range = np.abs((x_mp[:,1]-x_mp[:,0])/2)
        
        x_mp = np.mean(x_mp,axis=1)
        
        if labe[0]=='I':
        
            ax.scatter(x_mp,r_mp,c = col, s = siz, alpha = alph,label = labe+ ' MP',zorder=3)
            
            ax.plot(df.ephx,np.sqrt(df.ephy**2+df.ephz**2),c = col, alpha = alph-.1,linewidth=.8,zorder=3)
            
            ax.errorbar(x_mp,r_mp,xerr=x_mp_range,yerr=r_mp_range,elinewidth=1.5,linestyle='None',color=col,alpha=alph,zorder=2)
            
        else:
            
            ax.scatter(x_mp,r_mp,c = col, s = siz, alpha = alph,label = labe+ ' MP',zorder=1)
            
            ax.errorbar(x_mp,r_mp,xerr=x_mp_range,yerr=r_mp_range,elinewidth=1.5,linestyle='None',color=col,alpha=alph,zorder=1)
            
            ax.plot(df.ephx,np.sqrt(df.ephy**2+df.ephz**2),c = col, alpha = alph-.1,linewidth=.8,zorder=1)
        
        
        r_bs_range = np.abs((r_bs[:,1]-r_bs[:,0])/2)
        
        r_bs = np.mean(r_bs,axis=1)

        x_bs_range = np.abs((x_bs[:,1]-x_bs[:,0])/2)
        
        x_bs = np.mean(x_bs,axis=1)
        
        if labe[0] == 'I':
        
            ax.scatter(x_bs,r_bs,c = col, s = siz, alpha = alph, label = labe+' BS', marker = 's',zorder=4)
        
            ax.errorbar(x_bs,r_bs,xerr=x_bs_range,yerr=r_bs_range,elinewidth=1.5,linestyle='None',color=col,alpha=alph,zorder=4)
            
        else:
            ax.scatter(x_bs,r_bs,c = col, s = siz, alpha = alph, label = labe+' BS', marker = 's',zorder=1)
        
            ax.errorbar(x_bs,r_bs,xerr=x_bs_range,yerr=r_bs_range,elinewidth=1.5,linestyle='None',color=col,alpha=alph,zorder=1)
            
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        
        # Plot legend with only unique labels
        ax.legend(unique_labels.values(), unique_labels.keys(),loc='upper right')
        ax.set_ylim([0, 6])
        
        ax.set_ylim([0,6])
        
       
        
        def estimate_MA():
            
            MA = np.array([])
            
            psw_total = np.array([])
            
            B_SW_total = np.array([])
            
            Bx_total = np.array([])
            By_total = np.array([])
            Bz_total = np.array([])
            
            phi_total = np.array([])
            
            Bx_std_total = np.array([])
            
            By_std_total = np.array([])
            
            Bz_std_total = np.array([])
            
            
            B_SW_std_total = np.array([])
            
            pmag_std_total = np.array([])
            
            phi_std_total = np.array([])
            
            pmag_total = np.array([])
            
            psw_std_total = np.array([])
            
            MA_std = np.array([])
            
            for p in range(2):
                
                mp_crossing = mp_time[gd_mp[p],:]
                
                if len(gd_bs) == 0:
                    
                    bs_crossing = mp_crossing
                
                
                if len(gd_bs) == 1:
                    
                    bs_crossing = bs_time[gd_bs[0],:]
                    
                if len(gd_bs) > 1:
                
                    bs_crossing = bs_time[gd_bs[p],:]
                
             
                
                if p == 0:
                    
                    up_range = np.array([bs_crossing[0] - datetime.timedelta(minutes=30),bs_crossing[0]])
                    
                    mp_range = np.array([mp_crossing[1], mp_crossing[1] + datetime.timedelta(minutes=1)])
                    
                if p == 1:
                    
                    up_range = np.array([bs_crossing[1], bs_crossing[1] + datetime.timedelta(minutes=30)])
                    
                    mp_range = np.array([mp_crossing[0] - datetime.timedelta(minutes=1), mp_crossing[0]])
                    

                magamp=df.magamp.to_numpy()
                
                
                mag=df[['magx','magy','magz']].to_numpy()
                
                eph=df[['ephx','ephy','ephz']].to_numpy()
                
                
                Rss=1.45
                alpha=0.5
    
                phi2 = (np.linspace(0,2*np.pi,1000))
    
                rho=Rss*(2/(1+np.cos(phi2)))**(alpha)
    
                xmp=rho*np.cos(phi2)
    
                ymp=rho*np.sin(phi2)
    
                curve=np.transpose(np.vstack((xmp,ymp)))
                
                ea_test = np.mean(eph[(df.time > mp_range[0]) & (df.time < mp_range[1]),:],axis=0)
                
                ea_std = np.std(eph[(df.time > mp_range[0]) & (df.time < mp_range[1]),:],axis=0)
                
                df_MP=df[(df.time > mp_range[0]) & (df.time < mp_range[1])]
                
                percentile = np.percentile(df_MP.magamp.to_numpy(), 80)
                
                df_MP=df_MP[(df.magamp > percentile)]
                
                
                B_MP=np.mean(df_MP.magamp.to_numpy())
                
                B_MP_std = np.std(df_MP.magamp.to_numpy())
                
                B_SW = np.mean(magamp[(df.time > up_range[0]) & (df.time < up_range[1])])
                
                B_SW_std = np.std(magamp[(df.time > up_range[0]) & (df.time < up_range[1])])
                
                B_X = np.mean(mag[(df.time > up_range[0]) & (df.time < up_range[1]),0])
                B_Y = np.mean(mag[(df.time > up_range[0]) & (df.time < up_range[1]),1])
                B_Z = np.mean(mag[(df.time > up_range[0]) & (df.time < up_range[1]),2])
                
                B_X_std = np.std(mag[(df.time > up_range[0]) & (df.time < up_range[1]),0])
                B_Y_std = np.std(mag[(df.time > up_range[0]) & (df.time < up_range[1]),1])
                B_Z_std = np.std(mag[(df.time > up_range[0]) & (df.time < up_range[1]),2])
                munaught=4*np.pi*1E-7
                
                ra_test=np.sqrt(ea_test[1]**2+ea_test[2]**2)
                
                ra_std = 0.5*(2*ea_std[1]+2*ea_std[2])
                
                # Find distance to inner point in curve (mp uncertainty)
                
                da,point=distance_point_to_curve([ea_test[0]-ea_std[0],ra_test-ra_std],curve,get_point=True)
                
                diff=curve-[point[0],point[1]]
                
                
                diff=np.sqrt(diff[:,0]**2+diff[:,1]**2)
                
                index=np.where(np.abs(diff)==np.min(np.abs(diff)))[0][0]
                
                n=calculate_normal_vector(curve,index)
                
                phi_1=np.arccos(np.dot(n,np.array([1,0])))
                
                #Find distance to outer point in curve (mp uncertainty)
                
                da,point=distance_point_to_curve([ea_test[0]+ea_std[0],ra_test+ra_std],curve,get_point=True)
                
                diff=curve-[point[0],point[1]]

                diff=np.sqrt(diff[:,0]**2+diff[:,1]**2)
                
                index=np.where(np.abs(diff)==np.min(np.abs(diff)))[0][0]
                
                n=calculate_normal_vector(curve,index)
                
                phi_2=np.arccos(np.dot(n,np.array([1,0])))
                
                phi = np.mean([phi_1,phi_2])
                
                phi_std = np.abs(phi_2-phi_1)
                
                
                B_MP=B_MP*1E-9
                
                pmag=B_MP**2/(2*munaught)
                
                pmag_std=pmag*B_MP_std*1E-9/B_MP*2
                

                
                B_SW=B_SW*1E-9
                
                B_MP_std=B_MP_std*1E-9
            
                B_SW_std = B_SW_std*1E-9
                
                rel_uncertainty_mp = B_MP_std/B_MP
                
                rel_uncertainty_sw = B_SW_std/B_SW
                
                psw1=(B_MP**2/(2*munaught) - B_SW**2/(2*munaught))*(.88*np.cos(phi)**2)**(-1)
               
                psw_max =  ((B_MP+B_MP_std)**2/(2*munaught) - (B_SW-B_SW_std)**2/(2*munaught))*(.88*np.cos(np.max([phi_1,phi_2]))**2)**(-1)
               
                psw_min =  ((B_MP-B_MP_std)**2/(2*munaught) - (B_SW+B_SW_std)**2/(2*munaught))*(.88*np.cos(np.min([phi_1,phi_2]))**2)**(-1) 
               
                
               
                psw_std = psw_max-psw1
               
                if ((phi >= 120*np.pi/180) | (phi <= np.radians(60))):
                
                    MA = np.append(MA,np.sqrt(psw1*munaught)/B_SW)
                    
                    MA_min = np.sqrt(psw_min*munaught)/(B_SW+B_SW_std)
                    
                    MA_max = np.sqrt(psw_max*munaught)/(B_SW-B_SW_std)
                    
                    MA_std = np.append(MA_std,MA_max-MA_min)
                    
                    pmag_total = np.append(pmag_total, pmag)
                    psw_total = np.append(psw_total,psw1)
                    
                    B_SW_total = np.append(B_SW_total, B_SW)
                    
                    Bx_total = np.append(Bx_total, B_X)
                    
                    Bx_std_total = np.append(Bx_std_total, B_X_std)
                    
                    By_std_total = np.append(By_std_total, B_Y_std)
                    
                    Bz_std_total = np.append(Bz_std_total, B_X_std)
                    
                    By_total = np.append(By_total, B_Y)
                    
                    Bz_total = np.append(Bz_total, B_Z)
                    
                    phi_total = np.append(phi_total,phi)
                    
                    B_SW_std_total = np.append(B_SW_std_total, B_SW_std)
                    
                    pmag_std_total = np.append(pmag_std_total, pmag_std)
                    
                    phi_std_total = np.append(phi_std_total,phi_std)
                    
                    psw_std_total = np.append(psw_std_total,psw_std)
            
            return MA,psw_total,B_SW_total,np.stack((Bx_total,By_total,Bz_total)),pmag_total,phi_total, \
                MA_std,psw_std_total,B_SW_std_total,np.stack((Bx_std_total,By_std_total,Bz_std_total)),\
                    pmag_std_total,phi_std_total
        
        Ma,psw1,B_SW,B_IMF,pmag,phi,Ma_std,psw_std,B_SW_std,B_IMF_std,pmag_std,phi_std = estimate_MA()
        munaught=4*np.pi*1E-7
        return Ma,psw1*1E9,B_SW*1E9,B_IMF,pmag*1E9,180-phi*180/np.pi,Ma_std,psw_std*1E9,B_SW_std*1E9,B_IMF_std,pmag_std*1E9,phi_std*180/np.pi

    

    def add_onto_density_panel(ax,dates,df,col,siz,alph,labe):
        
        ax.scatter(dates,df.n,c=col,s=siz,alpha=alph, label = labe, zorder=3)
        
        diff=np.roll(dates,1)-dates
        
        #Find where model transitions from one region to another
        transitions=np.where(np.abs(diff) > 2)[0]
        
        if transitions[0]!=0:
            
            transitions=np.insert(transitions,0,0)
            
        if transitions[-1] != len(dates):
            transitions = np.insert(transitions,len(transitions),len(dates))
        
        #transitions=np.insert(transitions,len(transitions),len(diff)-1)
        
        for j in range(len(transitions)-1):
            
            si=transitions[j]
            
            fi=transitions[j+1]
            
            ax.plot(dates[si:fi],df.n.iloc[si:fi],color=col,alpha=alph)
            
        
        
    ax_3d=make_3d_mercury()
        
        #ax.scatter(ad.ephx,ad.ephy,ad.ephz,color='black',s=.3)
    
    r1_range = [165,212]
    
    r2_range = [230,250]
    
    r3_range = [250,260]
    
    r4_range = [260,290]
    
    fs = 18
    
    ax1,ax2,ax3=make_mercury()
    
    # All Data
    
    #fig, (ax1_mag,ax2_mag,ax3_mag,ax4_mag,ax_nobs,ax_HF,ax_cb)=plt.subplots(7,sharex=True,gridspec_kw={'height_ratios': [1,1,1,1,1,1,0.3]})
    
    # Only mag and n
    
    fig, (ax1_mag,ax2_mag,ax3_mag,ax4_mag,ax_nobs,ax_cb)=plt.subplots(6,sharex=True,gridspec_kw={'height_ratios': [1,1,1,1,1,0.3]})
    
    # Create a figure and axis objec
    
    # Plot the data
    
    # Set the x-axis label and format
    
    ax1_mag.axhline(y=0.0, color='black', linestyle='--',linewidth=1.0)
    ax2_mag.axhline(y=0.0, color='black', linestyle='--',linewidth=1.0)
    ax3_mag.axhline(y=0.0, color='black', linestyle='--',linewidth=1.0)
    
    #All Data
    #axes=[ax1_mag,ax2_mag,ax3_mag,ax4_mag,ax_nobs,ax_HF,ax_cb]
    
    # Only mag and n
    axes=[ax1_mag,ax2_mag,ax3_mag,ax4_mag,ax_nobs,ax_cb]
    
    ax1_mag.set_ylabel("$BX_{MSM'}$ (nT)",fontsize=fs-5)
    ax2_mag.set_ylabel("$BY_{MSM'}$ (nT)",fontsize=fs-5)
    ax3_mag.set_ylabel("$BZ_{MSM'}$ (nT)",fontsize=fs-5)
    ax4_mag.set_ylabel("$|B|$ (nT)",fontsize=fs-5)
    ax_nobs.set_ylabel('$n_{H^+} cm^{-3}$',fontsize=fs-5)
    ax_nobs.set_yscale('log')
    
    # ax_HF.set_ylabel('H+ Flux',fontsize=fs-5)
    # ax_HF.set_yscale('log')
    
    icme=pd.read_excel(ICME_List_file)
    
    time_a = icme[['ICME_A_YEAR','ICME_A_DOY','ICME_A_HH','ICME_A_MM','ICME_A_SS']]\
        .iloc[ii]
        
    time_e = icme[['ICME_E_YEAR','ICME_E_DOY','ICME_E_HH','ICME_E_MM','ICME_E_SS']]\
        .iloc[ii]
        
        
    def create_datestring(year, day_of_year, hour, minute, second):
        # Create a datetime object for January 1st of the given year
        date = datetime.datetime(int(year), 1, 1)
    
        # Add the number of days (day_of_year - 1) to get to the desired date
        date += datetime.timedelta(days=float(day_of_year) - 1, hours=float(hour), minutes=float(minute),seconds=float(second))
        
        def convert_datetime_to_string(time):
            dstring=time.strftime("%Y-%m-%d %H:%M:%S")
            return dstring
        
        return convert_datetime_to_string(date), date
    
    date_string_a, date_a = create_datestring(time_a[0], time_a[1], time_a[2], time_a[3], time_a[4])
    
    date_string_e, date_e = create_datestring(time_e[0], time_e[1], time_e[2], time_e[3], time_e[4])
    
    
    ax_ma=make_Ma_plot()
    
    def make_mag_and_eph_plot(df,df_nobs,df_HF,ylims,xlims,count,event_number,orbit_number,ganymede=False):
        
        diff = df.time.iloc[-1]-df.time
        
        diff_a = df.time.iloc[-1]-date_a
        
        diff_e = df.time.iloc[-1]-date_e
        
        dates =np.array([d.total_seconds() for d in diff])/60
        
        dates = np.max(dates)-dates
        
        diff_nobs = df.time.iloc[-1]-df_nobs.time
        
        dates_nobs = np.array([d.total_seconds() for d in diff_nobs])/60

        dates_nobs = np.max(dates)-dates_nobs
        
        diff_HF = df.time.iloc[-1]-df_HF.time
        
        dates_HF = np.array([d.total_seconds() for d in diff_HF])/60

        dates_HF = np.max(dates)-dates_HF
        
        dates_a = diff_a.total_seconds()/60
        
        dates_e = diff_e.total_seconds()/60

        
        if ranges==True:
            
            r_ranges=[r1_range,r2_range,r3_range,r4_range]
            
            colors = ['green', 'pink', 'orange','blue']
            
            for r in range(len(r_ranges)):
                
                ra=r_ranges[r]
                
                df_ranges = df[((dates > ra[0]) & (dates < ra[1]))]
                
                add_eph_onto_traj(ax1,ax2,ax3,df_ranges.ephx,df_ranges.ephy,df_ranges.ephz,colors[r],3,alpha_gray)
                    
                add_eph_onto_traj(ax_3d,ax2,ax3,df_ranges.ephx,df_ranges.ephy,df_ranges.ephz,colors[r],3,alpha_gray,d3=True)
                         
                         
                         
        
        
        CME_Colors=['darkorange','goldenrod','mediumpurple','mediumturquoise','brown','green']
        
        
        if (((df.time.iloc[0] < date_a) and (df.time.iloc[-1] > date_a)) or \
            ((df.time.iloc[0] < date_e) and (df.time.iloc[-1] > date_e))) or \
            ((df.time.iloc[0] > date_a) and (df.time.iloc[-1] < date_e)):
            
            gd=((df.time > date_a) & (df.time < date_e))    
            
            gd_nobs = ((df_nobs.time > date_a) & (df_nobs.time < date_e))    
            
            df_cme=df[gd]   
            
            df_nobs_cme = df_nobs
            
            gd_HF = ((df_HF.time > date_a) & (df_HF.time < date_e))   
            df_HF_cme = df_HF
    
            
            if len(df_cme)> 0.7*len(df):
                df_cme[((dates[gd]>105) & (dates[gd]<107))]=np.nan
                
                ax1_mag.plot(dates[gd],df_cme.magx,CME_Colors[count],linewidth=1,alpha=1, label = 'ICME Orbit ',zorder=3)
                
                ax2_mag.plot(dates[gd],df_cme.magy,CME_Colors[count],linewidth=1,alpha=1, label = 'ICME Orbit ',zorder=3)
                
                ax3_mag.plot(dates[gd],df_cme.magz,CME_Colors[count],linewidth=1,alpha=1, label = 'ICME Orbit ',zorder=3)
                       
                ax4_mag.plot(dates[gd],df_cme.magamp,CME_Colors[count],linewidth=1,alpha=1, label = 'ICME Orbit ',zorder=3)
                
                add_onto_density_panel(ax_nobs,dates_nobs[gd_nobs],df_nobs_cme,CME_Colors[count],1.5,1,'ICME Orbit')
                
                #add_onto_density_panel(ax_HF,dates_HF[gd_HF],df_HF_cme,CME_Colors[count],1.5,1,'ICME Orbit')
                
                
                Ma, psw, B_SW, B_IMF, B_MP, phi, Ma_std, psw_std, B_SW_std,B_IMF_std,B_MP_std,phi_std  = add_onto_boundary_plot(ax_c,df_cme,CME_Colors[count],50,1, 'ICME Orbit ',philpott=True)
                
                add_onto_MA_plot(ax_ma,Ma,psw, Ma_std, psw_std, CME_Colors[count],50,1, 'ICME Orbit ', '$P_{SW}^{dyn}$ (nPa)')
                
                add_onto_MA_plot(ax_ma,Ma,B_SW, Ma_std, B_SW_std, CME_Colors[count],50,1, 'ICME Orbit ', '$B_{IMF}$ (nT)')
                
                add_onto_MA_plot(ax_ma,Ma,B_MP, Ma_std, B_MP_std, CME_Colors[count],50,1, 'ICME Orbit ', '$P_{M}^{mag}$ (nPa)')
                
                add_onto_MA_plot(ax_ma,Ma, phi, Ma_std, phi_std, CME_Colors[count],50,1, 'ICME Orbit ', '$\theta$ (deg)')
                
                theta=np.arctan2(df_cme.magx,df_cme.magy)*180/np.pi
                
                phi=np.arctan2(df_cme.magy,df_cme.magz)*180/np.pi
                
                psi=np.arctan2(df_cme.magx,df_cme.magz)*180/np.pi
                
                #ax_new = ax1_mag.twiny()
                #new_tick_locations = np.linspace(0,700,num=15)  # Example new tick locations
                
                #new_df = df.iloc[::int((700*60)/14)]
                #new_tick_labels = np.array([convert_datetime_to_string(new_df.time.iloc[d])[11:16] for d in range(len(new_df))])    # Example new tick labels
                #ax_new.set_xticks(new_tick_locations)
                #ax_new.set_xticklabels(new_tick_labels)

                #ax_new.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
                #ax_new.spines['top'].set_color('royalblue')
                #ax_new.tick_params(axis='both', colors='royalblue')
                #ax_new.tick_params(axis='both', which='major', length=8)
                #ax_new.tick_params(axis='both', which='minor', length=4)
                
                #np.save('B_IMF_21.npy',B_IMF)
                
                if count == 0:
                    add_eph_onto_traj(ax1,ax2,ax3,df_cme.ephx,df_cme.ephy,df_cme.ephz,dates[gd],4,1,scat=True)

                        
                    add_eph_onto_traj(ax_3d,ax2,ax3,df_cme.ephx,df_cme.ephy,df_cme.ephz,dates[gd],4,1,scat=True,d3=True)
                    
                    
                    magx_kth_0,magy_kth_0,magz_kth_0,magamp_kth_0 =\
                        get_KTH_pred_from_MSM(df.ephx, df.ephy, df.ephz, convert_datetime_to_string(df.time.iloc[0])[0:10],0)
        
                    
                    magx_kth_100,magy_kth_100,magz_kth_100,magamp_kth_100 =\
                        get_KTH_pred_from_MSM(df.ephx, df.ephy, df.ephz, convert_datetime_to_string(df.time.iloc[0])[0:10],100)
                    
                        
                    if ganymede==False:
                        ax1_mag.fill_between(dates,magx_kth_0,magx_kth_100, interpolate=True, color='red', alpha=0.5,label='KTH Model',zorder=2)
                        ax2_mag.fill_between(dates,magy_kth_0,magy_kth_100, interpolate=True, color='red', alpha=0.5,label='KTH Model',zorder=2)
                        ax3_mag.fill_between(dates,magz_kth_0,magz_kth_100, interpolate=True, color='red', alpha=0.5,label='KTH Model',zorder=2)
                        ax4_mag.fill_between(dates,magamp_kth_0,magamp_kth_100, interpolate=True, color='red', alpha=0.5,label='KTH Model',zorder=2)
                        
                    if ganymede == True:
                        ax1_mag.fill_between(dates,magx_kth_0/300,magx_kth_100/300, interpolate=True, color='red', alpha=0.5,label='KTH Model',zorder=2)
                        ax2_mag.fill_between(dates,magy_kth_0/300,magy_kth_100/300, interpolate=True, color='red', alpha=0.5,label='KTH Model',zorder=2)
                        ax3_mag.fill_between(dates,magz_kth_0/300,magz_kth_100/300, interpolate=True, color='red', alpha=0.5,label='KTH Model',zorder=2)
                        ax4_mag.fill_between(dates,magamp_kth_0/300,magamp_kth_100/300, interpolate=True, color='red', alpha=0.5,label='KTH Model',zorder=2)
                        
                        gm = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/7105334/ganymede_model.pkl')
                        
                        ax1_mag.plot(dates,gm.magx/750,color='black',label = 'Ganymede')
                        ax2_mag.plot(dates,gm.magy/750,color='black',label = 'Ganymede')
                        ax3_mag.plot(dates,gm.magz/750,color='black',label = 'Ganymede')
                        ax4_mag.plot(dates,gm.magamp/750,color='black',label = 'Ganymede')
                        
                    df[['magx_kth_100','magy_kth_100','magz_kth_100','magamp_kth_100']] = np.transpose(np.vstack((magx_kth_100,magy_kth_100,magz_kth_100,magamp_kth_100)))
                    
                    df[['magx_kth_0','magy_kth_0','magz_kth_0','magamp_kth_0']] = np.transpose(np.vstack((magx_kth_0,magy_kth_0,magz_kth_0,magamp_kth_0)))
                    for x in axes:
                        
                        shade_in_time_series(x,dates_a,dates_e,'maroon')
                    
                    if conduct==True:
                        
                        x_diff=df_cme.magx-np.mean(np.stack((magx_kth_0[gd],magx_kth_100[gd])),axis=0)
                        y_diff=df_cme.magy-np.mean(np.stack((magy_kth_0[gd],magy_kth_100[gd])),axis=0)
                        z_diff=df_cme.magz-np.mean(np.stack((magz_kth_0[gd],magz_kth_100[gd])),axis=0)
                        
                        mag_amp_diff=np.sqrt(x_diff**2+y_diff**2+z_diff**2)
                        
                        ax1_mag.plot(dates[gd],x_diff,color='green',linewidth=1,alpha=1,label='Difference')
                        ax2_mag.plot(dates[gd],y_diff,color='green',linewidth=1,alpha=1,label='Difference')
                        ax3_mag.plot(dates[gd],z_diff,color='green',linewidth=1,alpha=1,label='Difference')
                        ax4_mag.plot(dates[gd],mag_amp_diff,color='green',linewidth=1,alpha=1,label='Difference')
                        
                        fig,(ax1_con,ax2_con,ax3_con,ax4_con,ax5_con) = plt.subplots(5,sharex=True)
                        
                        ax1_con.plot(dates[gd],x_diff,color='green',linewidth=1,alpha=1,label='Difference')
                        ax2_con.plot(dates[gd],y_diff,color='green',linewidth=1,alpha=1,label='Difference')
                        ax3_con.plot(dates[gd],z_diff,color='green',linewidth=1,alpha=1,label='Difference')
                        ax4_con.plot(dates[gd],mag_amp_diff,color='green',linewidth=1,alpha=1,label='Difference')
                        
                        ax1_con.plot(dates[gd],B_IMF[0,:]-x_diff,color='brown',linewidth=1,alpha=1,label='Difference')
                        ax2_con.plot(dates[gd],B_IMF[1,:]-y_diff,color='brown',linewidth=1,alpha=1,label='Difference')
                        ax3_con.plot(dates[gd],B_IMF[2,:]-z_diff,color='brown',linewidth=1,alpha=1,label='Difference')
                        
                        mag_IMF=np.sqrt(B_IMF[0,0]**2+B_IMF[1,0]**2+B_IMF[2,0]**2)
                        
                        ax4_con.plot(dates[gd],np.abs(mag_IMF-mag_amp_diff),color='brown',linewidth=1,alpha=1,label='Difference')
                        
                        beta=.017
                        
                        B0=mag_IMF
                        
                        sigma = np.abs(mag_IMF-mag_amp_diff)/(0.017*mag_IMF)
                        
                        ax5_con.plot(dates[gd],sigma,color='black',linewidth=1,alpha=1,label='sigma')
                        df[['dates','magx_diff','magy_diff','magz_diff','magamp_diff']] = np.transpose(np.vstack((dates[gd],x_diff,y_diff,z_diff,mag_amp_diff)))
                        
                        dates_aw=np.load('dates_21_aw.npy')
                        
                        ICME_21 = {'Ma':Ma,'psw':psw,'B_SW':B_SW,'B_IMF':B_IMF,'df':df,'aw':dates_aw, 'ICME_time':[date_a,date_e]}
                        import pickle
                        with open('ICME_21.pkl', 'wb') as pickle_file:
                            pickle.dump(ICME_21, pickle_file)
# =============================================================================
#                     ax5_mag.plot(dates[gd],theta,color='blue',label='$tan^{-1}(X/Y)$')
#                     ax5_mag.plot(dates[gd],phi,color='red',label='$tan^{-1}(Y/Z)$')
#                     ax5_mag.plot(dates[gd],psi,color='green',label='$tan^{-1}(X/Z)$')
# =============================================================================
                    
    
                    bounds = np.logspace(-2, -1, 15)
                    cmap = plt.get_cmap('jet', 15)
                    import matplotlib.colors as cl
                    norm = cl.BoundaryNorm(bounds, cmap.N)

                    ax_cb.imshow(dates[gd].reshape(1, -1), cmap=cmap, aspect='auto',extent=[dates[gd][0],dates[gd][-1],-1,1],label='Minutes')

                    ax_cb.set_yticks([])
                    
                    #ax_cb.set_label('Traj')
                    print('ICME Orbit '+str(i))
            

                    
                    
                    
                    
                    
                count=count+1
                
            else:
                
                add_eph_onto_traj(ax1,ax2,ax3,df.ephx,df.ephy,df.ephz,'gray',3,alpha_gray)
                    
                add_eph_onto_traj(ax_3d,ax2,ax3,df.ephx,df.ephy,df.ephz,'gray',3,alpha_gray,d3=True)
                
                Ma, psw, B_SW, B_IMF, B_MP, phi, Ma_std, psw_std, B_SW_std, B_IMF_std, B_MP_std, phi_std  =  add_onto_boundary_plot(ax_c,df,'gray',50,.7, 'Surrounding Orbits',philpott=True)
                
                add_onto_MA_plot(ax_ma,Ma,psw, Ma_std, psw_std,'gray',50,.7, 'Surrounding Orbits','$P_{SW}^{dyn}$ (nPa)')
                
                add_onto_MA_plot(ax_ma,Ma,B_SW, Ma_std, B_SW_std, 'gray',50,.7, 'Surrounding Orbits', '$B_{IMF}$ (nT)')
                
                add_onto_MA_plot(ax_ma,Ma,B_MP, Ma_std, B_MP_std, 'gray',50,.7, 'Surrounding Orbits', '$P_{M}^{mag}$ (nPa)')
                
                add_onto_MA_plot(ax_ma,Ma,phi, Ma_std, phi_std, 'gray',50,.7, 'Surrounding Orbits', '$\theta$ (deg)')

       
                ax1_mag.plot(dates,df.magx,'gray',linewidth=.75,alpha=.5,label='Surrounding Orbits',zorder=1)
                
                ax2_mag.plot(dates,df.magy,'gray',linewidth=.75,alpha=.5,label='Surrounding Orbits',zorder=1)
                
                ax3_mag.plot(dates,df.magz,'gray',linewidth=.75,alpha=.5,label='Surrounding Orbits',zorder=1)
                       
                ax4_mag.plot(dates,df.magamp,'gray',linewidth=.75,alpha=.5,label='Surrounding Orbits',zorder=1)
                
                add_onto_density_panel(ax_nobs, dates_nobs, df_nobs, 'gray', 1.5, .5, 'Surrounding Orbits')

                #add_onto_density_panel(ax_HF, dates_HF, df_HF, 'gray', 1.5, .5, 'Surrounding Orbits')
          

        else:
            add_eph_onto_traj(ax1,ax2,ax3,df.ephx,df.ephy,df.ephz,'gray',3,alpha_gray)
                
            add_eph_onto_traj(ax_3d,ax2,ax3,df.ephx,df.ephy,df.ephz,'gray',3,alpha_gray,d3=True)
            
            Ma, psw, B_SW, B_IMF, B_MP, phi, Ma_std, psw_std, B_SW_std, B_IMF_std, B_MP_std, phi_std =  add_onto_boundary_plot(ax_c,df,'gray',50,.7, 'Surrounding Orbits',philpott=True)
            
            add_onto_MA_plot(ax_ma,Ma,psw, Ma_std,psw_std,'gray',50,.7, 'Surrounding Orbits','$P_{SW}^{dyn}$ (nPa)')
            
            add_onto_MA_plot(ax_ma,Ma,B_SW, Ma_std, B_SW_std, 'gray',50,.7, 'Surrounding Orbits', '$B_{IMF}$ (nT)')
            
            add_onto_MA_plot(ax_ma,Ma,B_MP, Ma_std, B_MP_std, 'gray',50,.7, 'Surrounding Orbits', '$P_{M}^{mag}$ (nPa)')
            
            add_onto_MA_plot(ax_ma,Ma,phi, Ma_std, phi_std, 'gray',50,.7, 'Surrounding Orbits', '$\theta$ (deg)')

   
            ax1_mag.plot(dates,df.magx,'gray',linewidth=.75,alpha=.5,label='Surrounding Orbits',zorder=1)
            
            ax2_mag.plot(dates,df.magy,'gray',linewidth=.75,alpha=.5,label='Surrounding Orbits',zorder=1)
            
            ax3_mag.plot(dates,df.magz,'gray',linewidth=.75,alpha=.5,label='Surrounding Orbits',zorder=1)
                   
            ax4_mag.plot(dates,df.magamp,'gray',linewidth=.75,alpha=.5,label='Surrounding Orbits',zorder=1)
            
            add_onto_density_panel(ax_nobs, dates_nobs, df_nobs, 'gray', 1.5, .5, 'Surrounding Orbits')
            
            #add_onto_density_panel(ax_HF, dates_HF, df_HF, 'gray', 1.5, .5, 'Surrounding Orbits')
            

        axes[-1].set_xlabel('Minutes from Apoapsis',fontsize=fs)
        for p in range(len(axes)):
            x=axes[p]
            
            x.set_ylim(ylims[p])
            
            if p == 0:
            
                handles, labels = x.get_legend_handles_labels()
                unique_labels = {}
                for handle, label in zip(handles, labels):
                    if label not in unique_labels:
                        unique_labels[label] = handle
                     
                x.legend(unique_labels.values(), unique_labels.keys(),loc='upper left',fontsize=fs-3)
            
            x.set_xlim(xlims)
            
            n=15
            
            xt = np.linspace(xlims[0],xlims[-1],num=n)
            
            xt = [int(round(p)) for p in xt]
            
            x.set_xticks(xt)
            
            # if orbit_number == 6:
                
            #     new_tick_locations = np.linspace(0,700,num=n)  # Example new tick locations
                
            #     new_df = df.iloc[::int((700*60)/14)]
                
            #     df_gd = df[((dates > xlims[0]) & (dates < xlims[-1]))]
                
            #     new_df = df_gd.iloc[::int((xlims[-1]-xlims[0])*60/(n-1))]
                
            #     new_tick_labels = np.array([convert_datetime_to_string(new_df.time.iloc[d])[11:16] for d in range(len(new_df))])    # Example new tick labels

            #     ax_new.set_xticks(new_tick_locations)
            #     ax_new.set_xticklabels(new_tick_labels)
                
            #     x.tick_params(labelsize=15)
                
            #     x.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
            #     x.yaxis.set_minor_locator(AutoMinorLocator())
                
            #     x.tick_params(axis='both', which='major', length=8)
            #     x.tick_params(axis='both', which='minor', length=4)
            
        axes_eph = [ax1,ax2,ax3]
        
        
        
        for p in range(len(axes_eph)):
            x=axes_eph[p]
            
            
            x.tick_params(labelsize=15)
            
            x.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
            x.yaxis.set_minor_locator(AutoMinorLocator())
            
            x.tick_params(axis='both', which='major', length=8)
            x.tick_params(axis='both', which='minor', length=4)
        
     
        icme_event = {
            
            'event_number': event_number,
            'orbit_number': orbit_number,
            'df': df,
            'Ma': Ma,
            'Ma_std': Ma_std,
            'psw':psw,
            'psw_std':psw_std,
            'B_IMF':B_IMF,
            'B_IMF_std':B_IMF_std,
            'B_SW':B_SW,
            'pmag':B_MP,
            'B_SW_std':B_SW_std,
            'pmag_std':B_MP_std,
            'min_from_ap':dates,
            'phi':phi,
            'phi_std':phi_std
            
        }
        
        import pickle
        file_path='ICME_Dictionary_'+str(event_number)+'_'+str(orbit_number)+'.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(icme_event, file)

        
        
        return axes_eph,axes,count
    
    
    # Full Data
    
    xlims=[0,700]
    
    
    # All Data
    
    #ylims=[[-140,140],[-140,140],[-140,140],[0,300],[0,50],[1E7,1E11],[-1,1]]
    
    # Only Mag and n
    
    ylims=[[-140,140],[-140,140],[-140,140],[0,300],[0,50],[-1,1]]
    
    count=0
    for i in range(11):
        
        df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_'+str(ii)+'_'+str(i)+'.pkl')
        df_nobs = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_NOBS_'+str(ii)+'_'+str(i)+'.pkl')
        df_HF = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_HF_'+str(ii)+'_'+str(i)+'.pkl')
        
        if ganymede == False:
            axes_eph,axes_mag,count = make_mag_and_eph_plot(df,df_nobs,df_HF,ylims,xlims,count,ii,i,ganymede = False)
        
        if ganymede==True:
            
            df[['magx','magy','magz','magamp']] = df[['magx','magy','magz','magamp']]/300
            
            ylims=[[-1,1],[-1,1],[-1,1],[0,1],[0,50],[1E7,1E11],[-1,1]]
            
            axes_eph,axes_mag,count = make_mag_and_eph_plot(df,df_nobs,df_HF,ylims,xlims,count,ii,i,ganymede = True)
            
                
            
                

        
        #axes_mag[0].set_xlim(xlims)
        df_nobs = df_nobs[(df_nobs.qual ==0)]
        
        if ((i ==0) & (ii==21)) :
            
            for p in range(len(axes_mag)-1):
                dates_aw=np.load('dates_21_aw.npy')
                
                shade_in_time_series(axes_mag[p],dates_aw[0],dates_aw[-1],'lightsteelblue',alpha=0.5,shade=True)

def mhd_visualizer():
    
    
    ''' Figure 4 in the manuscript'''
    
    
    from scipy.interpolate import griddata
    result_file = '/Users/bowersch/Desktop/Python_Code/Alfven_Wing/MHDSim_MESSENGER_ICME_20111230_B_V_PlasmaDensity_XZ_run2.dat'
    
    df = pd.read_csv(result_file,delimiter = ' ')
    
    X_unique = np.sort(df['X'].unique())
    Z_unique = np.sort(df['Z'].unique())
    X, Z = np.meshgrid(X_unique, Z_unique)
    
    # Reshape Density and Velocity to match the meshgrid
    n = df.pivot(index='Z', columns='X', values='Density').values
    ux = df.pivot(index = 'Z',columns = 'X',values = 'U_x')
    uy = df.pivot(index = 'Z',columns = 'X',values = 'U_y')
    uz = df.pivot(index = 'Z',columns = 'X',values = 'U_z')
    
    Bx = df.pivot(index = 'Z',columns = 'X',values = 'B_x')
    By = df.pivot(index = 'Z',columns = 'X',values = 'B_y')
    Bz = df.pivot(index = 'Z',columns = 'X',values = 'B_z')
    
    magamp = np.sqrt(Bx**2+By**2+Bz**2)
    
    X_s,Z_s = np.meshgrid(np.linspace(-25,15,num=len(X_unique)),np.linspace(-30,30,num=len(Z_unique)))
    

    u = np.stack((ux,uy,uz),axis=2)
    
    u_mag = np.linalg.norm(u,axis=2)
    B = np.stack((Bx,By,Bz),axis=2)
    
    E = np.cross(-u*1E3,B*1E-9)
    
    mp = 1.67E-27
    
    munaught = 4*np.pi * 1E-7
    
    rho = n*mp*100**3
    
    va_x = Bx*1E-9/np.sqrt(munaught*rho)/1000
    
    va_y = By*1E-9/np.sqrt(munaught*rho)/1000
    
    va_z = Bz*1E-9/np.sqrt(munaught*rho)/1000
    
    va = np.stack((va_x*1000,va_y*1000,va_z*1000),axis=2)
    
    c_plus = u+va
    
    iv = {'BX':19,'BY':0,'BZ':-60,'n':27,'ux':-400,'uy':50,'uz':0}
    
    va_iv = np.array([iv['BX']*1E-9/np.sqrt(iv['n']*mp*100**3),\
                     iv['BY']*1E-9/np.sqrt(iv['n']*mp*100**3),\
                     iv['BZ']*1E-9/np.sqrt(iv['n']*mp*100**3)])
        
    ua_iv = np.array([iv['ux'],iv['uy'],iv['uz']])
    
    prop_angle = np.arccos(np.dot([19,0,-60],[-400,0,0])/(np.linalg.norm([19,0,-60])\
                                                          *400))

    sigma_plus = 1/(munaught*np.linalg.norm(va_iv*1000)*np.sqrt(1+1.5**2-2*1.5*np.sin(prop_angle)))
    
    c_mag = np.linalg.norm(c_plus,axis=2)
    
    c_mag = np.stack((c_mag,c_mag,c_mag),axis=2)
    
    B_perp = munaught*sigma_plus*np.cross(c_plus/c_mag,E)*1E9
    
    
    def make_vis(ax,var,mi,ma,cma,labe,log=False,MESS=True):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        if log ==True:
            fig1 = ax.pcolormesh(X,Z,var,norm=colors.LogNorm(vmin=mi,vmax=ma),cmap=cma)
            
        else:
            fig1 = ax.pcolormesh(X,Z,var,vmin=mi,vmax=ma,cmap=cma)
        ax.tick_params(which='both', labelbottom=True)
        ax.tick_params(which='both',labelleft=True)
        
        fs=18
        divider1 = make_axes_locatable(ax)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cbar1 = plt.colorbar(fig1, cax=cax1)
        cbar1.set_label(labe,fontsize=fs-4)
        cbar1.ax.tick_params(labelsize=fs-6)
        #fig.colorbar(fig1)
        ax.set_aspect('equal',adjustable='box')
        ax.set_ylim([-8,5])
        ax.set_xlim([-8,5])
        
        sp = ax.streamplot(X_s, Z_s, Bx, Bz, color='black',density=2.1,linewidth=.4,
                      broken_streamlines=False,zorder=4,arrowstyle='-')
        
        
        #Plot Mercury
        
        theta = np.linspace(0, 2*np.pi, 1000)
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Plot the circle in all 3 plots
        ax.plot(x, y, color='gray',zorder=1)
        # Color the left hemisphere red and the right hemisphere gray
        ax.fill_between(x,y,color='white',interpolate=True,zorder=1)
        ax.fill_between(x, y, where=x<0, color='black', interpolate=True,zorder=1)

        
        ax.set_xlabel("$X_{MSO'}$",fontsize=fs-2)
        
        ax.set_ylabel("$Z_{MSO'}$",fontsize=fs-2)
        
        streamQuiver(ax,sp,n=10)
        format_ticks(ax,fs)
        
        #icme = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Dictionary_21_6.pkl')
        
        
        
        if MESS==True:
            df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_'+str(21)+'_'+str(6)+'.pkl')
            diff = df.time.iloc[-1]-df.time

            dates =np.array([d.total_seconds() for d in diff])/60
            bounds = np.logspace(-2, -1, 15)
            cmap = plt.get_cmap('jet', 15)
            import matplotlib.colors as cl
            norm = cl.BoundaryNorm(bounds, cmap.N)
                
            #ax.scatter(df.ephx,df.ephz+.19,c=np.flip(dates),s=2,cmap=cmap,zorder=3)
            ax.plot(df.ephx,df.ephz+.19,color='black')
            make_alfven_wing(1.5,ax,1,[19,0,-60])
            make_alfven_wing(1.5,ax,-1,[19,0,-60])
            
                

            
        
        #make_alfven_wing(1.5,ax,1,[19,0,-60])
        #make_alfven_wing(1.5,ax,-1,[19,0,-60])
        
    fig, axes = plt.subplots(2,2,sharex=True,sharey=True)   
    make_vis(axes[0,0],n,1E-2,100,'plasma','n (cm$^{-3}$)',log=False)
    make_vis(axes[1,0],ux,-500,500,'bwr','$u_X$ (km/s)',log=False)
    make_vis(axes[0,1],uy,-500,500,'bwr','$u_Y$ (km/s)',log=False)
    make_vis(axes[1,1],uz,-500,500,'bwr','$u_Z$ (km/s)',log=False)
    
    # fig,axes = plt.subplots(1,3,sharex=True,sharey=True)
    
    # make_vis(axes[0],B_perp[:,:,0],-100,100,'bwr','$B^X_{perp}$',log=False)
    # make_vis(axes[1],B_perp[:,:,1],-100,100,'bwr','$B^Y_{perp}$',log=False)
    # make_vis(axes[2],B_perp[:,:,2],-100,100,'bwr','$B^Z_{perp}$',log=False)
    
    fig, axes = plt.subplots(3,2,sharex=True,sharey=True)
    
    mp = 1.67E-27
    
    munaught = 4*np.pi * 1E-7
    
    rho = n*mp*100**3
    
    va_x = Bx*1E-9/np.sqrt(munaught*rho)/1000
    
    va_z = Bz*1E-9/np.sqrt(munaught*rho)/1000
    
    make_vis(axes[0,0],ux,-500,500,'bwr','$u_X$ (km/s)',log=False)
    make_vis(axes[0,1],uz,-500,500,'bwr','$u_Z$ (km/s)',log=False)
    
    make_vis(axes[1,0],va_x,-1000,1000,'bwr','$VA_X$ (km/s)',log=False)
    make_vis(axes[1,1],va_z,-1000,1000,'bwr','$VA_Z$ (km/s)',log=False)
    
    
    make_vis(axes[2,0],ux+va_x,-1000,1000,'bwr','$C^X_A+$ (km/s)')
    make_vis(axes[2,1],uz+va_z,-1000,1000,'bwr','$C^Z_A+$ (km/s)')
           

    
    fig, axes = plt.subplots(3,2,sharex=True,sharey=True)
    
    make_vis(axes[0,0],(ux-ua_iv[0]),-500,500,'bwr','$u_X$ - $u^{sw}_X$ (km/s)',log=False)
    make_vis(axes[0,1],(uz-ua_iv[2]),-500,500,'bwr','$u_Z$ - $u^{sw}_Z$  (km/s)',log=False)
    
    make_vis(axes[1,0],va_x-va_iv[0],-1000,1000,'bwr','$VA_X$ - $VA^{sw}_X$ (km/s)',log=False)
    make_vis(axes[1,1],va_z-va_iv[2],-1000,1000,'bwr','$VA_Z$ - $VA^{sw}_Z$ (km/s)',log=False)
    
    
    make_vis(axes[2,0],ux+va_x-(ua_iv[0]+va_iv[0]),-1000,1000,'bwr','$C^X+$ - $C^X_{SW}+$ (km/s)')
    make_vis(axes[2,1],uz+va_z-(ua_iv[2]+va_iv[2]),-1000,1000,'bwr','$C^Z+$ - $C^Z_{SW}+$ (km/s)')
    
    fig,ax = plt.subplots(1)
    
    make_vis(ax,n-iv['n'],-10,10,'bwr','$n$-$n_sw$')
    
    fig, axes = plt.subplots(2,2,sharex=True,sharey=True)
    
    make_vis(axes[0,0],Bx,-150,150,'bwr','$B_X$ (nT)',log=False)
    make_vis(axes[1,0],By,-150,150,'bwr','$B_Y$ (nT)',log=False)
    make_vis(axes[0,1],Bz,-150,150,'bwr','$B_Z$ (nT)',log=False)
    make_vis(axes[1,1],magamp,15,300,'plasma','$|B|$ (nT)',log=True)
    
    
    fig, axes = plt.subplots(3,sharex=True,sharey=True)
    
    make_vis(axes[0],magamp,15,300,'plasma','$|B|$ (nT)',log=True)
    
    make_vis(axes[1],n,1E-2,100,'plasma','n (cm$^{-3}$)',log=True)
    
    make_vis(axes[2],u_mag,0,700,'jet','$|u|$ (km/s)',log=False)
    
    
    
    
    
    # fig, axes = plt.subplots(1,3,sharex=True,sharey=True)
    
    # make_vis(axes[0],n,1E-2,100,'plasma','n (cm$^{-3}$)',log=True)
    # make_vis(axes[1],magamp,0,300,'viridis','$|B|$ (nT)',log=False)
    # make_vis(axes[2],u_mag,0,700,'jet','$|u|$ (km/s)',log=False)
    
    fig, axes = plt.subplots(3,3,sharex=True,sharey=True)
    
    make_vis(axes[0,0],Bx,-150,150,'bwr','$B_X$ (nT)',log=False)
    make_vis(axes[0,1],By,-150,150,'bwr','$B_Y$ (nT)',log=False)
    make_vis(axes[0,2],Bz,-150,150,'bwr','$B_Z$ (nT)',log=False)
    make_vis(axes[1,0],magamp,15,300,'plasma','$|B|$ (nT)',log=True)
    make_vis(axes[1,1],n,1E-2,100,'plasma','n (cm$^{-3}$)',log=True)
    make_vis(axes[1,2],u_mag,0,800,'plasma','$|u|$ (km/s)',log=False)
    
    make_vis(axes[2,0],ux,-500,500,'bwr','$u_X$ (km/s)',log=False)
    make_vis(axes[2,1],uy-50,-500,500,'bwr','$u_Y$ (km/s)',log=False)
    make_vis(axes[2,2],uz,-500,500,'bwr','$u_Z$ (km/s)',log=False)
    
    for x in range(3):
        for y in range(3):
            if y!= 0:
                
                axes[x,y].set_ylabel(' ')
            
            if x!=2:
                
                axes[x,y].set_xlabel(' ')

def specific_multi_orbit(ii,full=False,dayside=False,nightside=False):
    
    '''FIGURES 5, 6, and 9 in the manuscript
    
    Figure 5: specific_multi_orbit(21, full=True)
    
    Figure 6: specific_multi_orbit(21,dayside=True)
    
    Figure 9: specific_multi_orbit(21,nightside = True)
    
    '''
    
    date_range_full = [281,521]
    if full==True:
        number_for_top_axis = 20
        date_range = [283,520]
        k = False
        ylims = [[-150,150],[-150,150],[-150,150],[0,300],[.05,150],[-1,1]]
    
    if dayside == True:
        number_for_top_axis = 15
        date_range = [283,351]
        k = False
        ylims = [[-150,150],[-150,150],[-150,150],[0,300],[.05,150],[-1,1]]
        
    if nightside == True:
        number_for_top_axis=15
        date_range = [358,433]
        k = False
        ylims = [[-80,80],[-80,80],[-80,80],[0,160],[.01,150],[0,90],[-1,1]]
        
        
    
    else:
        number_for_top_axis = 15
        k=False
        ylims = [[-150,150],[-150,150],[-150,150],[0,300],[.05,150],[-1,1]]
    
    ax1,ax2,ax3 = make_mercury()
    
    ax_eph = [ax1,ax2,ax3]
    
    for x in ax_eph:
        format_ticks(x,18)
    def make_mag_and_eph_plot(variables,event_number,orbit_number,axis):
        

            

        
        diff = df.time.iloc[-1]-df.time
        
        
        dates =np.array([d.total_seconds() for d in diff])/60
        
        dates = np.max(dates)-dates
        
        diff_nobs = df.time.iloc[-1]-df_nobs.time
        
        dates_nobs = np.array([d.total_seconds() for d in diff_nobs])/60

        dates_nobs = np.max(dates)-dates_nobs
        
        diff_HF = df.time.iloc[-1]-df_HF.time
        
        dates_HF = np.array([d.total_seconds() for d in diff_HF])/60

        dates_HF = np.max(dates)-dates_HF
        
        CME_Colors=['darkorange','goldenrod','mediumpurple','mediumturquoise','brown','green']
        count=0
        alph = .2
        
        
        add_eph_onto_traj(ax1,ax2,ax3,df.ephx,df.ephy,df.ephz,'gray',3,.5)
        dr = ((dates > date_range_full[0]) & (dates <=date_range_full[1]))
        df_dr = df[dr]
        
        if orbit_number==6:
            add_eph_onto_traj(ax1,ax2,ax3,df_dr.ephx,df_dr.ephy,df_dr.ephz,dates[dr],4,1,scat=True)

            
        
        
        for i in range(len(variables)):
        
    
                
            if orbit_number==6:
                
                df_cme=df
                
                df_nobs_cme = df_nobs
                  
                df_HF_cme = df_HF
        
                df_cme[((dates>105) & (dates<107))]=np.nan
                
                
                    
                

                
                if len(dates)==len(variables[i]):

                    axis[i].plot(dates,variables[i],CME_Colors[count],linewidth=1,alpha=1, label = 'ICME Orbit ',zorder=3)
                
                    if i ==0:
                        
                        
                        ax_new = axis[i].twiny()
                        new_tick_locations = np.linspace(0,date_range[1],num=number_for_top_axis)  # Example new tick locations
                        dr = date_range[1]-date_range[0]
                        
                        df_range = df[((dates > date_range[0]) & (dates < date_range[1]))]
                            
                        new_df = df_range.iloc[::int((dr*60)/(number_for_top_axis-1))]
                        new_tick_labels = np.array([convert_datetime_to_string(new_df.time.iloc[d])[11:16] for d in range(len(new_df))])    # Example new tick labels
                        ax_new.set_xticks(new_tick_locations,labels = new_tick_labels)
                        #ax_new.set_xticklabels(new_tick_labels)
                        ax_new.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
                        ax_new.spines['top'].set_color('darkorange')
                        ax_new.tick_params(axis='both', colors='darkorange')
                        ax_new.tick_params(axis='both', which='major', length=8)
                        ax_new.tick_params(axis='both', which='minor', length=4)
                        format_ticks(ax_new,15)
                        
                        
                        
                        
                    
                    
                       
                
                if len(dates_nobs) == len(variables[i]):
                    
                    
                    
                    add_onto_density_panel(axis[i],dates_nobs,variables[i],CME_Colors[count],5,1,'ICME Orbit')
                    
                    
                    if orbit_number == 6:
                        #add_onto_density_panel(axis[i],dates_nobs[(df_nobs.qual==1.0)],variables[i][(df_nobs.qual==1.0)],'red',5,1,'Surrounding Orbits')
                        
                        if 'Class' in df_nobs.columns:
                            
                            axis[i].scatter(dates_nobs[(df_nobs.Class==0)],variables[i][(df_nobs.Class==0)],color='red',s=9,marker = "^",zorder=10)
                            
                        else:
                            axis[i].scatter(dates_nobs[(df_nobs.qual==1.0)],variables[i][(df_nobs.qual==1.0)],color='red',s=9,marker = "^",zorder=10)
                            
                    axis[i].set_yscale('log')
                #magx_kth_0,magy_kth_0,magz_kth_0,magamp_kth_0 =\
                    #get_KTH_pred_from_MSM(df.ephx, df.ephy, df.ephz, convert_datetime_to_string(df.time.iloc[0])[0:10],0)
    
                
                #magx_kth_100,magy_kth_100,magz_kth_100,magamp_kth_100 =\
                    #get_KTH_pred_from_MSM(df.ephx, df.ephy, df.ephz, convert_datetime_to_string(df.time.iloc[0])[0:10],100)   
                    
                    
                
            else:
                
                if len(dates) == len(variables[i]):
                    
                    axis[i].plot(dates,variables[i],'gray',linewidth=.75,alpha=alph,label='Surrounding Orbits',zorder=1)
                    
                if len(dates_nobs)==len(variables[i]):
                    add_onto_density_panel(axis[i],dates_nobs,variables[i],'gray',.8,alph,'Surrounding Orbits')
                    
                    #if 'Class' in df_nobs.columns:
                        
                        #axis[i].scatter(dates_nobs[(df_nobs.Class==0)],variables[i][(df_nobs.Class==0)],color='red',s=6,marker = "^",zorder=10,alpha=.5)
                        
                    #else:
                        #axis[i].scatter(dates_nobs[(df_nobs.qual==1.0)],variables[i][(df_nobs.qual==1.0)],color='red',s=6,marker = "^",zorder=10,alpha=.5)
                    
                    
        bounds = np.logspace(-2, -1, 15)
        cmap = plt.get_cmap('jet', 15)
        import matplotlib.colors as cl
        norm = cl.BoundaryNorm(bounds, cmap.N)

        dates = dates[((dates > date_range_full[0]) & (dates <= date_range_full[1]))]
        axis[len(axis)-1].imshow(dates.reshape(1, -1), cmap=cmap, aspect='auto',extent=[dates[0],dates[-1],-1,1],label='Minutes')

        axis[len(axis)-1].set_yticks([])
        
        
        return axis
    
    xlims=date_range
    
    
    

    
    # ylims=[[0,300],[.01,20],[-1,1]]
    
    # ylabels = ['$|B|$ (nT)',"n ($cm^{-3}$)"," "]
    
    
    if nightside == True:
        
        ylabels = ["$BX_{MSM'}$ \n [nT]","$BY_{MSM'}$ \n [nT]","$BZ_{MSM'}$ \n [nT]","|$B$| \n [nT]", 'n \n [cm$^{-3}$]','$\Psi$ (deg)',' ']#,'V_A','Beta','v_estimate','Ma',' ']
    
        scales = ['linear','linear','linear','linear','log','linear','linear']
    else:
        ylabels = ["$BX_{MSM'}$ \n [nT]","$BY_{MSM'}$ \n [nT]","$BZ_{MSM'}$ \n [nT]","|$B$| \n [nT]", 'n \n [cm$^{-3}$]',' ']
        
        scales = ['linear','linear','linear','linear','log','linear']
    
   
     
    
    
    for i in range(11):
        
        df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_'+str(ii)+'_'+str(i)+'.pkl')
        df_nobs = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_NOBS_'+str(ii)+'_'+str(i)+'.pkl')
        df_HF = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_HF_'+str(ii)+'_'+str(i)+'.pkl')
        
        if k==True:
            df_nobs = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_K_'+str(ii)+'_'+str(i)+'.pkl')
        #df_nobs = df_nobs[(df_nobs.qual ==0)]
        #variables = [df.magamp,df_nobs.n]#,df_nobs.va,df_nobs.beta,df_nobs.v_estimate,df_nobs.v_estimate/df_nobs.va]
        
        if nightside==True:
            variables = [df.magx,df.magy,df.magz,df.magamp,df_nobs.n,np.abs(np.arctan2(np.abs(df.magz),np.abs(df.magx))*180/np.pi)]
        else:
            variables = [df.magx,df.magy,df.magz,df.magamp,df_nobs.n]
        
        if i==0:
            
            panel_number = len(variables)
            
            h_ratios = np.zeros(panel_number)+1
            
            h_ratios = np.append(h_ratios,.3)
            fig, axis =plt.subplots(panel_number+1,sharex=True,gridspec_kw={'height_ratios': h_ratios})
        
            
        axes_mag = make_mag_and_eph_plot(variables,ii,i,axis)
        
        
        
        for x in range(len(axes_mag)):
            
            axes_mag[x].set_ylim(ylims[x])
            format_ticks(axes_mag[x],18)
            axes_mag[x].set_yscale(scales[x])
            axes_mag[x].set_xlim(xlims)
            
            if x == len(axes_mag)-1:
                axes_mag[x].set_xlabel('Minutes after Apoapsis',fontsize=15)
                
            axes_mag[x].set_ylabel(ylabels[x],fontsize=15)
            
    dates_a, magamp_averaged, dates_n, n_averaged = average_parameters(ii,k)
    axis[-1].set_yticks([])
    data_frame = pd.DataFrame(data={'n_averaged':n_averaged})
    
    for p in range(4):
        
        axes_mag[p].plot(dates_a,magamp_averaged[:,p],color='black',label = 'Average Surrounding Orbits')
        
    if nightside ==True:
        
        axes_mag[5].plot(dates_a,np.abs(np.arctan2(np.abs(magamp_averaged[:,2]),np.abs(magamp_averaged[:,0]))*180/np.pi),color='black')
    
        add_onto_density_panel(axes_mag[len(axes_mag)-3],dates_n,data_frame.n_averaged,'black',.5,1,'Averaged Surrounding Orbits')
        
    else:
        add_onto_density_panel(axes_mag[len(axes_mag)-2],dates_n,data_frame.n_averaged,'black',.5,1,'Averaged Surrounding Orbits')
    
    
    
    
    
    
    
   
    
    
    for p in range(len(axes_mag)):
        
        
        dates_aw=np.load('dates_21_aw.npy')
        
    
    # Load in MHD simulation
    mhd_run = pd.read_csv('/Users/bowersch/Desktop/DIAS/Other Papers/All Papers/Alfven_Wing/MHDSim_MESSENGER_ICME_20111230_MAG_Density_run2.dat',index_col=False)
    
    # Rename variables
    
    df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_'+str(ii)+'_'+str(6)+'.pkl')
    
    diff = df.time.iloc[-1]-df.time
    
    dates =np.array([d.total_seconds() for d in diff])/60
    
    dates = np.max(dates)-dates
    
    #df_nobs
    
    
    mhd_run = mhd_run.rename(columns={'Sim_Bx [nT]':'magx',
                                      'Sim_By [nT]':'magy'})
    
    mhd_run.columns = ['time','magx','magy','magz','magamp','n','ephx','ephy','ephz']
    
    time_datetime = np.array([convert_to_datetime(d) for d in mhd_run.time.to_numpy()])
    
    mhd_run.time = time_datetime
    
    dates_mhd = dates[((df.time >= mhd_run.time.iloc[0]) & (df.time <= mhd_run.time.iloc[-1]))]
    
    if nightside==True:
        
        mhd_variables = [mhd_run.magx,mhd_run.magy,mhd_run.magz,mhd_run.magamp,mhd_run.n,
                         np.abs(np.arctan2(np.abs(mhd_run.magz),np.abs(mhd_run.magx))*180/np.pi)]
        
    else:
        mhd_variables = [mhd_run.magx,mhd_run.magy,mhd_run.magz,mhd_run.magamp,mhd_run.n]
        
    
    icme = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Dictionary_21_6.pkl')

    df_icme = icme['df']
    
    var_0 = [df_icme.magx_kth_0,df_icme.magy_kth_0,df_icme.magz_kth_0,df_icme.magamp_kth_0]
    
    var_1 = [df_icme.magx_kth_100,df_icme.magy_kth_100,df_icme.magz_kth_100,df_icme.magamp_kth_100]
    
    
    for i in range(len(mhd_variables)):
        axes_mag[i].plot(dates_mhd,mhd_variables[i],color='darkorchid',label = 'MHD Model')
        
    for i in range(4):    
        axes_mag[i].axhline(y=0,linestyle='--',color='black')
    
    #for i in range(len(var_0)):
        #axes_mag[i].fill_between(dates,var_0[i],var_1[i], interpolate=True, color='mediumturquoise', alpha=0.5,label='KTH Model',zorder=2)
        
    handles, labels = axes_mag[0].get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
            
            
    # Plot legend with only unique labels
    # legend = axes_mag[0].legend(unique_labels.values(), unique_labels.keys(),loc='upper right',frameon=True,fontsize=12)
    
    # legend.get_frame().set_facecolor('white')  
    # legend.get_frame().set_alpha(1.0) 
        #if p < len(axes_mag)-1:
    
            #shade_in_time_series(axes_mag[p],dates_aw[0],dates_aw[-1],'lightsteelblue',alpha=0.3,shade=True)
    return

def generate_FIPS_flux_map_event(min_range,event_number, avg=False, Na = False):
    
    '''FIGURE 7 and 10 in the manuscript
    
    Figure 7: generate_FIPS_flux_map_event([310,335],6,avg=True)
    
    Figure 10: generate_FIPS_flux_map_events([385,410],6,avg=True)
    
    '''
    
    df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_21_'+str(event_number)+'.pkl')
    
    diff = df.time.iloc[-1]-df.time
    fr = [1E8,5E12]
    
    dates =np.array([d.total_seconds() for d in diff])/60
    
    dates = np.max(dates)-dates
    
    df['dates'] = dates
    
    df_oi = df[((df.dates > min_range[0]) & (df.dates < min_range[1]))]
    
    trange = [convert_datetime_to_string(df_oi.time.iloc[0]),convert_datetime_to_string(df_oi.time.iloc[-1])]
    
    
    print(trange)
    if Na == False:
        ax, flux_data = generate_FIPS_flux_map(trange,fr)
        
    if Na == True:
        ax, flux_data = generate_FIPS_flux_map(trange,fr,Na=True)
   # breakpoint()
    ax.set_title('Event number '+str(event_number))
    
    ax.set_title('ICME Event')
    
    ax = create_FIPS_plot(trange)
    
    ax[0].set_title('Event number '+str(event_number))
    
    if avg == True:
        flux_data_total = np.zeros((0,18,36))
        for i in range(11):
            
            if i !=6:
                df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_'+str(21)+'_'+str(i)+'.pkl')
                
                diff = df.time.iloc[-1]-df.time
                
                
                dates =np.array([d.total_seconds() for d in diff])/60
                
                dates = np.max(dates)-dates
                
                df['dates'] = dates
                
                df_oi = df[((df.dates > min_range[0]) & (df.dates < min_range[1]))]
                
                trange = [convert_datetime_to_string(df_oi.time.iloc[0]),convert_datetime_to_string(df_oi.time.iloc[-1])]
                
                
                print(trange)
                
                if Na == False:
                    
                    ax, flux_data = generate_FIPS_flux_map(trange,fr)
                    
                if Na == True:
                    
                    ax, flux_data = generate_FIPS_flux_map(trange,fr,Na = True)
                    
                
                flux_data_total = np.append(flux_data_total,[flux_data],axis=0)
                
                ax.set_title('Event number '+str(i))
                
                
        flux_data = np.nanmean(flux_data_total,axis=0)
        
        ax = plot_on_spherical_mesh(flux_data, fr)
        
        ax.set_title('Surrounding Orbits')


def current_sheet_analysis(event,orbit):
    '''Figure 8 in the manuscript
    
    Figure 8: current_sheet_analysis(21,np.arange(11))
    
    '''
    
    fig,ax = plt.subplots(1)
    L = np.array([])
    z0_full = np.array([])
    
    def add_to_cs_plots(df_cme,o):
    
        diff = df_cme.time.iloc[-1]-df_cme.time
        #fig, (ax1_mag,ax2_mag,ax3_mag,ax4_mag,ax_cb)=plt.subplots(5,sharex=True,gridspec_kw={'height_ratios': [1,1,1,1,0.3]})
                     
        if o==6:
            
            col = 'darkorange'
            alph = 1
            zo = 3
            
            labe = 'ICME Event'
            
        else:
            
            col = 'gray'
            
            alph = .5
            
            zo = 1

            labe = 'Surrounding Orbit'
            
        if o == 'model':
            
            col = 'mediumpurple'
            
            alph = 1
            
            zo = 2

            labe = 'MHD Model'
            
            
                
            
    
        # ax_cb.set_yticks([])
        
        dates = np.flip(np.array([d.total_seconds() for d in diff])/60)
        
        # ax_cb.imshow(dates.reshape(1, -1), cmap='tab20', aspect='auto',extent=[dates[0],dates[-1],-1,1],label='Minutes')
        
        df_cme['dates']=dates
        
        df_cme[((dates>105) & (dates<107))]=np.nan
        
        # ax1_mag.plot(dates,df_cme.magx,col,linewidth=1,alpha=alph, label = 'ICME Orbit ',zorder=3)
        
        # ax2_mag.plot(dates,df_cme.magy,col,linewidth=1,alpha=alph, label = 'ICME Orbit ',zorder=3)
        
        # ax3_mag.plot(dates,df_cme.magz,col,linewidth=1,alpha=alph, label = 'ICME Orbit ',zorder=3)
               
        # ax4_mag.plot(dates,df_cme.magamp,col,linewidth=1,alpha=alph, label = 'ICME Orbit ',zorder=3)
        
        ds = 1
        
        def downsample(new_resolution,arr):
            
            new_length = len(arr) // new_resolution
            
            arr_reshaped = arr[:new_length * new_resolution].reshape(new_length, new_resolution)
            
            arr_averaged= np.mean(arr_reshaped, axis=1)
            
            return arr_averaged
        
        
        df_cs = df_cme[((df_cme.ephz > -0.6) & (df_cme.ephz < .2) & (df_cme.ephx < 0))]
        
        ephz = df_cs.ephz.to_numpy()
        
        magx = df_cs.magx.to_numpy()
        
        ephz = downsample(ds,ephz)
        
        magx = downsample(ds, magx)
        
        
        ax.plot(ephz,magx,color = col, linewidth=1.4,alpha=alph, label = labe, zorder = zo)
        
        #B0 = 58
        
        magx_1 = df_cs.magx.to_numpy()
        
        magx_1 = downsample(ds,magx_1)
        
        def find_zero_crossings(array):
            # Ensure the array is a numpy array
            array = np.asarray(array)
            
            # Calculate the sign of each element
            signs = np.sign(array)
            
            # Find where the sign changes
            zero_crossings = np.where(np.diff(signs))[0][0]
    
            return zero_crossings
        
        crossing = find_zero_crossings(magx_1)
        
        z0_calc = np.mean([ephz[crossing],ephz[crossing+1]])

        
        from scipy.optimize import curve_fit
        
        def Harris_cs(x, B0, L, z0):
            return B0 * np.tanh((x - z0) / L)
        bounds = ([0, 0, -1], [100, 2, 1])
        
        # Perform curve fitting with bounds
        popt, popx = curve_fit(Harris_cs, ephz, magx, bounds=bounds)
    
        fit =  Harris_cs(ephz,popt[0],popt[1],popt[2])
        
        #breakpoint()
        
        
        
        #ax.plot(df_cs.ephz,Harris_cs(df_cs.ephz,popt[0],popt[1]),color='red')
       
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        #fit = fit/np.max(fit)
        
        #magx = magx/np.max(magx)
        
        chi_squared = np.sum((fit-magx)**2/np.abs(magx))
        
        
       #print('chi2 = '+str(chi_squared))
        
        # Calculate R^2
        R_squared = r2_score(magx, fit)
        
        #print('r2 = '+str(R_squared))
        
        # Calculate RMSE
        RMSE = np.sqrt(mean_squared_error(magx, fit))
        
        #print('RMSE = '+str(RMSE))
        
        
        # Calculate MAE
        MAE = mean_absolute_error(magx, fit)
        if o != 'model':
            icme_event=pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Dictionary_'+str(event)+'_'+str(o)+'.pkl')
            
            icme_event['L'] = popt[1]
            
            icme_event['z0'] = z0_calc
            
            icme_event['B0'] = popt[0]
            
            
            import pickle
            file_path='ICME_Dictionary_'+str(event)+'_'+str(o)+'.pkl'
            with open(file_path, 'wb') as file:
                pickle.dump(icme_event, file)
        
        if R_squared > 0.95:
            return popt[1]
        
        else:
            return np.nan
        
        # #print('MAE = '+str(MAE))
        # if R_squared > 0.95:
        #     L = np.append(L,popt[1])
            
        # else:
        #     L = np.append(L,np.nan)
        

    
    for o in orbit:
    
        df_cme = pd.read_pickle('ICME_Event_'+str(event)+'_'+str(o)+'.pkl')
        
        L1= add_to_cs_plots(df_cme,o)
        

        
        L = np.append(L,L1)
        
    mhd_run = pd.read_csv('/Users/bowersch/Desktop/DIAS/Other Papers/All Papers/Alfven_Wing/MHDSim_MESSENGER_ICME_20111230_MAG_Density_run2.dat',index_col=False)
    
    #df_nobs
    mhd_run = mhd_run.rename(columns={'Sim_Bx [nT]':'magx',
                                      'Sim_By [nT]':'magy'})
    
    mhd_run.columns = ['time','magx','magy','magz','magamp','n','ephx','ephy','ephz']
    mhd_run.ephz = mhd_run.ephz.to_numpy()-0.19
    time_datetime = np.array([convert_to_datetime(d) for d in mhd_run.time.to_numpy()])
    
    mhd_run.time = time_datetime
    L1 = add_to_cs_plots(mhd_run,'model')
    
    L = np.append(L,L1)
    
    
    
    def average_parameters_z(ii,K):
        
        def create_nobs_array(dates_nobs,df_nobs):
            
            dates_nobs_line = np.linspace(0,690,num=690//2)
            
            n_values = np.zeros(len(dates_nobs_line))+np.nan
            
            for i in range(len(dates_nobs)-1):
                
            
                xi = np.searchsorted(dates_nobs_line,dates_nobs[i])-1
                
                
                if np.abs(dates_nobs[i]-dates_nobs_line[xi]) < 2:
                       
                    n_values[xi] = df_nobs.n.iloc[i]

                    
                else:
                    
                    n_values[xi] = np.nan
                    
            non_nan = np.where(np.isnan(n_values)==False)[0]
            
            for i in np.arange(non_nan[0],non_nan[-1]):
                if np.isnan(n_values[i])==True:
                    n_values[i] = 0
                    
                
            return n_values
                
            
            
            

        for i in range(11):
            if i != 6:
                
                df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_'+str(ii)+'_'+str(i)+'.pkl')
                df = df[((df.ephz > -0.6) & (df.ephz < .2) & (df.ephx < 0))]
                
                
                dates = get_dates(df,0,0)
                

                
                
                
            
                
                if i == 0:
                    
                    gd = (dates<690)
                                        
                    df = df[gd]
                    
                    
                    magamp_averaged = np.zeros((0,len(df)))
                    
                    bx_averaged = np.zeros((0,len(df)))
                    by_averaged = np.zeros((0,len(df)))
                    bz_averaged = np.zeros((0,len(df)))
                    ephz_averaged = np.zeros((0,len(df)))
                    
                    dates_nobs_line = np.linspace(0,690,num=690//2)
                    
                    n_averaged = np.zeros((0,len(dates_nobs_line)))
                
                
                m = np.max(np.where(gd==True)[0])

               
                
                df = df.iloc[0:m+1]
                
              
                
                dates = dates[0:m+1]
                

                magamp_averaged = np.append(magamp_averaged,[df.magamp.to_numpy()],axis=0)
                
                bx_averaged = np.append(bx_averaged,[df.magx.to_numpy()],axis=0)
                by_averaged = np.append(by_averaged,[df.magy.to_numpy()],axis=0)
                bz_averaged = np.append(bz_averaged,[df.magz.to_numpy()],axis=0)
                
                ephz_averaged = np.append(ephz_averaged,[df.ephz.to_numpy()],axis=0)
                
            
            magamp_total = np.mean(magamp_averaged,axis=0)
            bx_total = np.mean(bx_averaged,axis=0)
            by_total = np.mean(by_averaged,axis=0)
            bz_total = np.mean(bz_averaged,axis=0)
            
            ephz_total = np.mean(ephz_averaged,axis=0)

            mag_total = np.transpose(np.vstack((bx_total,by_total,bz_total,magamp_total)))
            
            
        return bx_total, ephz_total
    
    bx_avg,ephz_avg = average_parameters_z(21,False)

    
    ax.plot(ephz_avg,bx_avg,color = 'black', linewidth=1.4,alpha=1,
            label = 'Mean Surrounding Orbits', zorder = 2)
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
    fs=19       
    #ax.legend(unique_labels.values(), unique_labels.keys(),loc='upper left',fontsize=fs-3)
    
    format_ticks(ax,fs)
    ax.set_xlabel("$Z_{MSM'}$ ($R_M$)",fontsize = fs-2)
    ax.set_ylabel("$BX_{MSM'}$ (nT)",fontsize = fs-2)
    ax.invert_xaxis()
    ax.axhline(y=0,color='black',linestyle= '--',linewidth=1.5)
    ax.axvline(x=0,color='black',linestyle= '--',linewidth=1.5)
    ax.set_xlim([0.2,-0.2])
    
    
    

    
    
    def event_properties(event,parameter):
        
        p = np.array([])
        
        for i in range(11):
            
            icme_event=pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Dictionary_'+str(event)+'_'+str(i)+'.pkl')
            try:
                p = np.append(p,icme_event[parameter][0])
            except:
                p = np.append(p,icme_event[parameter])
                           
        return p
    
    z0 = event_properties(21,'z0')
    
    
    z0 = np.append(z0, .023)
    
    err = event_properties(21,'Ma_std')
    
    err = np.append(err,0)
    
    
    
    fig,ax = plt.subplots(1)
    sz = 30
    
    ax.scatter(z0, L, color = 'gray', alpha = 0.5,label = 'Surrounding Orbits',s=sz)
    
    ax.scatter(z0[6], L[6], color = 'darkorange', alpha = 1, label = 'ICME Orbit',s=sz)
    
    ax.scatter(z0[-1], L[-1], color = 'mediumpurple', alpha = 1, label = 'MHD Model',s=sz)
    
    #ax.errorbar(z0,L,xerr=err,elinewidth=1.5,linestyle='None',color='gray',alpha=0.5)
    
    #ax.errorbar(z0[6],L[6],xerr=err[6],elinewidth=1.5,linestyle='None',color='darkorange',alpha=1)
   
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
     
    #ax.legend(unique_labels.values(), unique_labels.keys(),loc='upper left',fontsize=fs-3)
    ax.set_xlabel('$Z_0$ ($R_M$)',fontsize = fs-2)
    ax.set_ylabel('$L$ ($R_M$)',fontsize = fs-2)
    
    
    ax.set_xlim(-0.05,0.05)
    format_ticks(ax,fs)
        
    
        
    return

def add_eph_onto_traj(ax1,ax2,ax3,ephx,ephy,ephz,col,lw,alph,scat=False,d3=False):
    
    if scat==False:
        
        if d3==True:
            
            ax1.plot(ephx,ephy,ephz,color=col,linewidth=lw,alpha=alph,zorder=2)
            
        else:
            
            ax1.plot(ephx,ephz,color=col,linewidth=lw,alpha=alph,zorder=2)
            
            ax2.plot(ephx,ephy,color=col,linewidth=lw,alpha=alph,zorder=2)
            
            ax3.plot(ephy,ephz,color=col,linewidth=lw,alpha=alph,zorder=2)
        

        
    if scat==True:
        
        bounds = np.logspace(-2, -1, 15)
        cmap = plt.get_cmap('jet', 15)
        import matplotlib.colors as cl
        norm = cl.BoundaryNorm(bounds, cmap.N)
        
        if d3==True:
            
            ax1.scatter(ephx.to_numpy(),ephy.to_numpy(),ephz.to_numpy(),c=col,s=9,cmap=cmap,zorder=3)
        else:
            
            # ax1.scatter(ephx,ephz,c=col,s=6,cmap=cmap,zorder=3)
            
            # ax2.scatter(ephx,ephy,c=col,s=6,cmap=cmap,zorder=3)
            
            # ax3.scatter(ephy,ephz,c=col,s=4,cmap=cmap,zorder=3)
            
            ax1.scatter(ephx,ephz,c='darkorange',s=6,zorder=3)
            
            ax2.scatter(ephx,ephy,c='darkorange',s=6,cmap=cmap,zorder=3)
            
            ax3.scatter(ephy,ephz,c='darkorange',s=4,cmap=cmap,zorder=3)
            
            
                    
        
def shade_in_time_series(ax,tstart,tstop,color,shade=False,alpha=.05):
    import matplotlib.pyplot as plt
    
    y=np.array([-10000000000000,100000000000000])
    
    if shade==False:
    
        ax.axvline(tstart,color=color)
            
        ax.axvline(tstop,color=color)
    
    if shade==True:
        ax.fill_betweenx(y,tstart,tstop,color=color,alpha=alpha)
def get_eph_mso(date_string,res='01',full=False):
    
    doy=get_day_of_year(date_string)
    month=date_string[5:7]
    year=date_string[2:4]
    
    year_full=date_string[0:4]
    if doy < 10: doy_s='00'+str(doy)
        
    elif (doy<100) & (doy>=10):doy_s='0'+str(doy)
        
    else: doy_s=str(doy)
    
    
    
    #file='/Users/bowersch/Desktop/MESSENGER Data/mess-mag-calibrated avg/MAGMSOSCIAVG'+year+str(doy)+'_'+res+'_V08.TAB'
    
    file='/Users/bowersch/Desktop/MESSENGER Data/mess-mag-calibrated - avg'+year+'/'+month+'/'+'MAGMSOSCIAVG'+year+doy_s+'_'+res+'_V08.TAB'
    if full==True:
        file='/Users/bowersch/Desktop/MESSENGER Data/mess-mag-calibrated/MAGMSOSCI'+year+str(doy)+'_V08.TAB'
    df = np.genfromtxt(file)
    
    hour=df[:,2]
    
    #print(hour[0])
    
    minute=df[:,3]
    
    second=df[:,4]
    
    year=date_string[0:4]
    
    doy=int(doy_s)-1
    
 
   

    
    date=datetime.datetime(year=int(year),month=1,day=1)+datetime.timedelta(doy)
    
    #print(date)
    
    date2=[]
    
    for i in range(np.size(hour)):
        if int(hour[i])-int(hour[i-1]) < 0:
            
            doy=doy+1
            
            date=datetime.datetime(year=int(year),month=1,day=1)+datetime.timedelta(doy)
        
        date2.append(date+datetime.timedelta(hours=hour[i], minutes=minute[i], seconds=second[i]))
        
    #print(date2[0])
    
    #time=[d.strftime("%Y-%m-%d %H:%M:%S") for d in date2]
    
    #print(time[0])
    
    time=date2
    
    #time=[d.timestamp for d in date2]
    
    #return time
    
    
    #Get B
    mag1=df[:,10:13]
        
    
    
    #Get ephemeris data
    eph=df[:,7:10]
    

    
    ephx=df[:,7]
    ephy=df[:,8]
    ephz=df[:,9]
    
    return ephx,ephy,ephz


    
    
def rotate_into_mso(x,y,z,t):
    from trying3 import get_aberration_angle
    z_mso=z+479
    
    #Aberration:
        
    phi=get_aberration_angle(t)
    
    x_mso=x*np.cos(-phi)-y*np.sin(-phi)
    
    y_mso=y*np.sin(-phi)+y*np.cos(-phi)
    
    return x_mso,y_mso,z_mso
    
def rotate_into_msm(x,y,z,time):
    from trying3 import get_aberration_angle
    
    #Aberration:
        
    phi=get_aberration_angle(time)
    
    x_msm=x*np.cos(phi)-y*np.sin(phi)
    
    y_msm=y*np.sin(phi)+y*np.cos(phi)
    
    return x_msm,y_msm,z
    
    
def create_FIPS_plot(trange):
    
    fig,ax = plt.subplots(3,sharex=True)
    
    fd = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/PUB_ANN_Test/fd_prep_w_boundaries.pkl')
    
    date_times = [convert_to_datetime(t) for t in trange]
    
    fd = fd[((fd.time > date_times[0]) & (fd.time < date_times[1]))]
    
    fips_data = pd.read_pickle('/Users/bowersch/FIPS_Dictionary.pkl')
    
    ax[0].plot(fd.time,fd.magx,color='blue')
    ax[0].plot(fd.time,fd.magy,color='green')
    ax[0].plot(fd.time,fd.magz,color='red')
    
    ax[1].plot(fd.time,fd.magamp,color='black')
    
    ax[1].set_xlim(date_times[0],date_times[1])
    
    ax[0].set_ylim([-200,200])
    ax[1].set_ylim([0,400])
    
    def plot_others(time,H_data,erange,df_ntp):
        
        tF = time
        
        HF = H_data
        
        erange = erange
        
        gd_fips = np.where((tF > date_times[0]) & (tF < date_times[1]))[0]
        
        tF = tF[gd_fips]
        
        HF = HF[gd_fips,:]
        
        import matplotlib.colors as colors
        figi=ax[2].pcolormesh(tF,erange,np.transpose(HF),shading='nearest',\
                      norm=colors.LogNorm(),cmap='inferno')
        
            
        
        cb_ax = fig.add_axes([.93,.091,.02,.2])
        fig.colorbar(figi,orientation='vertical',cax=cb_ax,label='DEF (cm$^2$ s sr)$^{-1}$')
    
        ax[2].set_yscale('log')
        
        ax[2].set_ylim(.01,20)
        
        ax[2].set_ylabel('E/q [keV/q]')
        
        # df_ntp = df_ntp[((df_ntp.time > date_times[0]) & (df_ntp.time < date_times[1]))]
        
        # ax[2].scatter(df_ntp.time,df_ntp.n)
        # ax[2].set_yscale('log')
        
        # ax[3].scatter(df_ntp.time,df_ntp.t)
        
        # ax[3].set_yscale('log')
        
        #ax[4].scatter(df_ntp.time,df_ntp.p)
        
        
        
    plot_others(fips_data['time'],fips_data['H_data'],fips_data['erange'],fips_data['df_ntp'])
    
    return ax
    

    
def get_dates(df,date_max,time_max):

    
    if date_max==0:
        
        time_max = df.time.iloc[-1]
        
        diff = time_max-df.time

        dates =np.array([d.total_seconds() for d in diff])/60
        
        date_max = np.max(dates)
        
        
        
        
    else:
        
        diff = time_max-df.time

        dates =np.array([d.total_seconds() for d in diff])/60
        
        
    dates = date_max-dates  
    
    return dates
    
def analysis_of_event_0():
# =============================================================================
#     i=0
#     
#     icme=pd.read_excel(ICME_List_file)
#     
#     time_a = icme[['ICME_A_YEAR','ICME_A_DOY','ICME_A_HH','ICME_A_MM','ICME_A_SS']]\
#         .iloc[i]
#         
#     time_e = icme[['ICME_E_YEAR','ICME_E_DOY','ICME_E_HH','ICME_E_MM','ICME_E_SS']]\
#         .iloc[i]
#         
#     def create_datestring(year, day_of_year, hour, minute, second):
#         # Create a datetime object for January 1st of the given year
#         date = datetime.datetime(int(year), 1, 1)
#     
#         # Add the number of days (day_of_year - 1) to get to the desired date
#         date += datetime.timedelta(days=float(day_of_year) - 1, hours=float(hour), minutes=float(minute),seconds=float(second))
#         
#         def convert_datetime_to_string(time):
#             dstring=time.strftime("%Y-%m-%d %H:%M:%S")
#             return dstring
#         
#         return convert_datetime_to_string(date), date
#     
#     date_string_a, date_a = create_datestring(time_a[0], time_a[1], time_a[2], time_a[3], time_a[4])
#     
#     date_string_e, date_e = create_datestring(time_e[0], time_e[1], time_e[2], time_e[3], time_e[4])
#     
#     trange_ICME=[date_string_a,date_string_e]
#     from trying3 import load_MESSENGER_into_tplot
#     time,mag,magamp,eph=load_MESSENGER_into_tplot(trange_ICME[0][0:10])
#     time=np.array(time)
#     gd=np.where((time > date_a) & (time < date_e))[0]
#     
#     def create_KTH_prediction(trange):
#             
#         control_param_path = '/Users/bowersch/Desktop/Python_Code/KTH22-model-main/control_params_v8b.json'
#         fit_param_path = '/Users/bowersch/Desktop/Python_Code/KTH22-model-main/kth_own_cf_fit_parameters_opt_total_March23.dat'
#         date_string_mso=trange[0][0:10]
#         
#         ephx_mso,ephy_mso,ephz_mso=get_eph_mso(date_string_mso)
#         
#         x_mso=ephx_mso
#         
#         y_mso=ephy_mso
#         
#         z_mso=ephz_mso
#         
#         r_hel = np.ones(len(x_mso))*get_mercury_distance_to_sun(time[0])
#         
#         di = np.ones(len(x_mso))*100
#         
#         B_KTH =  kth.kth22_model_for_mercury_v8(x_mso, y_mso, z_mso, r_hel, di, 
#                                                 control_param_path, fit_param_path, 
#                                                 dipole=True, neutralsheet=True, 
#                                                 ringcurrent=True, internal=True, 
#                                                 external=True)
# 
#         Bx_KTH = B_KTH[0]
#         By_KTH = B_KTH[1]
#         Bz_KTH = B_KTH[2]
#         
#         
#         magamp_KTH=np.sqrt(Bx_KTH**2+Bz_KTH**2+By_KTH**2)
#         
#             
#         return Bx_KTH,By_KTH,Bz_KTH,magamp_KTH
#     
#     Bx_KTH,By_KTH,Bz_KTH,magamp_KTH = create_KTH_prediction(trange_ICME)
#     Bx_KTH=Bx_KTH[gd]
#     By_KTH=By_KTH[gd]
#     Bz_KTH=Bz_KTH[gd]
#     magamp_KTH=magamp_KTH[gd]
#     
#     data= {'time':time[gd], 'magx':mag[gd,0],'magy':mag[gd,1],'magz':mag[gd,2],
#            'magamp':magamp[gd],'ephx':eph[gd,0]/2440,
#            'ephy':eph[gd,1]/2440,'ephz':eph[gd,2]/2440,
#            'magx_KTH':Bx_KTH,'magy_KTH':By_KTH,
#            'magz_KTH':Bz_KTH,'magamp_KTH':magamp_KTH}
#     
#     df=pd.DataFrame(data)
#     df.to_pickle('ICME_Event_0.pkl')
# =============================================================================
    
    df=pd.read_pickle('ICME_Event_0.pkl')
    
    #cut = np.array([convert_to_datetime('2012-05-08 06:41:00'),convert_to_datetime('2012-05-08 07:53:30')])
    
    #df=df[((df.time > cut[0]) & (df.time < cut[1]))]
    
    Alfven_wing = np.array([convert_to_datetime('2012-05-08 07:20:00'),convert_to_datetime('2012-05-08 07:24:30')])
    
    df_aw=df[((df.time > Alfven_wing[0]) & (df.time < Alfven_wing[1]))]
    
    upstream = np.array([convert_to_datetime('2012-05-08 07:29:00'),convert_to_datetime('2012-05-08 07:57:30')])
    
    df_up=df[((df.time > upstream[0]) & (df.time < upstream[1]))]
    
    lobe = np.array([convert_to_datetime('2012-05-08 06:03:00'),convert_to_datetime('2012-05-08 06:31:30')])
    
    df_lobe=df[((df.time > lobe[0]) & (df.time < lobe[1]))]
    
    cusp = np.array([convert_to_datetime('2012-05-08 07:10:00'),convert_to_datetime('2012-05-08 07:13:30')])
    
    df_cusp=df[((df.time > cusp[0]) & (df.time < cusp[1]))]
    
    fluctuations = np.array([convert_to_datetime('2012-05-08 05:05:00'),convert_to_datetime('2012-05-08 06:01:30')])
    
    df_fluctuations = df[((df.time > fluctuations[0]) & (df.time < fluctuations[1]))]
    
    fig, (ax1,ax2,ax3)=plt.subplots(1,3)
    #Plot Mercury
    
    theta = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(theta)
    y = np.sin(theta)-0.2
    
    # Plot the circle in all 3 plots
    ax1.plot(x, y, color='gray')
    ax2.plot(x,y+.2,color='gray')
    ax3.plot(x,y,color='gray')
    # Color the left hemisphere red and the right hemisphere gray
    ax1.fill_between(x, y, where=x<0, color='black', interpolate=True)
    ax2.fill_between(x, y+.2, where=x<0,color='black',interpolate=True)
    
    #Set equal aspect so Mercury is circular
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')
    ax1.set_xlim([-3, 1.6])
    ax1.set_ylim([-3, 1.2])
    
    ax2.set_xlim([-3,1.6])
    ax2.set_ylim([-2.5,2.5])
    
    ax3.set_xlim(-2.5,2.5)
    ax3.set_ylim(-3,1.2)
    
    ax1.set_xlabel("$X_{MSM'}$",fontsize=15)
    
    ax1.set_ylabel("$Z_{MSM'}$",fontsize=15)
    
    ax2.set_xlabel("$X_{MSM'}$",fontsize=15)
    ax2.set_ylabel("$Y_{MSM'}$",fontsize=15)
    
    ax3.set_ylabel("$Z_{MSM'}$",fontsize=15)
    
    ax3.set_xlabel("$Y_{MSM'}$",fontsize=15)
    
    
    plot_mp_and_bs(ax1)
    plot_mp_and_bs(ax2)
    
    def add_eph_onto_traj(ax1,ax2,ax3,ephx,ephy,ephz,col,lw):
    
        ax1.plot(ephx,ephz,color=col,linewidth=lw)
        
        ax2.plot(ephx,ephy,color=col,linewidth=lw)
        
        ax3.plot(ephy,ephz,color=col,linewidth=lw)
        
    add_eph_onto_traj(ax1,ax2,ax3,df.ephx,df.ephy,df.ephz,'black',1.5)
    
    add_eph_onto_traj(ax1,ax2,ax3,df_aw.ephx,df_aw.ephy,df_aw.ephz,'green',3.0)
    
    add_eph_onto_traj(ax1,ax2,ax3,df_up.ephx,df_up.ephy,df_up.ephz,'red',3.0)
    
    add_eph_onto_traj(ax1,ax2,ax3,df_lobe.ephx,df_lobe.ephy,df_lobe.ephz,'blue',3.0)
    
    add_eph_onto_traj(ax1,ax2,ax3,df_cusp.ephx,df_cusp.ephy,df_cusp.ephz,'orange',3.0)
    
    add_eph_onto_traj(ax1,ax2,ax3,df_fluctuations.ephx,df_fluctuations.ephy,df_fluctuations.ephz,'purple',3.0)
        
    
    fig, (ax1,ax2,ax3,ax4)=plt.subplots(4,sharex=True)
    
    
    
    dates=df.time

    # Create a figure and axis objec
    
    # Plot the data
    
    # Set the x-axis label and format
    
    ax1.axhline(y=0.0, color='black', linestyle='--',linewidth=0.5)
    ax2.axhline(y=0.0, color='black', linestyle='--',linewidth=0.5)
    ax3.axhline(y=0.0, color='black', linestyle='--',linewidth=0.5)
    
    
    
    ax1.plot(dates,df.magx,'black',linewidth=.75,label='MESSENGER')

    ax1.plot(dates,df.magx_KTH,'red',linewidth=.75,label='KTH_Model')
    
    
    ax2.plot(dates,df.magy,'black',linewidth=.75,label='MESSENGER')
    ax2.plot(dates,df.magy_KTH,'red',linewidth=.75,label='KTH_Model')
    
    
    ax3.plot(dates,df.magz,'black',linewidth=.75,label='MESSENGER')
    ax3.plot(dates,df.magz_KTH,'red',linewidth=.75,label='KTH_Model')
    
    
    ax4.plot(dates,df.magamp,'black',linewidth=.75,label='MESSENGER')
    ax4.plot(dates,df.magamp_KTH,'red',linewidth=.75,label='KTH_Model')


    axes=[ax1,ax2,ax3,ax4]
    
    ylims=[[-300,300],[-300,300],[-300,300],[0,500]]
    
    ax1.set_ylabel('$BX_{MSM}$ (nT)')
    ax2.set_ylabel('$BY_{MSM}$ (nT)')
    ax3.set_ylabel('$BZ_{MSM}$ (nT)')
    ax4.set_ylabel('$|B|$ (nT)')
    
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.set_xlabel('2012-05-08')
    
    for p in range(len(axes)):
        x=axes[p]
        shade_in_time_series(x,Alfven_wing[0],Alfven_wing[1],'green',shade=True,alpha=.15)
        
        shade_in_time_series(x,lobe[0],lobe[1],'blue',shade=True,alpha=.15)
        
        shade_in_time_series(x,cusp[0],cusp[1],'orange',shade=True,alpha=.15)
        
        shade_in_time_series(x,upstream[0],upstream[1],'red',shade=True,alpha=.15)
        
        shade_in_time_series(x,fluctuations[0],fluctuations[1], 'purple',shade=True,alpha=.15)
        
        x.set_ylim(ylims[p])
        
        x.legend(loc='upper left')
        
        
        
def create_orbit_dataframes_0(num_orbits,ii):
    '''Create the dataframes used to make multi-orbit-plots, ii is the ICME event number (21 for AW)
    
    Saves orbit by orbit dataframes that can be recalled by specific_multi_orbit plot
    '''
    
    
    
#def create_orbit_dataframes_0(num_orbits,date_string_a,date_string_e):
    
    i=ii

    icme=pd.read_excel(ICME_List_file)
    
    time_a = icme[['ICME_A_YEAR','ICME_A_DOY','ICME_A_HH','ICME_A_MM','ICME_A_SS']]\
        .iloc[i]
        
    time_e = icme[['ICME_E_YEAR','ICME_E_DOY','ICME_E_HH','ICME_E_MM','ICME_E_SS']]\
        .iloc[i]
        
        
    def create_datestring(year, day_of_year, hour, minute, second):
        # Create a datetime object for January 1st of the given year
        date = datetime.datetime(int(year), 1, 1)
    
        # Add the number of days (day_of_year - 1) to get to the desired date
        date += datetime.timedelta(days=float(day_of_year) - 1, hours=float(hour), minutes=float(minute),seconds=float(second))
        
        def convert_datetime_to_string(time):
            dstring=time.strftime("%Y-%m-%d %H:%M:%S")
            return dstring
        
        return convert_datetime_to_string(date), date
    
    date_string_a, date_a = create_datestring(time_a[0], time_a[1], time_a[2], time_a[3], time_a[4])
    
    date_string_e, date_e = create_datestring(time_e[0], time_e[1], time_e[2], time_e[3], time_e[4])
    
    
    date_before = date_a-datetime.timedelta(hours=8*num_orbits)
    
    date_after = date_e+datetime.timedelta(hours=8*num_orbits)
    
    fd=pd.read_pickle('/Users/bowersch/Desktop/Python_Code/PUB_ANN_Test/fd_prep_w_boundaries.pkl')
    
    fips_data = pd.read_pickle('/Users/bowersch/FIPS_Dictionary.pkl')
    
    df_ntp = fips_data['df_ntp']
    
    df_k = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/James_Kappa_Data.pkl')
    
    HF = fips_data['H_data']
    
    tF = fips_data['time']
    
    df_all = fd[((fd.time > date_before) & (fd.time < date_after))]
    
    r_all = np.sqrt(df_all.ephx.to_numpy()**2 + df_all.ephy.to_numpy()**2 + \
                    df_all.ephz.to_numpy()**2)
    
    from scipy.signal import find_peaks
    
    peaks=find_peaks(r_all,distance=60*60*6)
    
    for p in range(len(peaks[0])-1):
        
        orbit_range = [peaks[0][p],peaks[0][p+1]]
        
        time_range = [df_all.time.iloc[orbit_range[0]],df_all.time.iloc[orbit_range[1]]]
        
        df_ntp_o = df_ntp[((df_ntp.time > time_range[0]) & (df_ntp.time < time_range[1]))]
        
        df_k_o = df_k[((df_k.time > time_range[0]) & (df_k.time < time_range[1]))]
        
        HF_o = HF[((tF > time_range[0]) & (tF < time_range[1]))]
        
        summed_HF = np.sum(HF_o,axis = 1)
        
        
        tF_s = tF[((tF > time_range[0]) & (tF < time_range[1]))]
        
        tF_s = tF_s[(summed_HF > 0)]
        
        new_resolution = 10
        
        new_length = len(tF_s) // new_resolution
        # Reshape the arrays to prepare for averaging
        t_reshaped = tF_s[:new_length * new_resolution].reshape(new_length, new_resolution)
        
        def tstamp(x):
            return x.timestamp()
        t_utc=np.vectorize(tstamp)(t_reshaped)
        
        # Calculate the average values
        t_averaged_utc = np.mean(t_utc, axis=1)
        
        t_averaged=np.array([convert_to_datetime(convert_to_date_2_utc(t)) for t in t_averaged_utc])
        
        
        
        summed_HF = summed_HF[(summed_HF > 0)]
        
        
        summed_HF = downsample(new_resolution,summed_HF)

        qual = np.zeros(len(summed_HF))
        
        df_HF = pd.DataFrame({'time':t_averaged,
                                 'n':summed_HF,
                                 'qual':qual})
        

        df_orbit = df_all[((df_all.time > time_range[0]) & (df_all.time < time_range[1]))]
        
        df_orbit.to_pickle('ICME_Event_'+str(i)+'_'+str(p)+'.pkl')
        
        df_HF.to_pickle('ICME_Event_HF_'+str(i)+'_'+str(p)+'.pkl')
        
        df_ntp_o.to_pickle('ICME_Event_NOBS_'+str(i)+'_'+str(p)+'.pkl')
        
        df_k_o.to_pickle('ICME_Event_K_'+str(i)+'_'+str(p)+'.pkl')
        
        print('Done '+str(p))
        
def get_KTH_pred_from_MSM(ephx,ephy,ephz,time,d):
    
    control_param_path = '/Users/bowersch/Desktop/Python_Code/KTH22-model-main/control_params_v8b.json'
    fit_param_path = '/Users/bowersch/Desktop/Python_Code/KTH22-model-main/kth_own_cf_fit_parameters_opt_total_March23.dat'
    
    ephx=ephx*2440
    
    ephy=ephy*2440
    
    ephz=ephz*2440
    
    x_mso,y_mso,z_mso=rotate_into_mso(ephx,ephy,ephz,time)

    r_hel = np.ones(len(x_mso))*get_mercury_distance_to_sun(time)
    
    di = np.ones(len(x_mso))*d
    
    B_KTH =  kth.kth22_model_for_mercury_v8(x_mso, y_mso, z_mso, r_hel, di, 
                                            control_param_path, fit_param_path, 
                                            dipole=True, neutralsheet=True, 
                                            ringcurrent=True, internal=True, 
                                            external=True)

    Bx_KTH = B_KTH[0]
    By_KTH = B_KTH[1]
    Bz_KTH = B_KTH[2]
    
    
    magamp_KTH=np.sqrt(Bx_KTH**2+Bz_KTH**2+By_KTH**2)
    
    
    Bx_KTH,By_KTH,Bz_KTH = rotate_into_msm(Bx_KTH,By_KTH,Bz_KTH,time)
    
    
    return Bx_KTH,By_KTH,Bz_KTH,magamp_KTH


        

            
    
def average_parameters(ii,K):
    
    def create_nobs_array(dates_nobs,df_nobs):
        
        dates_nobs_line = np.linspace(0,690,num=690//2)
        
        n_values = np.zeros(len(dates_nobs_line))+np.nan
        
        for i in range(len(dates_nobs)-1):
            
        
            xi = np.searchsorted(dates_nobs_line,dates_nobs[i])-1
            
            
            if np.abs(dates_nobs[i]-dates_nobs_line[xi]) < 2:
                   
                n_values[xi] = df_nobs.n.iloc[i]

                
            else:
                
                n_values[xi] = np.nan
                
        non_nan = np.where(np.isnan(n_values)==False)[0]
        
        for i in np.arange(non_nan[0],non_nan[-1]):
            if np.isnan(n_values[i])==True:
                n_values[i] = 0
                
            
        return n_values
            
        
        
        

    for i in range(11):
        if i != 6:
            
            df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_'+str(ii)+'_'+str(i)+'.pkl')
            df_nobs = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_NOBS_'+str(ii)+'_'+str(i)+'.pkl')
            
            if K==True:
                df_nobs = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_K_'+str(ii)+'_'+str(i)+'.pkl')
            
            dates = get_dates(df,0,0)
            
            dates_nobs = get_dates(df_nobs,np.max(dates),df.time.iloc[-1])
            
            
            
        
            
            if i == 0:
                
                gd = (dates<690)
                
                gd_nobs = (dates_nobs<690)
                
                df = df[gd]
                
                df_nobs = df_nobs[gd_nobs]
                
                magamp_averaged = np.zeros((0,len(df)))
                
                bx_averaged = np.zeros((0,len(df)))
                by_averaged = np.zeros((0,len(df)))
                bz_averaged = np.zeros((0,len(df)))
                
                dates_nobs_line = np.linspace(0,690,num=690//2)
                
                n_averaged = np.zeros((0,len(dates_nobs_line)))
            
            
            m = np.max(np.where(gd==True)[0])

            n_array = create_nobs_array(dates_nobs,df_nobs)
            
            m_nobs = np.max(np.where(gd_nobs==True)[0])
            
            df = df.iloc[0:m+1]
            
            df_nobs = df_nobs.iloc[0:m_nobs+1] 
            
            dates = dates[0:m+1]
            
            dates_nobs = dates_nobs[0:m_nobs+1]
            
            n_array = create_nobs_array(dates_nobs,df_nobs)
            
            n_averaged = np.append(n_averaged,[n_array],axis=0)

            magamp_averaged = np.append(magamp_averaged,[df.magamp.to_numpy()],axis=0)
            
            bx_averaged = np.append(bx_averaged,[df.magx.to_numpy()],axis=0)
            by_averaged = np.append(by_averaged,[df.magy.to_numpy()],axis=0)
            bz_averaged = np.append(bz_averaged,[df.magz.to_numpy()],axis=0)
            
        
        magamp_total = np.mean(magamp_averaged,axis=0)
        bx_total = np.mean(bx_averaged,axis=0)
        by_total = np.mean(by_averaged,axis=0)
        bz_total = np.mean(bz_averaged,axis=0)

        mag_total = np.transpose(np.vstack((bx_total,by_total,bz_total,magamp_total)))
        
        
    return dates, mag_total, dates_nobs_line, np.nanmean(n_averaged,axis=0)
        
        
        
        

        

def add_onto_density_panel(ax,dates,df,col,siz,alph,labe):
    
    
    ax.scatter(dates,df,c=col,s=siz,alpha=alph, label = labe, zorder=3)
    
    diff=np.roll(dates,1)-dates
    
    #Find where model transitions from one region to another
    try:
        transitions=np.where(np.abs(diff) > 15)[0]
        
    except:
        #diff=diff.to_numpy()
        d_diff = np.array([d.total_seconds() for d in diff])
        
        transitions=np.where(np.abs(d_diff) > 15*60)[0]
        
        
    
    if transitions[0]!=0:
        
        transitions=np.insert(transitions,0,0)
        
    if transitions[-1] != len(dates):
        transitions = np.insert(transitions,len(transitions),len(dates))
    
    #transitions=np.insert(transitions,len(transitions),len(diff)-1)
    
    for j in range(len(transitions)-1):
        
        si=transitions[j]
        
        fi=transitions[j+1]
        
        ax.plot(dates[si:fi],df.iloc[si:fi],color=col,alpha=alph)   


def calculate_beta():
    
    for i in range(11):
        ii=21
        
        df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_'+str(ii)+'_'+str(i)+'.pkl')
        
        df_nobs = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_NOBS_'+str(ii)+'_'+str(i)+'.pkl')
    
        magx = df.magx
        magy = df.magy
        magz = df.magz
        magamp = df.magamp
        
        mags = [magx,magy,magz,magamp]
        
        time_mag = df.time
        
        n = df_nobs.n
        
        time_n = df_nobs.time
        
        
        magx_int = np.interp(time_n,time_mag,magx)
        magy_int = np.interp(time_n,time_mag,magy)
        magz_int = np.interp(time_n,time_mag,magz)
        
        
        
        magamp_int = np.interp(time_n,time_mag,magamp)
        dates = get_dates(df,0,0)
        
        mp = 1.67E-27
        
        munaught=4*np.pi*1E-7
        
        n = n*100**3
        
        rho = n*mp
        
        def calculate_va(mag,rho):
            
            return mag*1E-9/np.sqrt(munaught*rho)/1000
        
        va_x = calculate_va(magx_int,rho)
        va_y = calculate_va(magy_int,rho)
        va_z = calculate_va(magz_int,rho)
        
        va = calculate_va(magamp_int,rho)
        
        def estimate_MA_in_sheath():
            
            df_boundaries = pd.read_pickle('df_boundaries_philpott.pkl')
            
            mp_in_1 = df_boundaries[df_boundaries.Cross_Type=='mp_in_1'].time
            mp_in_2 = df_boundaries[df_boundaries.Cross_Type=='mp_in_2'].time
            
            mp_out_1 = df_boundaries[df_boundaries.Cross_Type=='mp_out_1'].time
            mp_out_2 = df_boundaries[df_boundaries.Cross_Type=='mp_out_2'].time
            
            mp_1 = np.sort(np.append(mp_in_1,mp_out_1))
            
            mp_2 = np.sort(np.append(mp_in_2,mp_out_2))
            
            mp_time = np.stack((mp_1,mp_2),axis=1)
            
            mp_time = mp_time.astype('datetime64[s]').astype('datetime64[us]').astype(object)
            
            bs_in_1 = df_boundaries[df_boundaries.Cross_Type=='bs_in_1'].time
            bs_in_2 = df_boundaries[df_boundaries.Cross_Type=='bs_in_2'].time
            
            bs_out_1 = df_boundaries[df_boundaries.Cross_Type=='bs_out_1'].time
            bs_out_2 = df_boundaries[df_boundaries.Cross_Type=='bs_out_2'].time

            
            bs_1 = np.sort(np.append(bs_in_1,bs_out_1))
            
            bs_2 = np.sort(np.append(bs_in_2,bs_out_2))
            
            bs_time = np.stack((bs_1,bs_2),axis=1)
            
            bs_time = bs_time.astype('datetime64[s]').astype('datetime64[us]').astype(object)
            
        
            time = df.time
            
            time_nobs = df_nobs.time
            
            gd_mp_nobs = np.where((mp_time[:,0] > time_nobs.iloc[0]) & (mp_time[:,1] < time_nobs.iloc[-1]))[0]
            
            gd_mp = np.where((mp_time[:,0] > time.iloc[0]) & (mp_time[:,1] < time.iloc[-1]))[0]
            
            t_mp = mp_time[gd_mp,:]
            
    
            gd_bs = np.where((bs_time[:,0] > time.iloc[0]) & (bs_time[:,1] < time.iloc[-1]))[0]
            
            #gd_bs_nobs = np.where((bs_time[:,0] > time_nobs.iloc[0]) & (bs_time[:,1] < time_nobs.iloc[-1]))[0]
            
            t_bs = bs_time[gd_bs,:]
            
            
            x_mp = np.zeros((0,2))
            
            r_mp = np.zeros((0,2))
            
            x_bs = np.zeros((0,2))
            
            r_bs = np.zeros((0,2))
            
            for l in range(len(t_mp)):
                
                df_0 = df[(df.time > t_mp[l,0]) & (df.time < t_mp[l,0]+datetime.timedelta(seconds=1.5))]
            
                df_1 = df[(df.time > t_mp[l,1]) & (df.time < t_mp[l,1]+datetime.timedelta(seconds=1.5))]
            
                x_mp0 = np.mean(df_0.ephx)
                
                r_mp0 = np.mean(np.sqrt(df_0.ephy**2+df_0.ephz**2))
                
                x_mp1 = np.mean(df_1.ephx)
                
                r_mp1 = np.mean(np.sqrt(df_1.ephy**2+df_1.ephz**2))
                
                x_mp = np.append(x_mp,[[x_mp0,x_mp1]],axis=0)
                
                r_mp = np.append(r_mp, [[r_mp0,r_mp1]],axis=0)
                
            for l in range(len(t_bs)):
            
                df_0 = df[(df.time > t_bs[l,0]) & (df.time < t_bs[l,0]+datetime.timedelta(seconds=1.5))]
            
                df_1 = df[(df.time > t_bs[l,1]) & (df.time < t_bs[l,1]+datetime.timedelta(seconds=1.5))]
            
                x_bs0 = np.mean(df_0.ephx)
                
                r_bs0 = np.mean(np.sqrt(df_0.ephy**2+df_0.ephz**2))
                
                x_bs1 = np.mean(df_1.ephx)
                
                r_bs1 = np.mean(np.sqrt(df_1.ephy**2+df_1.ephz**2))
                
                x_bs = np.append(x_bs,[[x_bs0,x_bs1]],axis=0)
                
                r_bs = np.append(r_bs, [[r_bs0,r_bs1]],axis=0)
                
    
            #eph_mp = mp['eph']
            
    
            
            r_mp_range = np.abs((r_mp[:,1]-r_mp[:,0])/2)
            
            r_mp = np.mean(r_mp,axis=1)
            
            x_mp_range = np.abs((x_mp[:,1]-x_mp[:,0])/2)
            
            x_mp = np.mean(x_mp,axis=1)
            
            MA = np.array([])
            
            MA_std = np.array([])
            
            for p in range(2):
                
                mp_crossing = mp_time[gd_mp[p],:]
                
                bs_crossing_nobs = bs_time[gd_bs[0],:]
                
                if len(gd_bs) == 0:
                    
                    bs_crossing = mp_crossing
                
                
                if len(gd_bs) == 1:
                    
                    bs_crossing = bs_time[gd_bs[0],:]
                    
                if len(gd_bs) > 1:
                
                    bs_crossing = bs_time[gd_bs[p],:]
                
             
                
                if p == 0:
                    up_range_nobs = np.array([bs_crossing_nobs[0], bs_crossing_nobs[0] + datetime.timedelta(minutes=10)])
                    
                    
                    up_range = np.array([mp_crossing[0] - datetime.timedelta(minutes=10),mp_crossing[0]])
                    
                    mp_range = np.array([mp_crossing[1], mp_crossing[1] + datetime.timedelta(minutes=1)])
                    
                if p == 1:
                    
                    up_range_nobs = np.array([bs_crossing_nobs[1] - datetime.timedelta(minutes=10),bs_crossing_nobs[1]])
                    
                    up_range = np.array([mp_crossing[1], mp_crossing[1] + datetime.timedelta(minutes=10)])
                    
                    mp_range = np.array([mp_crossing[0] - datetime.timedelta(minutes=1), mp_crossing[0]])
                    

                magamp=df.magamp.to_numpy()
                
                
                mag=df[['magx','magy','magz']].to_numpy()
                
                eph=df[['ephx','ephy','ephz']].to_numpy()
                
                
                Rss=1.45
                alpha=0.5
    
                phi2 = (np.linspace(0,2*np.pi,1000))
    
                rho=Rss*(2/(1+np.cos(phi2)))**(alpha)
    
                xmp=rho*np.cos(phi2)
    
                ymp=rho*np.sin(phi2)
    
                curve=np.transpose(np.vstack((xmp,ymp)))
                
                ea_test = np.mean(eph[(df.time > mp_range[0]) & (df.time < mp_range[1]),:],axis=0)
                
                ea_std = np.std(eph[(df.time > mp_range[0]) & (df.time < mp_range[1]),:],axis=0)
                
                df_MP=df[(df.time > mp_range[0]) & (df.time < mp_range[1])]
                
                percentile = np.percentile(df_MP.magamp.to_numpy(), 80)
                
                df_MP=df_MP[(df.magamp > percentile)]
                
                
                B_MP=np.mean(df_MP.magamp.to_numpy())
                
                B_MP_std = np.std(df_MP.magamp.to_numpy())
                
                B_SW = np.mean(magamp[(df.time > up_range_nobs[0]) & (df.time < up_range_nobs[1])])
                
                B_SW_std = np.std(magamp[(df.time > up_range_nobs[0]) & (df.time < up_range_nobs[1])])
                
                munaught=4*np.pi*1E-7
                
                ra_test=np.sqrt(ea_test[1]**2+ea_test[2]**2)
                
                ra_std = 0.5*(2*ea_std[1]+2*ea_std[2])
                
                # Find distance to inner point in curve (mp uncertainty)
                
                da,point=distance_point_to_curve([ea_test[0]-ea_std[0],ra_test-ra_std],curve,get_point=True)
                
                diff=curve-[point[0],point[1]]
                
                
                diff=np.sqrt(diff[:,0]**2+diff[:,1]**2)
                
                index=np.where(np.abs(diff)==np.min(np.abs(diff)))[0][0]
                
                n=calculate_normal_vector(curve,index)
                
                phi_1=np.arccos(np.dot(n,np.array([1,0])))
                
                #Find distance to outer point in curve (mp uncertainty)
                
                p_T = np.mean(df_nobs.p[((df_nobs.time > up_range_nobs[0]) & (df_nobs.time < up_range_nobs[1]))])
                
                
                p_T = p_T*1E-9
                
                da,point=distance_point_to_curve([ea_test[0]+ea_std[0],ra_test+ra_std],curve,get_point=True)
                
                diff=curve-[point[0],point[1]]

                diff=np.sqrt(diff[:,0]**2+diff[:,1]**2)
                
                index=np.where(np.abs(diff)==np.min(np.abs(diff)))[0][0]
                
                n=calculate_normal_vector(curve,index)
                
                phi_2=np.arccos(np.dot(n,np.array([1,0])))
                
                phi = np.mean([phi_1,phi_2])
                
                phi_std = np.abs(phi_2-phi_1)
                
                
                B_MP=B_MP*1E-9
                
                pmag=B_MP**2/(2*munaught)
                
                pmag_std=pmag*B_MP_std*1E-9/B_MP*2
                

                
                B_SW=B_SW*1E-9
                
                B_MP_std=B_MP_std*1E-9
            
                B_SW_std = B_SW_std*1E-9
                
                rel_uncertainty_mp = B_MP_std/B_MP
                
                rel_uncertainty_sw = B_SW_std/B_SW
                
                
                
                #breakpoint()
                psw1=(B_MP**2/(2*munaught) - (B_SW**2/(2*munaught)+p_T))*(.88*np.cos(phi)**2)**(-1)
               
                #psw_max =  ((B_MP+B_MP_std)**2/(2*munaught) - (B_SW-B_SW_std)**2/(2*munaught))*(.88*np.cos(np.max([phi_1,phi_2]))**2)**(-1)
               
                #psw_min =  ((B_MP-B_MP_std)**2/(2*munaught) - (B_SW+B_SW_std)**2/(2*munaught))*(.88*np.cos(np.min([phi_1,phi_2]))**2)**(-1) 
               
                
               
                #psw_std = psw_max-psw1
               
                if ((phi >= 120*np.pi/180) | (phi <= np.radians(60))):
                
                    MA = np.append(MA,np.sqrt(psw1*munaught)/B_SW)
                    
                    #MA_min = np.sqrt(psw_min*munaught)/(B_SW+B_SW_std)
                    
                    #MA_max = np.sqrt(psw_max*munaught)/(B_SW-B_SW_std)
                    
                    #MA_std = np.append(MA_std,MA_max-MA)
                    
            
            return MA
        
        Ma = estimate_MA_in_sheath()
        
        v_estimate = va*Ma
        
        beta = df_nobs.p*1E-9/((magamp_int*1E-9)**2/(2*munaught))

        df_nobs[['va_x','va_y','va_z','va','beta','v_estimate']] = np.transpose(np.vstack((va_x,va_y,va_z,va,beta,v_estimate)))
        
        df_nobs.to_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_NOBS_'+str(ii)+'_'+str(i)+'.pkl')
        
        

        
        # fig,ax = plt.subplots(5,sharex=True)
        
        # ax[0].plot(dates,magamp,color='black')
        # add_onto_density_panel(ax[1],get_dates(df_nobs,np.max(dates),np.max(time_mag)),n,'royalblue',3,1,'ICME Orbit')
        # add_onto_density_panel(ax[2],get_dates(df_nobs,np.max(dates),np.max(time_mag)),va_x,'blue',3,1,'VA_X')
        # add_onto_density_panel(ax[2],get_dates(df_nobs,np.max(dates),np.max(time_mag)),va_y,'green',3,1,'VA_Y')
        # add_onto_density_panel(ax[2],get_dates(df_nobs,np.max(dates),np.max(time_mag)),va_z,'red',3,1,'VA_Z')
        
        # ax[0].set_xlim([200,600])
        
        # v_estimate = 1.0*va
        
        # add_onto_density_panel(ax[3],get_dates(df_nobs,np.max(dates),np.max(time_mag)),v_estimate,'black',3,1,'VA_Z')
    
        
        
        # add_onto_density_panel(ax[4],get_dates(df_nobs,np.max(dates),np.max(time_mag)),beta,'black',3,1,'VA_Z')


def estimate_velocity_diff():
    
    ii = 21
    
    for i in [6]:
        
        df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_'+str(ii)+'_'+str(i)+'.pkl')
        df_nobs = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_NOBS_'+str(ii)+'_'+str(i)+'.pkl')
        df_HF = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_HF_'+str(ii)+'_'+str(i)+'.pkl')
        
        sw_range = [310,320]
        
        aw_range = [380,405]
        
        diff = df.time.iloc[-1]-df.time
        
        
        dates =np.array([d.total_seconds() for d in diff])/60
        
        dates = np.max(dates)-dates
        
        diff_nobs = df.time.iloc[-1]-df_nobs.time
        
        dates_nobs = np.array([d.total_seconds() for d in diff_nobs])/60

        dates_nobs = np.max(dates)-dates_nobs
        
        
        df_nobs_sw = df_nobs[((dates_nobs > sw_range[0]) & (dates_nobs < sw_range[1]))]
        
        df_nobs_aw = df_nobs[((dates_nobs > aw_range[0]) & (dates_nobs < aw_range[1]))]
        
        # v_sw+va_sw = v_aw+va_aw
        
        # v_sw - va_sw = v_aw - va_aw
        
        # v_sw-v_aw = va_aw-va_sw
        
        v_sw = np.array([-1000,0,0])
        
        #v_sw/va_sw = 1.5
        
        
        
        va_sw = -1.5*v_sw[0]
        munaught = 4*np.pi * 1E-7
        
        mp = 1.67E-27
        
        sw_density = (66E-9**2)/((va_sw*1000)**2*munaught*mp)
        
        #va_sw = B/sqrt(munuaght*rho)
        
        # rho*munaught*va_sw**2 = B**2
        
        # n = B**2/(munaught*va_sw**2*mp)
        
        print(sw_density/(100**3))
        
        #va_sw = [np.mean(df_nobs_sw.va_x),np.mean(df_nobs_sw.va_y),np.mean(df_nobs_sw.va_z)]/np.mean(df_nobs_sw.va)
        
        #va_aw = [np.mean(df_nobs_aw.va_x),np.mean(df_nobs_aw.va_y),np.mean(df_nobs_aw.va_z)]/np.mean(df_nobs_aw.va)
        
        c1 = v_sw + np.array([np.mean(df_nobs_sw.va_x),np.mean(df_nobs_sw.va_y),np.mean(df_nobs_sw.va_z)])
        
        c2 = v_sw - np.array([np.mean(df_nobs_sw.va_x),np.mean(df_nobs_sw.va_y),np.mean(df_nobs_sw.va_z)])

        c_norm1 = c1/np.linalg.norm(c1)
        
        
        c_norm2 = c2/np.linalg.norm(c2)
        
        print(c_norm1)
        
        print(c_norm2)
        
        #print(va_aw)
        
        #print(va_aw-c_norm)
        
        


        

    
def create_and_plot_ICME_Event(event):
    create_event_dfs(event)
    create_orbit_dataframes_0(2,event)
    multi_orbit_plot(event)
    #quiver_plot(event)
    
    #oi= 0 , 5, 20 (around 450 mins a possible exiting of an Alfven wing)
    
    
    # 21 (270 minutes see a large -BZ component where we'd expect an Alfven wing)
    
    
def make_mercury():
    fs=18
    fig, (ax1,ax2,ax3)=plt.subplots(1,3)
    #Plot Mercury
    
    theta = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(theta)
    y = np.sin(theta)-0.2
    
    # Plot the circle in all 3 plots
    ax1.plot(x, y, color='gray')
    ax2.plot(x,y+.2,color='gray')
    ax3.plot(x,y,color='gray')
    # Color the left hemisphere red and the right hemisphere gray
    ax1.fill_between(x, y, where=x<0, color='black', interpolate=True)
    ax2.fill_between(x, y+.2, where=x<0,color='black',interpolate=True)
    
    #Set equal aspect so Mercury is circular
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-6, 2])
    
    ax2.set_xlim([-3,3])
    ax2.set_ylim([-2.5,2.5])
    
    ax3.set_xlim(-2.5,2.5)
    ax3.set_ylim(-6,2)
    
    ax1.set_xlabel("$X_{MSM'}$",fontsize=fs)
    
    ax1.set_ylabel("$Z_{MSM'}$",fontsize=fs)
    
    ax2.set_xlabel("$X_{MSM'}$",fontsize=fs)
    ax2.set_ylabel("$Y_{MSM'}$",fontsize=fs)
    
    ax3.set_ylabel("$Z_{MSM'}$",fontsize=fs)
    
    ax3.set_xlabel("$Y_{MSM'}$",fontsize=fs)
    
    
    plot_mp_and_bs(ax1)
    plot_mp_and_bs(ax2)
    
    return ax1,ax2,ax3
def downsample(new_resolution,arr):
    
    new_length = len(arr) // new_resolution
    
    arr_reshaped = arr[:new_length * new_resolution].reshape(new_length, new_resolution)
    
    arr_averaged= np.mean(arr_reshaped, axis=1)
    
    return arr_averaged    
def calculate_normal_vector(curve_points, index):
    # Get the neighboring points
    prev_index = (index - 1) % len(curve_points)
    next_index = (index + 1) % len(curve_points)
    prev_point = curve_points[prev_index]
    next_point = curve_points[next_index]
    
    # Calculate the tangent vector
    tangent_vector = next_point - prev_point
    
    # Normalize the tangent vector
    tangent_vector = tangent_vector/ np.linalg.norm(tangent_vector)
    
    # Calculate the normal vector
    normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
    
    return normal_vector    

        
        
def make_alfven_wing(Ma,ax,factor,B_IMF):
        
    
    other = (180-np.rad2deg(np.arctan2(B_IMF[0],B_IMF[2])))
    

    phi = np.arctan(Ma)
    
    theta = np.radians(other)
    
    #tilt_angle = (phi-theta)*-1*factor
    
    if factor == 1:
        
        tilt_angle = np.pi/2 - (phi-theta)
        
    else:
        tilt_angle = np.pi-(phi-theta)
    
    
    slope = np.tan(tilt_angle)
    
    
    
    def make_line(x_point,z_point):
    
          
        x_vals = np.array([x_point - 10, x_point])
       
        # Calculate corresponding y values using the point-slope form: y - y1 = m(x - x1)
        z_vals = z_point + slope * (x_vals - x_point)
        
        ax.plot(x_vals,z_vals,color='gray',linewidth=2)
        
        dx = x_vals[-1]-x_vals[0]
        
        dz = z_vals[-1]-z_vals[0]
        
        d_mag = np.sqrt(dx**2+dz**2)
        
        unit_vector = [dx/d_mag,0,dz/d_mag]
        

    
    day_p = [1.45,.19]
    
    night_p = [-4.6,-.17]
    
    # angle = np.arctan(19/60)
    
    # def rotate_point(p,a):
        
    #     return [p[0]*np.cos(a)-p[1]*np.sin(a),p[0]*np.sin(a)+p[1]*np.cos(a)]
    
    # day_p = rotate_point(day_point,angle)
    # night_p = rotate_point(night_point,angle)
   
    
    
    make_line(day_p[0],day_p[1])
    make_line(night_p[0],night_p[1])
    
    

    
    
def quiver_plot(event,diff=False,full=False,model=False):
    
    #filename = 'ICME_21_Diff.pkl'
    
   # filename = 'ICME_Event_'+str(21)+'_'+str(6)+'.pkl'
    
    #filename = 'ICME_21.pkl'
    
    #filename = 'ICME_Event_'+str(21)+'_'+str(6)+'.pkl'
    
    filename = 'ICME_Dictionary_'+str(21)+'_'+str(6)+'.pkl'
    
    if filename[0:6] == 'ICME_D':
    
        icme_21 = pd.read_pickle(filename)
    
        df = icme_21['df']
        
    else: 
        df = pd.read_pickle(filename)
        
        diff_1 = df.time.iloc[-1]-df.time
        
        dates = np.flip(np.array([d.total_seconds() for d in diff_1])/60)
        
        df['dates'] = dates
    
    

    
    fig, ax1=plt.subplots(1)
    #Plot Mercury
    
    theta = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(theta)
    y = np.sin(theta)-0.2
    
    # Plot the circle in all 3 plots
    ax1.plot(x, y, color='gray')

    # Color the left hemisphere red and the right hemisphere gray
    ax1.fill_between(x, y, where=x<0, color='black', interpolate=True)

    fs=18
    #Set equal aspect so Mercury is circular
    ax1.set_aspect('equal', adjustable='box')

    ax1.set_xlim([-4.5, 2])
    ax1.set_ylim([-5, 1.7])
    

    
    ax1.set_xlabel("$X_{MSM'}$",fontsize=fs)
    
    ax1.set_ylabel("$Z_{MSM'}$",fontsize=fs)
        
    make_alfven_wing(icme_21['Ma'],ax1,-1,icme_21['B_IMF'])
    
    make_alfven_wing(icme_21['Ma'],ax1,1,icme_21['B_IMF'])
    
    dates_aw = np.load('dates_21_aw.npy')
    
    
    
    scale_factor=3
    

    

    
    ax1.scatter(df.ephx,df.ephz,c=df.dates,cmap='tab20',s=4)
    
    df = df[((df.dates>dates_aw[0]) & (df.dates<dates_aw[-1]))]
    
    if diff == True:
        
        colors='green'
    
        mx=df.magx_diff.to_numpy()/df.magamp_diff.to_numpy()
        
        my=df.magy_diff.to_numpy()/df.magamp_diff.to_numpy()
        
        mz=df.magz_diff.to_numpy()/df.magamp_diff.to_numpy()
        
    if full == True:
        
        colors = 'blue'
        
        mx=df.magx.to_numpy()/df.magamp.to_numpy()
        
        my=df.magy.to_numpy()/df.magamp.to_numpy()
        
        mz=df.magz.to_numpy()/df.magamp.to_numpy()
        
    if model == True:
        
        colors = 'red'
        
        mx=df.magx_kth.to_numpy()/df.magamp_kth.to_numpy()
        
        my=df.magy_kth.to_numpy()/df.magamp_kth.to_numpy()
        
        mz=df.magz_kth.to_numpy()/df.magamp_kth.to_numpy()
    
    ex = df.ephx.to_numpy()
    
    ey = df.ephy.to_numpy()
    
    ez = df.ephz.to_numpy()
    

        
    
    dates = df.dates.to_numpy()
    
    ds = 10
    
    # diff = df.time.iloc[-1]-df.time
    
    # dates = np.flip(np.array([d.total_seconds() for d in diff])/60)
    
    
    mx = downsample(ds, mx)
    
    my = downsample(ds, my)
    
    mz = downsample(ds, mz)
    
    ex = downsample(ds, ex)
    ey = downsample(ds, ey)
    ez = downsample(ds, ez)
    
    
    dates=downsample(ds,dates)
    
    
    

    
    ax1.quiver(ex,ez,mx/scale_factor,mz/scale_factor,scale=1, color=colors, width=0.005, headwidth=3, headlength=4)
    format_ticks(ax1,18)
    
    
#def create_event_dfs(event):
    
def create_event_dfs(event):
    
    i=event
    
    icme=pd.read_excel(ICME_List_file)
    
    time_a = icme[['ICME_A_YEAR','ICME_A_DOY','ICME_A_HH','ICME_A_MM','ICME_A_SS']]\
        .iloc[i]
        
    time_e = icme[['ICME_E_YEAR','ICME_E_DOY','ICME_E_HH','ICME_E_MM','ICME_E_SS']]\
        .iloc[i]
        
    def create_datestring(year, day_of_year, hour, minute, second):
        # Create a datetime object for January 1st of the given year
        date = datetime.datetime(int(year), 1, 1)
    
        # Add the number of days (day_of_year - 1) to get to the desired date
        date += datetime.timedelta(days=float(day_of_year) - 1, hours=float(hour), minutes=float(minute),seconds=float(second))
        
        def convert_datetime_to_string(time):
            dstring=time.strftime("%Y-%m-%d %H:%M:%S")
            return dstring
        
        return convert_datetime_to_string(date), date
    
    date_string_a, date_a = create_datestring(time_a[0], time_a[1], time_a[2], time_a[3], time_a[4])
    
    date_string_e, date_e = create_datestring(time_e[0], time_e[1], time_e[2], time_e[3], time_e[4])
    
    
    
    
    trange_ICME=[date_string_a,date_string_e]
    
    date_a = convert_to_datetime(date_string_a)
    
    date_e = convert_to_datetime(date_string_e)
    
    fd=pd.read_pickle('/Users/bowersch/Desktop/Python_Code/PUB_ANN_Test/fd_prep_w_boundaries.pkl')
    
    df= fd[((fd.time > date_a) & (fd.time < date_e))]
    
    return df

    #df.to_pickle('ICME_Event_'+str(event)+'.pkl')
    
def distance_point_to_curve(p, curve,get_point=False):

    """
    Calculates the distance between a point and a curve.

    Arguments:
    p -- A tuple or list representing the coordinates of the point (px, py).
    curve -- A list of tuples or lists representing the points on the curve [(x1, y1), (x2, y2), ...].

    Returns:
    distance -- The minimum distance between the point and the curve.
    get_point -- If you want to backout specific the point on the curve that is at 
    the minimum distance to p
    """

    # Initialize the minimum distance with a large value
    min_distance = float('inf')
    
    import matplotlib.path as mpl_path
    import matplotlib.pyplot as plt

    def is_point_inside_curve(curve_points, point):
        # Create a Path object from the curve points
        path = mpl_path.Path(curve_points)
        
        # Check if the given point lies inside the curve
        return path.contains_point(point)
    
    import numpy as np

    # Iterate over each point on the curve
    for point in curve:
        # Calculate the Euclidean distance between the point and the curve point
        distance = np.sqrt((p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2)

        # Update the minimum distance if the calculated distance is smaller
        if distance < np.abs(min_distance):
            min_distance = distance
            
            a=is_point_inside_curve(curve,[p[0],p[1]])
            
            cp_r=np.sqrt(point[0]**2+point[1]**2)
            
            r=np.sqrt(p[0]**2+p[1]**2)
            
            if a==True:
                min_distance = min_distance*(-1)
                
                
            min_point=point
    if get_point==False:
        return min_distance
    
    if get_point==True:
        return min_distance,min_point    

def event_properties(event,parameter):
    
    p = np.array([])
    
    for i in range(11):
        
        icme_event=pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Dictionary_'+str(event)+'_'+str(i)+'.pkl')
        
        if i != 6:
        
            p = np.append(p,icme_event[parameter])
            
        if i == 6:
            
            print('ICME '+parameter+' : '+str(icme_event[parameter]))
            
            
    return p
        

def estimate_m_number(event):
    #df = pd.read_pickle('ICME_Event_'+str(event)+'_'+str(6)+'.pkl')
    
    print(event)
    df = create_event_dfs(event)
    
    df_boundaries = pd.read_pickle('df_boundaries_philpott.pkl')
    
    mp_in_1 = df_boundaries[df_boundaries.Cross_Type=='mp_in_1'].time
    mp_in_2 = df_boundaries[df_boundaries.Cross_Type=='mp_in_2'].time
    
    mp_out_1 = df_boundaries[df_boundaries.Cross_Type=='mp_out_1'].time
    mp_out_2 = df_boundaries[df_boundaries.Cross_Type=='mp_out_2'].time
    
    mp_1 = np.sort(np.append(mp_in_1,mp_out_1))
    
    mp_2 = np.sort(np.append(mp_in_2,mp_out_2))
    
    mp_time = np.stack((mp_1,mp_2),axis=1)
    
    mp_time = mp_time.astype('datetime64[s]').astype('datetime64[us]').astype(object)
    
    bs_in_1 = df_boundaries[df_boundaries.Cross_Type=='bs_in_1'].time
    bs_in_2 = df_boundaries[df_boundaries.Cross_Type=='bs_in_2'].time
    
    bs_in_1 = bs_in_1.astype('datetime64[s]').astype('datetime64[us]').astype(object).to_numpy()
    
    bs_in_2 = bs_in_2.astype('datetime64[s]').astype('datetime64[us]').astype(object).to_numpy()
    
    bs_out_1 = df_boundaries[df_boundaries.Cross_Type=='bs_out_1'].time
    bs_out_2 = df_boundaries[df_boundaries.Cross_Type=='bs_out_2'].time
    
    bs_out_1 = bs_out_1.astype('datetime64[s]').astype('datetime64[us]').astype(object).to_numpy()
    
    bs_out_2 = bs_out_2.astype('datetime64[s]').astype('datetime64[us]').astype(object).to_numpy()

    
    bs_1 = np.sort(np.append(bs_in_1,bs_out_1))
    
    bs_2 = np.sort(np.append(bs_in_2,bs_out_2))
    
    bs_time = np.stack((bs_1,bs_2),axis=1)
    
    bs_time = bs_time.astype('datetime64[s]').astype('datetime64[us]').astype(object)
    

    time = df.time
    
    gd_mp = np.where((mp_time[:,0] > time.iloc[0]) & (mp_time[:,1] < time.iloc[-1]))[0]
    
    t_mp = mp_time[gd_mp,:]
    
    
    gd_bs = np.where((bs_time[:,0] > time.iloc[0]) & (bs_time[:,1] < time.iloc[-1]))[0]
    
    t_bs = bs_time[gd_bs,:]
    
    
    x_mp = np.zeros((0,2))
    
    r_mp = np.zeros((0,2))
    
    x_bs = np.zeros((0,2))
    
    r_bs = np.zeros((0,2))
    
    for l in range(len(t_mp)):
        
        df_0 = df[(df.time > t_mp[l,0]) & (df.time < t_mp[l,0]+datetime.timedelta(seconds=1.5))]
    
        df_1 = df[(df.time > t_mp[l,1]) & (df.time < t_mp[l,1]+datetime.timedelta(seconds=1.5))]
    
        x_mp0 = np.mean(df_0.ephx)
        
        r_mp0 = np.mean(np.sqrt(df_0.ephy**2+df_0.ephz**2))
        
        x_mp1 = np.mean(df_1.ephx)
        
        r_mp1 = np.mean(np.sqrt(df_1.ephy**2+df_1.ephz**2))
        
        x_mp = np.append(x_mp,[[x_mp0,x_mp1]],axis=0)
        
        r_mp = np.append(r_mp, [[r_mp0,r_mp1]],axis=0)
        
    for l in range(len(t_bs)):
    
        df_0 = df[(df.time > t_bs[l,0]) & (df.time < t_bs[l,0]+datetime.timedelta(seconds=1.5))]
    
        df_1 = df[(df.time > t_bs[l,1]) & (df.time < t_bs[l,1]+datetime.timedelta(seconds=1.5))]
    
        x_bs0 = np.mean(df_0.ephx)
        
        r_bs0 = np.mean(np.sqrt(df_0.ephy**2+df_0.ephz**2))
        
        x_bs1 = np.mean(df_1.ephx)
        
        r_bs1 = np.mean(np.sqrt(df_1.ephy**2+df_1.ephz**2))
        
        x_bs = np.append(x_bs,[[x_bs0,x_bs1]],axis=0)
        
        r_bs = np.append(r_bs, [[r_bs0,r_bs1]],axis=0)
        
    
    #eph_mp = mp['eph']
    
    
    
    r_mp_range = np.abs((r_mp[:,1]-r_mp[:,0])/2)
    
    r_mp = np.mean(r_mp,axis=1)
    
    x_mp_range = np.abs((x_mp[:,1]-x_mp[:,0])/2)
    
    x_mp = np.mean(x_mp,axis=1)
    
    
    r_bs_range = np.abs((r_bs[:,1]-r_bs[:,0])/2)
    
    r_bs = np.mean(r_bs,axis=1)
    
    x_bs_range = np.abs((x_bs[:,1]-x_bs[:,0])/2)
    
    x_bs = np.mean(x_bs,axis=1)
    
    
    
    def estimate_MA():
        
        MA = np.array([])
        
        psw_total = np.array([])
        
        B_SW_total = np.array([])
        
        Bx_total = np.array([])
        By_total = np.array([])
        Bz_total = np.array([])
        
        phi_total = np.array([])
        
        Bx_std_total = np.array([])
        
        By_std_total = np.array([])
        
        Bz_std_total = np.array([])
        
        
        B_SW_std_total = np.array([])
        
        pmag_std_total = np.array([])
        
        phi_std_total = np.array([])
        
        pmag_total = np.array([])
        
        psw_std_total = np.array([])
        
        MA_std = np.array([])
        
        
        for p in range(len(gd_mp)):
            
            skip = False
            mp_crossing = mp_time[gd_mp[p],:]
            
            if len(gd_bs) == 0:
                
                bs_crossing = mp_crossing
            
            
            if len(gd_bs) == 1:
                
                bs_crossing = bs_time[gd_bs[0],:]
                
            if len(gd_bs) > 1:
                
                try:
                    
                    bs_crossing = bs_time[gd_bs[p],:]
                    
                except:
                    
                    bs_crossing = bs_time[0,:]
                    
                    skip = True
            
         
            if ((bs_crossing[0] in bs_in_1) | (bs_crossing[0] in bs_in_2)) :
                
                up_range = np.array([bs_crossing[0] - datetime.timedelta(minutes=30),bs_crossing[0]])
                
                mp_range = np.array([mp_crossing[1], mp_crossing[1] + datetime.timedelta(minutes=1)])
                


                
            else:
                
                up_range = np.array([bs_crossing[1], bs_crossing[1] + datetime.timedelta(minutes=30)])
                
                mp_range = np.array([mp_crossing[0] - datetime.timedelta(minutes=1), mp_crossing[0]])
                
    
            magamp=df.magamp.to_numpy()
            
            
            mag=df[['magx','magy','magz']].to_numpy()
            
            eph=df[['ephx','ephy','ephz']].to_numpy()
            
            #breakpoint() 
            Rss=1.45
            alpha=0.5
    
            phi2 = (np.linspace(0,2*np.pi,1000))
    
            rho=Rss*(2/(1+np.cos(phi2)))**(alpha)
    
            xmp=rho*np.cos(phi2)
    
            ymp=rho*np.sin(phi2)
    
            curve=np.transpose(np.vstack((xmp,ymp)))
            
            ea_test = np.mean(eph[(df.time > mp_range[0]) & (df.time < mp_range[1]),:],axis=0)
            
            ea_std = np.std(eph[(df.time > mp_range[0]) & (df.time < mp_range[1]),:],axis=0)
            
            df_MP=df[(df.time > mp_range[0]) & (df.time < mp_range[1])]
            
            percentile = np.percentile(df_MP.magamp.to_numpy(), 80)
            
            df_MP=df_MP[(df.magamp > percentile)]
            
            
            B_MP=np.mean(df_MP.magamp.to_numpy())
            
            B_MP_std = np.std(df_MP.magamp.to_numpy())
            
            B_SW = np.mean(magamp[(df.time > up_range[0]) & (df.time < up_range[1])])
            
            B_SW_std = np.std(magamp[(df.time > up_range[0]) & (df.time < up_range[1])])
            
            B_X = np.mean(mag[(df.time > up_range[0]) & (df.time < up_range[1]),0])
            B_Y = np.mean(mag[(df.time > up_range[0]) & (df.time < up_range[1]),1])
            B_Z = np.mean(mag[(df.time > up_range[0]) & (df.time < up_range[1]),2])
            
            B_X_std = np.std(mag[(df.time > up_range[0]) & (df.time < up_range[1]),0])
            B_Y_std = np.std(mag[(df.time > up_range[0]) & (df.time < up_range[1]),1])
            B_Z_std = np.std(mag[(df.time > up_range[0]) & (df.time < up_range[1]),2])
            munaught=4*np.pi*1E-7
            
            ra_test=np.sqrt(ea_test[1]**2+ea_test[2]**2)
            
            ra_std = 0.5*(2*ea_std[1]+2*ea_std[2])
            
            # Find distance to inner point in curve (mp uncertainty)
            
            da,point=distance_point_to_curve([ea_test[0]-ea_std[0],ra_test-ra_std],curve,get_point=True)
            
            diff=curve-[point[0],point[1]]
            
            
            diff=np.sqrt(diff[:,0]**2+diff[:,1]**2)
            
            index=np.where(np.abs(diff)==np.min(np.abs(diff)))[0][0]
            
            n=calculate_normal_vector(curve,index)
            
            phi_1=np.arccos(np.dot(n,np.array([1,0])))
            
            #Find distance to outer point in curve (mp uncertainty)
            
            da,point=distance_point_to_curve([ea_test[0]+ea_std[0],ra_test+ra_std],curve,get_point=True)
            
            diff=curve-[point[0],point[1]]
    
            diff=np.sqrt(diff[:,0]**2+diff[:,1]**2)
            
            index=np.where(np.abs(diff)==np.min(np.abs(diff)))[0][0]
            
            n=calculate_normal_vector(curve,index)
            
            phi_2=np.arccos(np.dot(n,np.array([1,0])))
            
            phi = np.mean([phi_1,phi_2])
            
            phi_std = np.abs(phi_2-phi_1)
            
            
            B_MP=B_MP*1E-9
            
            pmag=B_MP**2/(2*munaught)
            
            pmag_std=pmag*B_MP_std*1E-9/B_MP*2
            
    
            
            B_SW=B_SW*1E-9
            
            B_MP_std=B_MP_std*1E-9
        
            B_SW_std = B_SW_std*1E-9
            
            rel_uncertainty_mp = B_MP_std/B_MP
            
            rel_uncertainty_sw = B_SW_std/B_SW
            
            psw1=(B_MP**2/(2*munaught) - B_SW**2/(2*munaught))*(.88*np.cos(phi)**2)**(-1)
           
            psw_max =  ((B_MP+B_MP_std)**2/(2*munaught) - (B_SW-B_SW_std)**2/(2*munaught))*(.88*np.cos(np.max([phi_1,phi_2]))**2)**(-1)
           
            psw_min =  ((B_MP-B_MP_std)**2/(2*munaught) - (B_SW+B_SW_std)**2/(2*munaught))*(.88*np.cos(np.min([phi_1,phi_2]))**2)**(-1) 
           
            
           
            psw_std = psw_max-psw1
           
            if ((phi >= 120*np.pi/180) | (phi <= np.radians(60))):
                
                if skip == False:
            
                    MA = np.append(MA,np.sqrt(psw1*munaught)/B_SW)
                    
                    MA_min = np.sqrt(psw_min*munaught)/(B_SW+B_SW_std)
                    
                    MA_max = np.sqrt(psw_max*munaught)/(B_SW-B_SW_std)
                    
                    MA_std = np.append(MA_std,MA_max-MA_min)
                    
                    pmag_total = np.append(pmag_total, pmag)
                    psw_total = np.append(psw_total,psw1)
                    
                    B_SW_total = np.append(B_SW_total, B_SW)
                    
                    Bx_total = np.append(Bx_total, B_X)
                    
                    Bx_std_total = np.append(Bx_std_total, B_X_std)
                    
                    By_std_total = np.append(By_std_total, B_Y_std)
                    
                    Bz_std_total = np.append(Bz_std_total, B_X_std)
                    
                    By_total = np.append(By_total, B_Y)
                    
                    Bz_total = np.append(Bz_total, B_Z)
                    
                    phi_total = np.append(phi_total,phi)
                    
                    B_SW_std_total = np.append(B_SW_std_total, B_SW_std)
                    
                    pmag_std_total = np.append(pmag_std_total, pmag_std)
                    
                    phi_std_total = np.append(phi_std_total,phi_std)
                    
                    psw_std_total = np.append(psw_std_total,psw_std)
        
        return MA,psw_total,B_SW_total,np.stack((Bx_total,By_total,Bz_total)),pmag_total,phi_total, \
            MA_std,psw_std_total,B_SW_std_total,np.stack((Bx_std_total,By_std_total,Bz_std_total)),\
                pmag_std_total,phi_std_total
                
    Ma,psw1,B_SW,B_IMF,pmag,phi,Ma_std,psw_std,B_SW_std,B_IMF_std,pmag_std,phi_std = estimate_MA()
    munaught=4*np.pi*1E-7
    
    # For all events:
        
        # all_events = 
    
    return Ma,psw1*1E9,B_SW*1E9,B_IMF,pmag*1E9,180-phi*180/np.pi,Ma_std,psw_std*1E9,B_SW_std*1E9,B_IMF_std,pmag_std*1E9,phi_std*180/np.pi
    

def estimate_Mach_Number(event,mp_time,up_range,aw_range,cusp_range,orbit_num):
    
   # df=pd.read_pickle('ICME_Event_'+str(event)+'.pkl')
    
    df = pd.read_pickle('ICME_Event_'+str(event)+'_'+str(orbit_num)+'.pkl')
    
    mp_time = np.array([df.time.iloc[0]+datetime.timedelta(minutes=mp_time[i]) for i in range(2)]) 
    
    up_range = np.array([df.time.iloc[0]+datetime.timedelta(minutes=up_range[i]) for i in range(2)]) 
    
    #aw_range = np.array([df.time.iloc[0]+datetime.timedelta(minutes=aw_range[i]) for i in range(2)])
    
    cusp_range = np.array([df.time.iloc[0]+datetime.timedelta(minutes=cusp_range[i]) for i in range(2)]) 
    
    magamp=df.magamp.to_numpy()
    
    mag=df[['magx','magy','magz']].to_numpy()
    
    eph=df[['ephx','ephy','ephz']].to_numpy()
    
    timespan=[mp_time[0],mp_time[1]+datetime.timedelta(seconds=60)]
    Rss=1.45
    alpha=0.5

    phi2 = (np.linspace(0,2*np.pi,100))

    rho=Rss*(2/(1+np.cos(phi2)))**(alpha)

    xmp=rho*np.cos(phi2)

    ymp=rho*np.sin(phi2)

    curve=np.transpose(np.vstack((xmp,ymp)))
    
    ea_test = np.mean(eph[(df.time > timespan[0]) & (df.time < timespan[1]),:],axis=0)
    
    df_MP=df[(df.time > timespan[0]) & (df.time < timespan[1])]
    
    percentile = np.percentile(df_MP.magamp.to_numpy(), 80)
    
    df_MP=df_MP[(df.magamp > percentile)]
    
    
    B_MP=np.mean(df_MP.magamp.to_numpy())
    
    B_SW = np.mean(magamp[(df.time > up_range[0]) & (df.time < up_range[1])])
    
    gd_cusp = np.where((df.time > cusp_range[0]) & (df.time < cusp_range[1]))[0]
    
    B_Cusp = np.mean(magamp[(df.time > cusp_range[0]) & (df.time < cusp_range[1])])
    
    
    
    timespan_for_mva=[mp_time[0]-datetime.timedelta(seconds=150), \
                      mp_time[1]+datetime.timedelta(seconds=150)]
        
    print(timespan)
        

    
    munaught=4*np.pi*1E-7
    
    magx_kth_0,magy_kth_0,magz_kth_0,magamp_kth_0 =\
        get_KTH_pred_from_MSM(df.ephx, df.ephy, df.ephz, convert_datetime_to_string(df.time.iloc[0])[0:10],0)
    
    ra_test=np.sqrt(ea_test[1]**2+ea_test[2]**2)
    
    
    da,point=distance_point_to_curve([ea_test[0],ra_test],curve,get_point=True)
    
    diff=curve-[point[0],point[1]]
    
    
    diff=np.sqrt(diff[:,0]**2+diff[:,1]**2)
    
    index=np.where(np.abs(diff)==np.min(np.abs(diff)))[0][0]
    
    n=calculate_normal_vector(curve,index)
    
    

    phi=np.arccos(np.dot(n,np.array([1,0])))
    
    

    if phi > np.pi: phi=np.abs(phi-np.pi)
    
    B_MP=B_MP*1E-9
    
    B_SW=B_SW*1E-9
    
    B_Cusp = B_Cusp*1E-9
    
    B_model = np.mean(magamp_kth_0[gd_cusp])*1E-9
    
    Pthermal = (B_model-B_Cusp)**2/(2*munaught)
    
    psw1=(B_MP**2/(2*munaught) - B_SW**2/(2*munaught) + Pthermal)*(.88*np.cos(phi)**2)**(-1)
    
    
    
    MA = np.sqrt(psw1*munaught)/B_SW
    
    
    return MA
    
       
def create_rom_ganymede():
    
    import xarray as xr
    
    scale_g = 1.85
    
    scale_m = 1.45
    
    plot_eph = True
    
    Beq_G = 750
    
    file_path = '/Users/bowersch/Desktop/Python_Code/Alfven_Wing/17341229/LatHyS_Ganymede_Mag_3D_t00300.nc'
    df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_21_6.pkl')
    dataset = xr.open_dataset(file_path)

    
    diff = df.time.iloc[-1]-df.time
    
    dates =np.array([d.total_seconds() for d in diff])/60
    
    dates = np.max(dates)-dates
   
    
    bounds = np.logspace(-2, -1, 15)
    cmap = plt.get_cmap('jet', 15)
    import matplotlib.colors as cl
    norm = cl.BoundaryNorm(bounds, cmap.N)
    
    
    
    x_axis = np.array(dataset['X_axis'])/2634.1/scale_g
    
    z_axis = np.array(dataset['Z_axis'])/2634.1/scale_g
    
    y_axis = np.array(dataset['Y_axis'])/2634.1/scale_g
    
    Bx = np.array(dataset['Bx'])*-1
    
    By = np.array(dataset['By'])
    
    Bz = np.array(dataset['Bz'])
    
    magamp = np.sqrt(Bx**2+By**2+Bz**2)
    
    #Bx = np.flip(Bx,axis=2)
    
    X,Z = np.meshgrid(x_axis*-1,z_axis)
    
    fig,ax = plt.subplots(1)
    
    fig1=ax.pcolormesh(X,Z,Bz[:-1,151,:-1]/Beq_G,shading='flat',cmap='coolwarm',vmin=-50,vmax=50)

    
    
    fig,ax = plt.subplots(1)
    
    fig1=ax.pcolormesh(X,Z,Bx[:-1,151,:-1]/Beq_G,shading='flat',cmap='coolwarm',vmin=-.5,vmax=.5)

    fig.colorbar(fig1, ax=ax,label='$B_{X}$/$B_{eq,G}$')
    
    if plot_eph == True:
    
        ax.scatter(df.ephx/scale_m,(df.ephz+.19)/scale_m,c=dates,cmap=cmap)

        format_eph_plot(ax)
        
    else:
        
        format_eph_plot(ax,ganymede=True)
    
    def estimate_orbit_time_series(ax):
        
        df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_21_6.pkl')
        diff = df.time.iloc[-1]-df.time
        
        dates =np.array([d.total_seconds() for d in diff])/60
        
        dates = np.max(dates)-dates
        
        df[((dates>105) & (dates<107))]=np.nan
        
        
        path_x = df.ephx.to_numpy()*-1/scale_m
        
        path_y = df.ephy.to_numpy()/scale_m
        
        path_z = (df.ephz.to_numpy()+.19)/scale_m
        
        dates_d = downsample(40,dates)
        
        path_x = downsample(40,path_x)
        
        path_y = downsample(40,path_y)
        
        path_z = downsample(40,path_z)
        
        
        X,Y,Z = np.meshgrid(x_axis,y_axis,z_axis)
        
        def interpolate_param_along_path(param_grid, path_x, path_y, path_z):
            
            import scipy.ndimage
            
            path_coords = np.vstack((path_x, path_y, path_z))
            
            param_array = np.array([])
            
            for i in range(len(path_x)):
                
                x = path_x[i]
                y = path_y[i]
                z = path_z[i]
                
                ix = np.searchsorted(x_axis, x) - 1
                iy = np.searchsorted(y_axis, y) - 1
                iz = np.searchsorted(z_axis, z) - 1
                

                
                param_array = np.append(param_array,param_grid[iz,iy,ix])
                
                
            
            
            return param_array
        
        
        Bx_along_path = interpolate_param_along_path(Bx,path_x,path_y,path_z)
        By_along_path = interpolate_param_along_path(By,path_x,path_y,path_z)
        Bz_along_path = interpolate_param_along_path(Bz,path_x,path_y,path_z)
        magamp_along_path = interpolate_param_along_path(magamp,path_x,path_y,path_z)
        
        
        #ax.scatter(df.ephx,df.ephz+.19,c = param_along_path,cmap = 'coolwarm',vmin=-50,vmax=50)
        Beq_G = 750
    
        Beq_M = 1
        
        
        fig,ax = plt.subplots(6,sharex=True,gridspec_kw={'height_ratios': [1,1,1,1,1,0.3]})
        
        fig,ax2 = plt.subplots(3,sharex=True,gridspec_kw={'height_ratios': [1,1,0.3]})
        
        axes = ax[0:-1]
        
        colors = ['orange','black']
        
        lw = 1.5
        
        labels = ['Ganymede Model','ICME Event']
        
        ylabels = ['$BX/B_{eq,X}$','$BY/B_{eq,Y}$','$BZ/B_{eq,Z}$','$|B|/|B_{eq}|$','n ($cm^{-3}$)',' ']
        
        ylabels = ["$BX_{MSM'}$ (nT)","$BY_{MSM'}$ (nT)","$BZ_{MSM'}$ (nT)",'|B| (nT)', 'n ($cm^{-3}$)', ' ']
        
        ax2[0].plot(dates_d,magamp_along_path/Beq_G,label = labels[0],linewidth = lw,color = colors[0])
        ax2[0].plot(dates,df.magamp/Beq_M,label = labels[1],linewidth = lw,color = colors[1])
        
        ax2[len(ax2)-1].imshow(dates.reshape(1, -1), cmap=cmap, aspect='auto',extent=[dates[0],dates[-1],-1,1],label='Minutes')

        ax2[len(ax2)-1].set_yticks([])
        
        ax[len(ax)-1].set_xlabel('Minutes from Apoapsis')
        ax2[2].set_xlabel('Minutes from Apoapsis')
        ax2[1].set_yscale('log')
        for x in range(len(axes)):
            format_ticks(axes[x],15)
            axes[x].axhline(y=0.0, color='black', linestyle='--',linewidth=0.5)
            axes[x].set_ylabel(ylabels[x])
            
        for x in range(len(ax2)):
            format_ticks(ax2[x],15)
            ax2[x].axhline(y=0.0, color='black', linestyle='--',linewidth=0.5)
            ax2[x].set_ylabel(ylabels[x+3])
            
            
        
        #ax[0].plot(dates_d,Bx_along_path/Beq_G,label = labels[0],linewidth = lw,color = colors[0])
        
        ax[0].plot(dates,df.magx/Beq_M,label = labels[1],linewidth = lw,color = colors[1])
        
        #ax[1].plot(dates_d,By_along_path/Beq_G,label = labels[0],linewidth = lw,color = colors[0])
        
        ax[1].plot(dates,df.magy/Beq_M,label = labels[1],linewidth = lw,color = colors[1])
        
        #ax[2].plot(dates_d,Bz_along_path/Beq_G,label = labels[0],linewidth = lw,color = colors[0])
        
        ax[2].plot(dates,df.magz/Beq_M,label = labels[1],linewidth = lw,color = colors[1])
        
        #ax[3].plot(dates_d,magamp_along_path/Beq_G,label = labels[0],linewidth = lw,color = colors[0])
        
        ax[3].plot(dates,df.magamp/Beq_M,label = labels[1],linewidth = lw,color = colors[1])
        
        ax[len(ax)-1].imshow(dates.reshape(1, -1), cmap=cmap, aspect='auto',extent=[dates[0],dates[-1],-1,1],label='Minutes')

        ax[len(ax)-1].set_yticks([])
        
        df[((dates>105) & (dates<107))]=np.nan
        
        ganymede_model = pd.DataFrame(data = {'magx':Bx_along_path,'magy':By_along_path,\
                                              'magz':Bz_along_path,'magamp':magamp_along_path})
            
        ganymede_model.to_pickle('ganymede_model.pkl')
                                              
        
        ax[3].set_ylim(0,1.5)
        
        
        
        n_along_path = create_Fatemi_ganymede()
        
        df_g = pd.DataFrame(data = {'n':n_along_path})
        
        #add_onto_density_panel(ax[4],dates_d,df_g.n,colors[0],3.3,1,'Ganymede')
        #add_onto_density_panel(ax2[1],dates_d,df_g.n,colors[0],3.3,1,'Ganymede')
        
        fips_data = pd.read_pickle('/Users/bowersch/FIPS_Dictionary_test.pkl')
        
        df_ntp = fips_data['df_ntp']
        
        df_ntp = df_ntp[((df_ntp.time > df.time.iloc[0]) & (df_ntp.time < df.time.iloc[-1]))]
        
        diff = df.time.iloc[-1]-df_ntp.time
        
        dates_ntp =np.array([d.total_seconds() for d in diff])/60
        
        dates_ntp = np.max(dates)-dates_ntp
        
        add_onto_density_panel(ax[4],dates_ntp,df_ntp.n,colors[1],3.3,1,'ICME Orbit')
        add_onto_density_panel(ax2[1],dates_ntp,df_ntp.n,colors[1],3.3,1,'ICME Orbit')

        
        
        dates_a, mag_total, dates_n, n_averaged = average_parameters(21)
        
        #for p in range(4):
            #ax[p].plot(dates_a,mag_total[:,p]/300,color='gray',label='Surrounding Orbits')
        
        ax2[0].plot(dates_a,mag_total[:,3]/300,color='gray',label = 'Surrounding Orbits')
        data_frame = pd.DataFrame(data={'n_averaged':n_averaged})
        
        #add_onto_density_panel(ax[4],dates_n,data_frame.n_averaged,'gray',3.3,1,'Surrounding Orbits')
        ax[4].set_yscale('log')
        ax[4].set_ylim([.001,10])
        
        add_onto_density_panel(ax2[1],dates_n,data_frame.n_averaged,'gray',3.3,1,'Surrounding Orbits')


        handles, labels = ax[3].get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        
        # Plot legend with only unique labels
        #ax[len(ax)-2].legend(unique_labels.values(), unique_labels.keys(),loc='upper right')
        #ax[4].plot(df.time,n_array/np.max(n_array))

        #df_nobs= pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_NOBS_21_6.pkl')
        handles, labels = ax2[1].get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        
        # Plot legend with only unique labels
        #ax[len(ax2)-2].legend(unique_labels.values(), unique_labels.keys(),loc='upper right')
        #ax[4].plot(df_nobs.time,df_nobs.n/np.max(df_nobs.n))
        
    estimate_orbit_time_series(ax)
    
    
        
    
    return      


def format_eph_plot(axis,ganymede = False):
    fs=18
    #axis.fill_between([10,-10],[-10,-10],color='gainsboro')
    
    #axis.fill_between([10,-10],[10,10],color='gainsboro')
    
    if ganymede == True:
        scale_factor = 1
        
        offset = 0
        
    if ganymede == False:
        
        offset = .19
        scale_factor = 1.45
    
    theta = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(theta)/scale_factor
    y = (np.sin(theta)-offset)/scale_factor
    
    axis.plot(x, y, color='black')
    
    axis.set_aspect('equal',adjustable='box')
    
    # X component
    xlim=[-5,5]
    
    ylim=[-8,2]
    
    
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    
    if ganymede == False:
        
        axis.set_xlabel("$X_{MPS}$ ($R_{MP}$)",fontsize=fs)
    
        axis.set_ylabel("$Z_{MPS}$ ($R_{MP}$)",fontsize=fs)
        
    if ganymede == True:
        axis.set_xlabel("$X_{G}$ ($R_G$)",fontsize=fs)
    
        axis.set_ylabel("$Z_{G}$ ($R_G$)",fontsize=fs)
        
    
    axis.tick_params(labelsize=fs-4)
    axis.fill_between(x,y,color='white',interpolate = True)
    axis.fill_between(x, y, where=x<0, color='black', interpolate=True)
    axis.fill_between(x, y, where=x<0,color='black',interpolate=True)
    
    
    
    axis.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    
    axis.tick_params(axis='both', which='major', length=8)
    axis.tick_params(axis='both', which='minor', length=4)  
      
def create_Fatemi_ganymede(uz=False):
    
    if uz == False:
        density_mesh_file = '/Users/bowersch/Desktop/Python_Code/Alfven_Wing/Fig_2b.txt'
        
    if uz == True:
        density_mesh_file = '/Users/bowersch/Desktop/Python_Code/Alfven_Wing/Fig_2d.txt'
    
    mesh = np.genfromtxt(density_mesh_file)
    
    scale_g = 1
    
    scale_m = 1
    
    x_axis = np.linspace(-4,4,num = len(mesh[0,:]))*-1/scale_g
    z_axis = np.linspace(-7.5,7.5, num = len(mesh[:,0]))/scale_g
    
    X,Z = np.meshgrid(x_axis,z_axis)
    
    fig, ax = plt.subplots(1)
    
    df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_21_6.pkl')
    
    mesh[(mesh==0)]=0
    
    if uz == False:
        fig1=ax.pcolormesh(X,Z,mesh[:-1,:-1]/8,shading='flat',cmap='plasma',norm=colors.LogNorm(vmin=.1))
    
        fig.colorbar(fig1, ax=ax,label = '$n/n_0$',)
        
    if uz == True:
        fig1=ax.pcolormesh(X,Z,mesh[:-1,:-1]/-140,shading='flat',cmap='bwr',vmin=-2.1,vmax=2.1)
        fig.colorbar(fig1, ax=ax,label = '$uz/u0$',)
    format_eph_plot(ax,ganymede=True)

    path_x = df.ephx.to_numpy()*1/scale_m
    
    path_y = df.ephy.to_numpy()/scale_m
    
    path_z = (df.ephz.to_numpy()+.19)/scale_m
    
    path_x  = downsample(40,path_x)
    
    path_y = downsample(40,path_y)
    
    path_z = downsample(40,path_z)
    

    X,Z = np.meshgrid(x_axis,z_axis)
    
    x_axis = np.linspace(-4,4,num = len(mesh[0,:]))/scale_g
    z_axis = np.linspace(-7.5,7.5, num = len(mesh[:,0]))/scale_g
    
    def interpolate_param_along_path(param_grid, path_x, path_z,x_axis,y_axis):
        
        param_array = np.array([])
        
        
        
        for i in range(len(path_x)):
            
            xi = path_x[i]

            zi = path_z[i]
            
            ix = np.searchsorted(x_axis, xi) - 1
            iz = np.searchsorted(z_axis, zi) - 1  
            
            ix  = len(x_axis)-ix-2
            
            if ((np.abs(xi) > np.max(x_axis)) or (np.abs(zi)) > np.max(z_axis)):
                
                param_array = np.append(param_array, np.nan)

                
            else:
            
                param_array = np.append(param_array,param_grid[iz,ix])
            
        
        return param_array
    
    
     #path_x = np.linspace(-2,2,num=len(path_x))
     #path_z = np.zeros(len(path_x))-1
     #path_y = np.zeros(len(path_y))
    
    
    n_along_path = interpolate_param_along_path(mesh,path_x,path_z,x_axis,z_axis)
    
    df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_21_6.pkl')
    
    diff = df.time.iloc[-1]-df.time
    
    dates =np.array([d.total_seconds() for d in diff])/60
    
    dates = np.max(dates)-dates
   
    
    bounds = np.logspace(-2, -1, 15)
    cmap = plt.get_cmap('jet', 15)
    import matplotlib.colors as cl
    norm = cl.BoundaryNorm(bounds, cmap.N)
    
    #ax.scatter(path_x,path_z,c=downsample(40,dates),cmap=cmap)
    
    fig,ax = plt.subplots(4,sharex = True,gridspec_kw={'height_ratios': [1,1,1,0.3]})
    
    ax[3].imshow(dates.reshape(1, -1), cmap=cmap, aspect='auto',extent=[dates[0],dates[-1],-1,1],label='Minutes')

    ax[3].set_yticks([])
    
    ax[2].plot(downsample(40,dates),path_x,color='blue')
    ax[2].plot(downsample(40,dates),path_y,color='green')
    ax[2].plot(downsample(40,dates),path_z,color='red')
    
    df[((dates>105) & (dates<107))]=np.nan
    
    
    ax[0].plot(dates,df.magamp,color='black')
    
    ax[1].plot(downsample(40,dates),n_along_path,color = 'gray')
    

    
    ylabels = np.array(['|B| (nT)', 'n (cm$^{-3}$)','Path'])
    
    xlabel = 'Minutes from Apoapsis'
    
    ax[len(ax)-1].set_xlabel(xlabel)
    
    for x in range(len(ax)-1):
        format_ticks(ax[x],14)
        
        if x != len(ax)-1:
            
            ax[x].set_ylabel(ylabels[x])
            
    return n_along_path
    
    
    
def generate_philpott_list():
    
    filename = '/Users/bowersch/Downloads/jgra55678-sup-0002-table_si-s01.csv'
    
    df_boundaries = pd.read_csv(filename)
    
    def create_datestring(year, day_of_year, hour, minute, second):
        # Create a datetime object for January 1st of the given year
        date = datetime.datetime(int(year), 1, 1)
    
        # Add the number of days (day_of_year - 1) to get to the desired date
        date += datetime.timedelta(days=float(day_of_year) - 1, hours=float(hour), minutes=float(minute),seconds=float(second))
        
        return date
    
    dt = np.array([create_datestring(df_boundaries.Yr_pass.iloc[p],\
                                                  df_boundaries.Day_orbit.iloc[p],\
                                                  df_boundaries.Hour.iloc[p],\
                                                  df_boundaries.Minute.iloc[p],\
                                                  round(df_boundaries.Second.iloc[p]))\
                                for p in range(len(df_boundaries))])
        
    df_boundaries['time'] = dt
    
    Z_MSM = df_boundaries['Z_MSO (km)']/2440+.19
    
    X_MSM = np.array([])
    
    Y_MSM = np.array([])
    
    cross_string = np.array([])
    
    cross_strings = np.array(['err','bs_in_1','bs_in_2','mp_in_1','mp_in_2',
                               'mp_out_1','mp_out_2','bs_out_1','bs_out_2','gap_1','gap_2'])
    
    for i in range(len(df_boundaries)):
        
    
        X_MSM_1, Y_MSM_1, Z_MSM_2 = rotate_into_msm(df_boundaries['X_MSO (km)'].iloc[i]/2440,
                                         df_boundaries['Y_MSO (km)'].iloc[i]/2440,
                                         Z_MSM[i],
                                         df_boundaries.time.iloc[i])
        
        X_MSM = np.append(X_MSM,X_MSM_1)
        
        Y_MSM = np.append(Y_MSM,Y_MSM_1)
        
        cross_string = np.append(cross_string,cross_strings[df_boundaries['Boundary number'].iloc[i]])
            
            
            
        
            
        
    df_boundaries[['X_MSM','Y_MSM','Z_MSM']] = np.stack((X_MSM,Y_MSM,Z_MSM),axis=1)
    
    df_boundaries['Cross_Type'] = cross_string
    
    df_boundaries.to_pickle('df_boundaries_philpott.pkl')
    
    
def generate_FIPS_flux_map(trange,fr,Na = False):
    import pandas as pd
    import datetime
    import spiceypy
    meta_kernel = '/Users/bowersch/Desktop/MESSENGER Data/msgr_spice_kernels/MESSENGER_meta_kernel.txt'

    import matplotlib.colors as colors
    if spiceypy.ktotal( 'ALL' )==0:
        
        spiceypy.furnsh(meta_kernel)


    from os import sys

    #from lobe_id import shade_in_time_series

    sys.path.append('/Users/bowersch/Desktop/Python_Code/MESSENGER_Lobe_Analysis/')
    from trying3 import check_for_mp_bs_WS,convert_to_datetime,convert_to_date_2,plot_mp_and_bs,\
        plot_MESSENGER_trange_cyl,convert_to_date_2_utc,get_mercury_distance_to_sun,\
            convert_datetime_to_string,plot_MESSENGER_trange_3ax,get_aberration_angle,\
                get_day_of_year,convert_to_date_s, read_in_Weijie_files,load_MESSENGER_into_tplot
    from generate_all_plot import generate_all_plot
    from ANN_work import plot_region_time_series_plots
    import kth22_model_for_mercury_v8 as kth   
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator,AutoMinorLocator
    from Machine_Learning import load_PAD
    from FIPS_FOV_Map import get_FIPS_FOV 
    
    date_string = trange[0][0:10]
    #res can be 01,05,10,60
    doy=get_day_of_year(date_string)
    month=date_string[5:7]
    year=date_string[2:4]
    import os
    
    year_full=date_string[0:4]
    if doy < 10: doy_s='00'+str(doy)
        
    elif (doy<100) & (doy>=10):doy_s='0'+str(doy)
        
    else: doy_s=str(doy)
    
    mcp_file = '/Users/bowersch/Desktop/MESSENGER Data/PA_Tutorial/FIPA_F2009229CDR_V1.TAB' 
    
    pha_file = '/Users/bowersch/Desktop/MESSENGER Data/mess_pha '+year+'/'+str(month)+'/FIPP_P'+year_full+doy_s+'CDR_V3.TAB'


    if os.path.isfile(pha_file)==False:
        pha_file = '/Users/bowersch/Desktop/MESSENGER Data/mess_pha '+year+'/'+str(month)+'/FIPP_P'+year_full+doy_s+'CDR_V2.TAB'
    
    
    scan_file = '/Users/bowersch/Desktop/MESSENGER Data/mess_scan '+year+'/'+str(month)+'/FIPS_R'+year_full+doy_s+'CDR_V3.TAB'
    
    if os.path.isfile(scan_file)==False:
        scan_file = '/Users/bowersch/Desktop/MESSENGER Data/mess_scan '+year+'/'+str(month)+'/FIPS_R'+year_full+doy_s+'CDR_V2.TAB'
        
    
    rotmso_file = '/Users/bowersch/Desktop/MESSENGER Data/mess_rotmso '+year+'/'+str(month)+'/FIPS_ROTMSO_'+year_full+doy_s+'_DDR_V02.TAB'

    if os.path.isfile(rotmso_file)==False:
        rotmso_file = '/Users/bowersch/Desktop/MESSENGER Data/mess_rotmso '+year+'/'+str(month)+'/FIPS_ROTMSO_'+year_full+doy_s+'_DDR_V01.TAB'
    
    erange= np.array([13.577,12.332,11.201,10.174,9.241,8.393,7.623,6.924,
            6.289,5.712,5.188,4.713,4.28,3.888,3.531,3.207,2.913,2.646,2.403,2.183,1.983,1.801,
            1.636,1.485,1.349,1.225,1.113,1.011,0.918,0.834,0.758,0.688,0.625,0.568,
            0.516,0.468,0.426,0.386,0.351,0.319,0.29,0.263,0.239,0.217,0.197,0.179,0.163,0.148,
            0.134,0.122,0.111,0.1,0.091,0.083,0.075,0.068,0.062,0.056,0.051,0.046])
    amu2kg = 1.661E-27
    e2c = 1.602E-19

    m = 1.0
    q = 1.0

    vrange = 438*np.sqrt(erange)

    import numpy as np

    vrange = np.sqrt(2*erange*1000.0/ (m*amu2kg/(q*e2c) ) )
    rotmso = np.genfromtxt(rotmso_file)
    
    # Load data from files
    pha = np.genfromtxt(pha_file)
    
    mcp = np.genfromtxt(mcp_file)
    
    scan = np.genfromtxt(scan_file)
    
    if date_string != trange[1][0:10]:
        
        # Append if it covers more than one day
        
        date_new = convert_to_datetime(date_string+' 00:00:00')+datetime.timedelta(days = 1)
        
        date_string_new = (convert_datetime_to_string(date_new))[0:10]
        
        doy=get_day_of_year(date_string_new)
        month=date_string_new[5:7]
        year=date_string_new[2:4]
        import os
        
        year_full=date_string_new[0:4]
        if doy < 10: doy_s='00'+str(doy)
            
        elif (doy<100) & (doy>=10):doy_s='0'+str(doy)
            
        else: doy_s=str(doy)
        
        pha_file = '/Users/bowersch/Desktop/MESSENGER Data/mess_pha '+year+'/'+str(month)+'/FIPP_P'+year_full+doy_s+'CDR_V3.TAB'


        if os.path.isfile(pha_file)==False:
            pha_file = '/Users/bowersch/Desktop/MESSENGER Data/mess_pha '+year+'/'+str(month)+'/FIPP_P'+year_full+doy_s+'CDR_V2.TAB'
        
        
        scan_file = '/Users/bowersch/Desktop/MESSENGER Data/mess_scan '+year+'/'+str(month)+'/FIPS_R'+year_full+doy_s+'CDR_V3.TAB'
        
        if os.path.isfile(scan_file)==False:
            scan_file = '/Users/bowersch/Desktop/MESSENGER Data/mess_scan '+year+'/'+str(month)+'/FIPS_R'+year_full+doy_s+'CDR_V2.TAB'
            
        
        rotmso_file = '/Users/bowersch/Desktop/MESSENGER Data/mess_rotmso '+year+'/'+str(month)+'/FIPS_ROTMSO_'+year_full+doy_s+'_DDR_V02.TAB'

        if os.path.isfile(rotmso_file)==False:
            rotmso_file = '/Users/bowersch/Desktop/MESSENGER Data/mess_rotmso '+year+'/'+str(month)+'/FIPS_ROTMSO_'+year_full+doy_s+'_DDR_V01.TAB'
        
        rotmso_2 = np.genfromtxt(rotmso_file)
        
        # Load data from files
        pha_2 = np.genfromtxt(pha_file)
        
        mcp_2 = np.genfromtxt(mcp_file)
        
        scan_2 = np.genfromtxt(scan_file)
        
        pha = np.append(pha,pha_2,axis=0)
        
        mcp = np.append(mcp,mcp_2,axis=0)
        
        scan = np.append(scan,scan_2,axis=0)
        
        rotmso = np.append(rotmso, rotmso_2,axis=0)
        
        
        
    # Extract columns from rotmso
    rotmso_met = rotmso[:, 1]
    
    
    rotmatrix_full = np.stack((rotmso[:, 2:5], rotmso[:, 5:8], rotmso[:, 8:11]), axis=1)
    
    # Create a dictionary for fast look-up of rotation matrices by met
    rotmso = {met: np.stack((rotmso[i, 2:5], rotmso[i, 5:8], rotmso[i, 8:11]), axis=1)
                   for i, met in enumerate(rotmso_met)}     
    
    mcp = pd.DataFrame({'x': mcp[:,0],
           'y': mcp[:,1],
           'zenith': mcp[:,2],
           'azimuth':mcp[:,3],
           'sa':mcp[:,4],
           'qual':mcp[:,6]})
    
    pha = pd.DataFrame({'met':pha[:,0],
           'utc':pha[:,1],
           'scantype':pha[:,3],
           'step':pha[:,5],
           'epq':pha[:,6],
           'x':pha[:,7],
           'y':pha[:,8],
           'zenith':pha[:,9],
           'azimuth':pha[:,10],
           'mpq':pha[:,16],
           'weight':pha[:,17]})
    
    rates = pd.DataFrame({'met':scan[:,0],
             'utc':scan[:,1],
             'qual':scan[:,2],
             'scantype':scan[:,3]})
    

    
    H_pha = pha[(pha.mpq == 1.0)]
    
    if Na == True:
        
        H_pha = pha[((pha.mpq > 21) & (pha.mpq < 30))]
        
    

    
    
    def convert_to_mso_angles(pha,rm):
        
        pha[((pha.zenith==-9999.0)|(pha.azimuth==-9999.0))] = np.nan
        
        # Compute unit FIPS vectors
        sin_theta = np.sin(np.deg2rad(pha.zenith))
        cos_theta = np.cos(np.deg2rad(pha.zenith))
        cos_phi = np.cos(np.deg2rad(pha.azimuth))
        sin_phi = np.sin(np.deg2rad(pha.azimuth))
        
        v_unit_FIPS_vector = -1 * np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]).T
        
        # Initialize the output array
        
        v_mso_vec = np.empty((len(pha.met), 3))
        
        if len(v_mso_vec) != len(rm):
            breakpoint()
            
        for i in range(len(pha.met)):
            v = v_unit_FIPS_vector[i,:]
            
            mat = rm[i,:,:]
            
            v_mso_vec_1 = np.dot(mat,v)
            
            v_mso_vec = np.append(v_mso_vec,[v_mso_vec_1],axis=0)
                
        theta_mso = np.arccos(v_mso_vec[:,2]/np.linalg.norm(v_mso_vec[:],axis=1))
        
        phi_mso = np.arctan2(v_mso_vec[:,1],v_mso_vec[:,0])
        
        theta_mso = np.rad2deg(theta_mso)
        
        phi_mso = np.rad2deg(phi_mso)
        
        
        for i in range(len(phi_mso)):
            
            if phi_mso[i] < 0 :
                
                
                phi_mso[i] = phi_mso[i]+360
            
            
        
        return theta_mso, phi_mso
    
    def convert_to_mso_angles_time(zenith,azimuth,t):
        # Compute unit FIPS vectors
        sin_theta = np.sin(np.deg2rad(zenith))
        cos_theta = np.cos(np.deg2rad(zenith))
        cos_phi = np.cos(np.deg2rad(azimuth))
        sin_phi = np.sin(np.deg2rad(azimuth))
        
        v_unit_FIPS_vector = (-1) * np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]).T
        
        # Initialize the output array
        
        #v_mso_vec = np.empty((1, 3))
        # Compute the transformed vectors
        
        rot_matrix = rotmatrix_full[np.where(rotmso_met == t)[0][0],:]
        
        v_mso_vec = rot_matrix.dot(v_unit_FIPS_vector)   
        
        theta_mso = np.arccos(v_mso_vec[2]/np.linalg.norm(v_mso_vec))
        
        phi_mso = np.arctan2(v_mso_vec[1],v_mso_vec[0])
        
        theta_mso = np.rad2deg(theta_mso)
        
        phi_mso = np.rad2deg(phi_mso)
        
        
        if phi_mso < 0:
            
            phi_mso = phi_mso+360
            
            
        
        return theta_mso, phi_mso
        
    
    min_pha_time = np.min(H_pha.met)
    
    max_pha_time = np.max(H_pha.met)
    
    def MET_to_dt(df_dates):
        
        cutoff=200411609
        
        #cutoff=cutoff.timestamp()
        
        if df_dates[-1] > cutoff:
            
            print('Normal Cutoff')
            datetime_MET=convert_to_date_s('2004-08-03 06:55:32')
            
        else: 
            
            print('Weird Cutoff!')
            
            datetime_MET=convert_to_date_s('2013-01-08 20:13:20')
            
            #datetime_MET=convert_to_date_s('2004-08-03 06:55:32')

        df_dates=df_dates+datetime_MET.timestamp()
        
        df_datetime=[convert_to_date_2(d) for d in df_dates]
        
        df_datetime = np.array([convert_to_datetime(d) for d in df_datetime])
        
        return df_datetime
    
    df_datetime = MET_to_dt(H_pha.met.to_numpy())
    
   #gd = np.where((df_datetime > convert_to_datetime('2012-05-26 16:00:00')) & (df_datetime < convert_to_datetime('2012-05-26 16:19:59')))[0]
    
    gd = np.where((df_datetime > convert_to_datetime(trange[0])) & (df_datetime < convert_to_datetime(trange[1])))[0]
    
    #gd = np.where((df_datetime >= convert_to_datetime('2013-07-19 08:26:20')) & (df_datetime <= convert_to_datetime('2013-07-19 08:26:50')))[0]
    
    min_pha_time = H_pha.met.iloc[gd[0]]
    max_pha_time = H_pha.met.iloc[gd[-1]]
    
    
    
    min_index = np.where(H_pha.met == min_pha_time)
    
    max_index = np.where(H_pha.met == max_pha_time)
    
    rot_arr_ind = np.where((rotmso_met > min_pha_time) & (rotmso_met < max_pha_time))

    rot_matrix = rotmatrix_full[rot_arr_ind,:]
    
    time_arr = rotmso_met[rot_arr_ind]
    
    N = len(time_arr)
    
    flux_tot_arr = np.zeros((18,36))
    
    viewtime_tot_arr = np.zeros((18,36))
    
    MSO_zenith_low_bin_edges = np.linspace(0,170,num = 18)
    
    MSO_azimuth_low_bin_edges = np.linspace(0,350,num = 36)
    
    rot_matrix = rot_matrix[0,:,:,:]
    
    for n in range(N):
        t_n = time_arr[n]
        
        flux_arr_SA_wtd = np.zeros((18,36))
        
        time_ind = np.where(H_pha.met == t_n)[0]

        scan_type = H_pha.scantype.iloc[time_ind]

        
        M = len(time_ind)
        
        v_table = vrange
        
        delta_epq = 0.06
        
        v_low = np.sqrt(2*erange*(1-delta_epq/2)*1000.0/(amu2kg/(q*e2c)))
        
        v_high = np.sqrt(2*erange*(1+delta_epq/2)*1000.0/(amu2kg/(q*e2c)))
        
        v_high[1:] = v_low[0:-1]
        
        dv = (v_high - v_low)
        
        
        #theta_mso,phi_mso = convert_to_mso_angles(H_pha.iloc[time_ind])
        
        
        
        for m in range(M):
            
            v = np.sqrt(2*H_pha.epq.iloc[time_ind[m]]*1000.0/ ( H_pha.mpq.iloc[time_ind[m]] *amu2kg/e2c))
            
            epq_bin = np.where(erange == H_pha.epq.iloc[time_ind[m]])[0][0]
            
            flux = H_pha.weight.iloc[time_ind[m]]*v**3*dv[epq_bin]
            

            
            if H_pha.zenith.iloc[time_ind[m]] != -9999.0:
                
            
                theta_MSO, phi_MSO = convert_to_mso_angles_time(H_pha.zenith.iloc[time_ind[m]],\
                                                            H_pha.azimuth.iloc[time_ind[m]],\
                                                                H_pha.met.iloc[time_ind[m]])
            else:
                
                theta_MSO = np.nan
                
            if ~np.isnan(theta_MSO):
                
                MSO_zenith_ind = np.where(MSO_zenith_low_bin_edges == np.floor(theta_MSO/10)*10)[0][0]
                    
                
    
                MSO_azimuth_ind = np.where(MSO_azimuth_low_bin_edges == np.floor(phi_MSO/10)*10)[0][0]

                try:
                    
                    pixel_ind = np.where((mcp.x == H_pha.x.iloc[time_ind[m]]) & (mcp.y == H_pha.y.iloc[time_ind[m]]))[0][0]
                    flux_arr_SA_wtd[MSO_zenith_ind,MSO_azimuth_ind] = flux_arr_SA_wtd[MSO_zenith_ind,MSO_azimuth_ind] + \
                        flux*mcp.sa.iloc[pixel_ind]
                except:
                    breakpoint()
                    flux_arr_SA_wtd[MSO_zenith_ind,MSO_azimuth_ind] = flux_arr_SA_wtd[MSO_zenith_ind,MSO_azimuth_ind]
                
                

                
           
        #breakpoint()        
        valid_pixel_ind = np.where(mcp.qual == 0)[0]
        L = len(valid_pixel_ind)
        
        SA_arr = np.zeros((18,36))
        
        #theta_mso_mcp,phi_mso_mcp = convert_to_mso_angles(mcp.iloc[valid_pixel_ind])
        
        for l in range(L):
            theta_MSO,phi_MSO = convert_to_mso_angles_time(mcp.zenith.iloc[valid_pixel_ind[l]],mcp.azimuth.iloc[valid_pixel_ind[l]],t_n)
            
            
            MSO_zenith_ind = np.where(MSO_zenith_low_bin_edges == np.floor(theta_MSO/10)*10)
            MSO_azimuth_ind = np.where(MSO_azimuth_low_bin_edges == np.floor(phi_MSO/10)*10)
            
            SA_arr[MSO_zenith_ind,MSO_azimuth_ind] = SA_arr[MSO_zenith_ind,MSO_azimuth_ind] + \
                mcp.sa.iloc[valid_pixel_ind[l]]
                
        flux_arr = flux_arr_SA_wtd / SA_arr
        
        zero_SA_ind = np.where(SA_arr == 0)
        
        flux_arr[zero_SA_ind] = 0 
        
        accum_time = 50/1000.
        
        flux_tot_arr = flux_tot_arr+flux_arr*accum_time
        
        nonzero_SA_ind = np.where(SA_arr>0)
        
        viewtime_tot_arr[nonzero_SA_ind] = viewtime_tot_arr[nonzero_SA_ind]+accum_time
        
    final_flux = flux_tot_arr/viewtime_tot_arr
    
    #final_flux = np.transpose(final_flux
    
    ax = plot_on_spherical_mesh(final_flux,fr)
    
    return (ax,final_flux)
        
def plot_on_spherical_mesh(flux_data,frange):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    
    plasma_cmap = plt.get_cmap('plasma')

    # Create a new colormap that has black for NaN values
    colors = plasma_cmap(np.arange(plasma_cmap.N))
    colors[:1, :] = 0.5  # Make the first color black
    new_cmap = ListedColormap(colors)
    # Avoid log(0) by setting a minimum flux value (epsilon)
    #epsilon = np.min(flux_data[(flux_data>0)])-np.min(flux_data[(flux_data>0)])/2
    flux_data[(flux_data ==0)] = frange[0]
    
    # Define the spherical coordinates
    theta = np.linspace(0, np.pi, 18)  # 0 to 180 degrees in radians (latitude)
    phi = np.linspace(0, 2 * np.pi, 36)  # 0 to 360 degrees in radians (longitude)
    phi, theta = np.meshgrid(phi, theta)  # Note the order here
    
    
    b_test = np.array([-28,0,20])
    
    
    theta_test = np.arccos(b_test[2]/np.linalg.norm(b_test))
    
    phi_test = np.arctan2(b_test[1],b_test[0])
    
    
    
    # Plot the data using pcolormesh
    fig, ax = plt.subplots(subplot_kw={'projection': 'mollweide'})
    
    
    #ax.scatter(np.pi-phi_test,np.pi/2-theta_test,color='red',s=20)
    pcm = ax.pcolormesh(np.pi-phi, np.pi/2-theta, flux_data, norm=LogNorm(vmin = frange[0],vmax=frange[1]), cmap=new_cmap, shading='auto')
    
    # Add a color bar which maps values to colors
    cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('H+ Flux (m$^{-2}$s$^{-1}$)')
    
    # Customize the plot
    #ax.set_xlabel('Longitude')
    #ax.set_ylabel('Latitude')
    #ax.set_title('Flux Map')
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Add custom labels
    # Equator and meridian labels

    # ax.text(np.pi / 2,0, '+$Y_{MSO}$', horizontalalignment='center', verticalalignment='center', fontsize=12)
    # ax.text(-np.pi / 2,0, '-$Y_{MSO}$', horizontalalignment='center', verticalalignment='center', fontsize=12)
    # ax.text(-np.pi, 0, '+$X_{MSO}$', horizontalalignment='center', verticalalignment='center', fontsize=12)
    # ax.text(np.pi, 0, '+$X_{MSO}$', horizontalalignment='center', verticalalignment='center', fontsize=12)
    
    # ax.text(0,0,'-$X_{MSO}$',horizontalalignment='center', verticalalignment='center', fontsize=12)
    
    # # Pole labels
    # ax.text(0, np.pi / 2, '+$Z_{MSO}$', horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
    # ax.text(0, -np.pi / 2, '-$Z_{MSO}$', horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
    
    def custom_grid(ax):
        
        theta = np.linspace(0, np.pi, 18)
        
        phi = np.linspace(0,2*np.pi, 36)
        
        offsets_1 = [0,np.pi/2,-np.pi/2,np.pi/4,-np.pi/4,3*np.pi/4,-3*np.pi/4]
        
        offsets_2 = [0,np.pi/4,-np.pi/4]
        
        for i in offsets_1:
            
            ax.plot(np.zeros(len(theta))+i,np.pi/2-theta,color='gray',alpha=.8,linewidth=.7)
            
        
        for i in offsets_2:
            
            ax.plot(np.pi-phi,np.zeros(len(phi))+i,color='gray',alpha=.8,linewidth=.7)
            
    custom_grid(ax)
    
        
    
    #ax.grid(True)
    
    return ax

def format_ticks(ax,fs):
    
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.tick_params(axis='both', which='major', length=8)
    ax.tick_params(axis='both', which='minor', length=4) 
    
    ax.tick_params(labelsize=fs-5)
    
def get_orbit_num(event):
        for i in range(13):
            
            df = df = pd.read_pickle('ICME_Event_'+str(event)+'_'+str(i)+'.pkl')
    
        
            df_boundaries = pd.read_pickle('df_boundaries_philpott.pkl')
            
            mp_in_1 = df_boundaries[df_boundaries.Cross_Type=='mp_in_1'].time
            mp_in_2 = df_boundaries[df_boundaries.Cross_Type=='mp_in_2'].time
            
            o_in_1 = df_boundaries[df_boundaries.Cross_Type=='mp_in_2'].Orbit_number
            o_in_2 = df_boundaries[df_boundaries.Cross_Type=='mp_in_2'].Orbit_number
            
            
            mp_out_1 = df_boundaries[df_boundaries.Cross_Type=='mp_out_1'].time
            mp_out_2 = df_boundaries[df_boundaries.Cross_Type=='mp_out_2'].time
            
            o_out_1 = df_boundaries[df_boundaries.Cross_Type=='mp_out_1'].Orbit_number
            o_out_2 = df_boundaries[df_boundaries.Cross_Type=='mp_out_2'].Orbit_number
            
            mp_1 = np.sort(np.append(mp_in_1,mp_out_1))
            
            mp_2 = np.sort(np.append(mp_in_2,mp_out_2))
            
         
            o_1 = np.sort(np.append(o_in_1,o_out_1))
            
            o_2 = np.sort(np.append(o_in_2,o_out_2))
            
            mp_time = np.stack((mp_1,mp_2),axis=1)
            
            o = np.stack((o_1,o_2),axis=1)
            
            mp_time = mp_time.astype('datetime64[s]').astype('datetime64[us]').astype(object)
            
            time = df.time
            
            gd_mp = np.where((mp_time[:,0] > time.iloc[0]) & (mp_time[:,1] < time.iloc[-1]))[0]
            
            t_mp = mp_time[gd_mp,:]
            
            o_mp = o[gd_mp,:]
            
            print(o_mp[0,0])
            
            print(t_mp[1,1])
        
        
def generate_FIPS_flux_map_event(min_range,event_number, avg=False, Na = False):
    
    
    
    df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_21_'+str(event_number)+'.pkl')
    
    diff = df.time.iloc[-1]-df.time
    fr = [1E8,5E12]
    
    dates =np.array([d.total_seconds() for d in diff])/60
    
    dates = np.max(dates)-dates
    
    df['dates'] = dates
    
    df_oi = df[((df.dates > min_range[0]) & (df.dates < min_range[1]))]
    
    trange = [convert_datetime_to_string(df_oi.time.iloc[0]),convert_datetime_to_string(df_oi.time.iloc[-1])]
    
    
    print(trange)
    if Na == False:
        ax, flux_data = generate_FIPS_flux_map(trange,fr)
        
    if Na == True:
        ax, flux_data = generate_FIPS_flux_map(trange,fr,Na=True)
   # breakpoint()
    ax.set_title('Event number '+str(event_number))
    
    ax.set_title('ICME Event')
    
    ax = create_FIPS_plot(trange)
    
    ax[0].set_title('Event number '+str(event_number))
    
    if avg == True:
        flux_data_total = np.zeros((0,18,36))
        for i in range(11):
            
            if i !=6:
                df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_'+str(21)+'_'+str(i)+'.pkl')
                
                diff = df.time.iloc[-1]-df.time
                
                
                dates =np.array([d.total_seconds() for d in diff])/60
                
                dates = np.max(dates)-dates
                
                df['dates'] = dates
                
                df_oi = df[((df.dates > min_range[0]) & (df.dates < min_range[1]))]
                
                trange = [convert_datetime_to_string(df_oi.time.iloc[0]),convert_datetime_to_string(df_oi.time.iloc[-1])]
                
                
                print(trange)
                
                if Na == False:
                    
                    ax, flux_data = generate_FIPS_flux_map(trange,fr)
                    
                if Na == True:
                    
                    ax, flux_data = generate_FIPS_flux_map(trange,fr,Na = True)
                    
                
                flux_data_total = np.append(flux_data_total,[flux_data],axis=0)
                
                ax.set_title('Event number '+str(i))
                
                
        flux_data = np.nanmean(flux_data_total,axis=0)
        
        ax = plot_on_spherical_mesh(flux_data, fr)
        
        ax.set_title('Surrounding Orbits')
        
        
            

        
        
    
    
    
    
    
def streamQuiver(ax,sp,*args,spacing=None,n=5,**kwargs):
    """ Plot arrows from streamplot data  
    The number of arrows per streamline is controlled either by `spacing` or by `n`.
    See `lines_to_arrows`.
    """
    def curve_coord(line=None):
        """ return curvilinear coordinate """
        x=line[:,0]
        y=line[:,1]
        s     = np.zeros(x.shape)
        s[1:] = np.sqrt((x[1:]-x[0:-1])**2+ (y[1:]-y[0:-1])**2)
        s     = np.cumsum(s)                                  
        return s

    def curve_extract(line,spacing,offset=None):
        """ Extract points at equidistant space along a curve"""
        x=line[:,0]
        y=line[:,1]
        if offset is None:
            offset=spacing/2
        # Computing curvilinear length
        s = curve_coord(line)
        offset=np.mod(offset,s[-1]) # making sure we always get one point
        # New (equidistant) curvilinear coordinate
        sExtract=np.arange(offset,s[-1],spacing)
        # Interpolating based on new curvilinear coordinate
        xx=np.interp(sExtract,s,x);
        yy=np.interp(sExtract,s,y);
        return np.array([xx,yy]).T

    def seg_to_lines(seg):
        """ Convert a list of segments to a list of lines """ 
        def extract_continuous(i):
            x=[]
            y=[]
            # Special case, we have only 1 segment remaining:
            if i==len(seg)-1:
                x.append(seg[i][0,0])
                y.append(seg[i][0,1])
                x.append(seg[i][1,0])
                y.append(seg[i][1,1])
                return i,x,y
            # Looping on continuous segment
            while i<len(seg)-1:
                # Adding our start point
                x.append(seg[i][0,0])
                y.append(seg[i][0,1])
                # Checking whether next segment continues our line
                Continuous= all(seg[i][1,:]==seg[i+1][0,:])
                if not Continuous:
                    # We add our end point then
                    x.append(seg[i][1,0])
                    y.append(seg[i][1,1])
                    break
                elif i==len(seg)-2:
                    # we add the last segment
                    x.append(seg[i+1][0,0])
                    y.append(seg[i+1][0,1])
                    x.append(seg[i+1][1,0])
                    y.append(seg[i+1][1,1])
                i=i+1
            return i,x,y
        lines=[]
        i=0
        while i<len(seg):
            iEnd,x,y=extract_continuous(i)
            lines.append(np.array( [x,y] ).T)
            i=iEnd+1
        return lines

    def lines_to_arrows(lines,n=5,spacing=None,normalize=True):
        """ Extract "streamlines" arrows from a set of lines 
        Either: `n` arrows per line
            or an arrow every `spacing` distance
        If `normalize` is true, the arrows have a unit length
        """
        if spacing is None:
            # if n is provided we estimate the spacing based on each curve lenght)
            spacing = [ curve_coord(l)[-1]/n for l in lines]
        try:
            len(spacing)
        except:
            spacing=[spacing]*len(lines)

        lines_s=[curve_extract(l,spacing=sp,offset=sp/2)         for l,sp in zip(lines,spacing)]
        lines_e=[curve_extract(l,spacing=sp,offset=sp/2+0.01*sp) for l,sp in zip(lines,spacing)]
        arrow_x  = [l[i,0] for l in lines_s for i in range(len(l))]
        arrow_y  = [l[i,1] for l in lines_s for i in range(len(l))]
        arrow_dx = [le[i,0]-ls[i,0] for ls,le in zip(lines_s,lines_e) for i in range(len(ls))]
        arrow_dy = [le[i,1]-ls[i,1] for ls,le in zip(lines_s,lines_e) for i in range(len(ls))]

        if normalize:
            dn = [ np.sqrt(ddx**2 + ddy**2) for ddx,ddy in zip(arrow_dx,arrow_dy)]
            arrow_dx = [ddx/ddn for ddx,ddn in zip(arrow_dx,dn)] 
            arrow_dy = [ddy/ddn for ddy,ddn in zip(arrow_dy,dn)] 
        return  arrow_x,arrow_y,arrow_dx,arrow_dy 

    # --- Main body of streamQuiver
    # Extracting lines
    seg   = sp.lines.get_segments() # list of (2, 2) numpy arrays
    lines = seg_to_lines(seg)       # list of (N,2) numpy arrays
    # Convert lines to arrows
    ar_x, ar_y, ar_dx, ar_dy = lines_to_arrows(lines,spacing=spacing,n=n)
    # Plot arrows
    qv=ax.quiver(ar_x, ar_y, ar_dx, ar_dy, *args, angles='xy', **kwargs,headwidth=6,headlength=4,width =.009)
    return qv    
def angle_tester(event):
    
    filename = 'ICME_Dictionary_'+str(21)+'_'+str(6)+'.pkl'
    
    icme_21 = pd.read_pickle(filename)
    
    B_IMF = icme_21['B_IMF']
    
    df = icme_21['df']
    
    # diff = df.time.iloc[-1]-df.time
    
    # dates = np.flip(np.array([d.total_seconds() for d in diff])/60)
    
    # df['dates']=dates
    
    dates_aw = np.load('dates_21_aw.npy')
    
    df = df[((df.dates>dates_aw[0]) & (df.dates<dates_aw[-1]))]
    
    Ma = icme_21['Ma'][0]
    
    other = -(180-np.rad2deg(np.arctan2(B_IMF[0],B_IMF[2])))
    

    phi = np.arctan(Ma)
    
    theta = np.radians(other)
    
    #tilt_angle = (phi-theta)*-1*factor

    tilt_angle = -(-phi-theta)
    
    print(tilt_angle*180/np.pi)

    
    mag_angle = np.arctan(df.magx_diff/df.magz_diff)*180/np.pi
    
    
    fig,ax = plt.subplots(1)
    
    ax.plot(df.dates,mag_angle)
    
    fig,ax = plt.subplots(2)
    
    for i in range(11):
        
        filename = 'ICME_Dictionary_'+str(21)+'_'+str(i)+'.pkl'
        
        icme_21 = pd.read_pickle(filename)
        
        df = icme_21['df']
        
        diff = df.time.iloc[-1]-df.time
        
        dates = np.flip(np.array([d.total_seconds() for d in diff])/60)
        
        df['dates']=dates

        dates_aw = np.load('dates_21_aw.npy')
        
        df = df[((df.dates>dates_aw[0]) & (df.dates<dates_aw[-1]))]
        magamp = df.magamp.to_numpy()
    
        magamp = downsample(300,magamp)
    
        dates = df.dates.to_numpy()
    
        dates = downsample(300,dates)

   


        ax[0].plot(dates,magamp,color='gray',alpha=.4)   
    
        dx = dates[1]-dates[0]
    
        grad = np.gradient(magamp,dx)
    
        ax[1].plot(dates,grad,color='gray',alpha = .4)
        
        if i ==6:
            ax[0].plot(dates,magamp,color='darkorange',alpha=1)  
            ax[1].plot(dates,grad,color='darkorange',alpha=1)  
            
    
    
def estimate_rho_and_u():
    
    icme = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Dictionary_21_6.pkl')
    
    psw = icme['psw'][0]*1E-9
    
    B = icme['B_SW'][0]*1E-9
    
    B_IMF = icme['B_IMF']*1E-9
    
    mp = 1.67E-27
    
    munaught = 4*np.pi * 1E-7
    
    n = np.linspace(10,155,num=1000)
    
    v = np.linspace(100,800,num=1000)

    N,V = np.meshgrid(n,v)
    
    rho = N*100**3*mp
    
    v_m = V*1000
    
    Ma = v_m/(B/np.sqrt(munaught*rho))
    
    fig,ax = plt.subplots(1)
    
    # Create the pseudocolor plot
    pcm = ax.pcolormesh(N, V, Ma, shading='auto')
    
    # Add contour lines
    contour = ax.contour(N, V, Ma, colors='white')
    
    # Add labels to the contour lines
    ax.clabel(contour, inline=True, fontsize=15)
    
    # Add a colorbar
    fig.colorbar(pcm, ax=ax, label='Alfvén Mach number ($M_A$)')
    
    # Label axes
    ax.set_xlabel('Number Density (cm$^{-3}$)')
    ax.set_ylabel('Velocity (km/s)')
    
    red_contour = ax.contour(N, V, Ma, levels=[1.5], colors='red')
    ax.clabel(red_contour, inline=True, fontsize=15, fmt='%.1f')

    #n = np.array([61,88,40,68,51,69,29,155,108,92])
    #v = np.array([471,368,435,713,504,508,451,282,398,607])
    
    n_mean = np.array([5.8*(1/.46**2)])
    n_median = np.array([6.9*(1/.46**2)])
    
    v_mean = np.array([399])
    v_median = np.array([402])
    
    ax.scatter(n_mean,v_mean,c='orange',s=20,label = 'Scaled Mean Properties')
    ax.scatter(n_median,v_median,c='red',s=20,label = 'Scaled Median Properties')
    
    ax.legend(fontsize=15)
    
    ax.set_title('$M_A$ Contours at $|B|$ = 63 nT')
    
    v_sw = 662E3
    
    #B/np.sqrt(munaught*rho) = v_sw/1.5
    
    n = ((1.5*B/v_sw)**2/munaught)/mp
    
    rho = n*mp
    
    v_A = B/np.sqrt(munaught*rho)
    
    v_A_x = B_IMF[0]/np.sqrt(munaught*rho)
    
    v_A_z = B_IMF[2]/np.sqrt(munaught*rho)
    
    print(v_sw/v_A)
    
    n_cm = n/(100**3)
    
    print(n_cm)
    
    print(n*mp*v_sw**2)
    
    print(psw)
    
    print('Alfvén Conductance:')
    print(1/(munaught*v_A))
    
    print('Ridley AC: '+str(1/(munaught*v_A*np.sqrt(1+icme['Ma'][0]**2))))

        
    kb = 1.380649E-23
    
    T = 8E4
    
    pT = n*kb*T
    
    beta = pT/(B**2/(2*munaught))
    
    print('Beta = '+str(beta))
    
    # Nominal Conditions at .42 AU
    
    B = 32.6E-9
    
    v = 406.2E3
    
    n = 44.2*(100**3)
    
    va = B/np.sqrt(munaught*mp*n)
    
    M_n = v/va
    
    print('M Nominal: '+str(M_n))
    
    
    c_1 = [-1*v_sw+v_A_x,0,v_A_z]
    
    c_p = c_1 / np.linalg.norm(c_1)
    
    c_2 = [-1*v_sw - v_A_x,0,-v_A_z]
    
    c_n = c_2 / np.linalg.norm(c_2)
    
    print('C+ '+str(c_p))

    print('C- '+str(c_n))    
    
    
    M_a = 1.5
    
    cme = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Dictionary_21_6.pkl')
    
    psw = icme['psw'][0]*1E-9
    
    B = icme['B_SW'][0]*1E-9
    
    B_IMF = icme['B_IMF']*1E-9
    
    v = [1,0,0]
    
    
    
    
    

    
    
    plt.show()
    
def plot_Helios(event):
    
    dates = ['1975-03-04','1975-03-19','1976-03-27','1978-05-12','1978-10-18','1979-05-28','1976-03-31','1977-10-26','1978-04-24','1979-05-09']
    date_string = dates[0]
    doy = get_day_of_year(dates[0])
    
    doy=get_day_of_year(date_string)
    month=date_string[5:7]
    year=date_string[2:4]
    
    year_full=date_string[0:4]
    if doy < 10: doy_s='00'+str(doy)
        
    elif (doy<100) & (doy>=10):doy_s='0'+str(doy)
        
    else: doy_s=str(doy)
    
    #filename = '/Users/bowersch/Desktop/Helios Data/'+year_full+'/h1'+year_full[2:]+doy_s+'.asc'
    
    filename = '/Users/bowersch/Desktop/Helios Data/he1mga'+year_full[2:]+'.asc'
    
    
    d = np.genfromtxt(filename)
    
    doy = d[:,2]
    hour = d[:,3]
    
    
    start = convert_to_datetime('1975-01-01 00:00:00')
    
    dates = np.array([])
    for i in range(len(d)):
        dates = np.append(dates,start+datetime.timedelta(days = doy[i]-1,hours=hour[i]))
        
    fig,ax = plt.subplots(5,sharex=True)
    
    ax[0].plot(dates,d[:,4],color='blue')
    ax[1].plot(dates,d[:,5],color='green')
    ax[2].plot(dates,d[:,6],color='red')
    ax[3].plot(dates,d[:,7],color='black')
    ax[4].plot(dates,d[:,2])
    ax[4].axhline(y=64,color='red')
    for x in ax:
        
        x.axhline(y=0,linestyle='--',color='black')

def check_kappa_fits():
    
    df_k=pd.read_csv('/Users/bowersch/Desktop/MESSENGER Data/FIPSProtonClass.dat',delim_whitespace=True,index_col=False)
    
    datetimes = np.array([convert_to_datetime_K(d) for d in df_k.UT.to_numpy()])
    
    start_time = convert_to_datetime('2011-12-30 22:00:00')
    
    end_time = convert_to_datetime('2011-12-31 04:08:00')
    
    gd = np.where((datetimes > start_time ) & (datetimes <= end_time))[0]
    
    fig,ax = plt.subplots(1)
    
    ax.scatter(datetimes[gd],df_k.Density.iloc[gd])
    
    ax.set_yscale('log')

    

def cme_onset_time():
    
    start_cme  = convert_to_datetime('2011-12-30 18:14:00')
    
    #cme_onset = convert_to_datetime('2011-12-29 16:24:00')
    
    cme_onset = convert_to_datetime('2011-12-29 15:52:00')
    
    t = start_cme-cme_onset
    
    t = t.total_seconds()
    
    v = 662
    
    distance = 0.42 * 1.496E8
    
    print(v*t)
    
    print(distance)
    
    print(distance/t)

        
# Event 34 mp_time = [259,260], up_range = [15,45], aw_range = [185,195], cusp_range=[255,256], orbit_num = 5
    
# Event 21 mp_time = [340,341], up_range = [277,307], aw_range = [100,200], cusp_range = [338,340], orbit_num = 3
    
    
def check_ma_dewey_estimate():
    
    n_d = .035 * (100**3)
    
    T_d = 2.25E6
    
    v_d = np.array([-159,-90,0])*1000
    
    B_d = np.array([15,20,-115])*1E-9
    
    B_u = np.array([19,0,-60])*1E-9
    
    # Co-Planarity Theorem
    
    n = np.cross(B_u-B_d,np.cross(B_u,B_d))
    
    
    
    n = n/np.linalg.norm(n)
    
    un_d = np.dot(v_d,n)
    
    ut_d = np.linalg.norm(v_d)-un_d
    
    print('un_d = '+str(un_d))
    
    print('ut_d = '+str(ut_d))
    
    Bn_d = np.dot(B_d,n)
    
    Bt_d = np.linalg.norm(B_d)-Bn_d
    
    Bn_u = np.dot(B_u,n)
    
    Bt_u = np.linalg.norm(B_u)-Bn_u
    
    mp = 1.67E-27
    
    munaught=4*np.pi*1E-7
    
    rho_d = n_d*mp
    
    d1 = rho_d*un_d
    
    d2 = un_d*Bt_d-Bn_d*ut_d
    
    d3 = rho_d*un_d*ut_d - Bn_d/munaught*Bt_d+Bn_u/munaught*Bt_u
    
    from scipy.optimize import fsolve

    # Define the function representing the system of equations
    def equations(vars):
        x, y, z = vars
        return [
            x * y - d1,
            a * y - b * z - d2,
            x * y * z - d3
        ]
    
    # Set the parameters and constants
    a = Bt_u  
    b = Bn_u  
    
    # Initial guess for x, y, z
    initial_guess = (rho_d, un_d, ut_d)
    
    # Solve the system of equations
    solution = fsolve(equations, initial_guess)
    x, y, z = solution
    
    un_u = y/1000
    
    ut_u = z/1000
    
    n_u = x/mp/(100**3)
      
    print(f"The solution is: n_u = {n_u}, un_u = {un_u}, ut_u = {ut_u}")
    
    mp = 1.67E-27
    
    munaught = 4*np.pi * 1E-7


    
    Ma = 2000*1000/(np.linalg.norm(B_u)/np.sqrt(munaught*rho_d))
    
    
    #Assume velocity of solar wind
    
    u_u = np.array([-662,0,0])*1000
    
    un_u = np.dot(u_u,n)
    
    ut_u = np.linalg.norm(u_u)-un_u
    
    rho_u = rho_d*un_d/un_u
    
    Ma = np.linalg.norm(u_u)/(np.linalg.norm(B_u)/np.sqrt(munaught*rho_u))
    
    
    
    
    
    
    
def calculate_K_coeff():
    
    gamma = 5/3.
    
    M = 1.5
    
    c1 = ((gamma+1)/2)**((gamma+1)/(gamma-1))
    
    c2 = 1/(gamma*(gamma-(gamma-1)/(2*M*M))**(1/(gamma-1)))
    
    K = c1*c2
    
    print(K)
    
def estimate_MF():
    
    va = 440*1000.
    
    mp = 1.67E-27
    
    munaught = 4*np.pi * 1E-7
    
    T  = 8.7E4
    
    gamma = 5/3
    
    kappa = 1.380649E-23
    
    Cs = np.sqrt(gamma*kappa*(2*T)/mp)
    
    print(Cs/1000)
    
    print(660*1000./np.sqrt(Cs**2+va**2))
    
    
    
    
    
    
    
    
    
