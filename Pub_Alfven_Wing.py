#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:13:42 2024

@author: bowersch
"""

# Path for ICME List from Winslow et al., 2017
ICME_List_file='/Users/bowersch/Desktop/DIAS/Work/My Papers/Alfven_Wing/ICME_MESSENGER_Winslow.xlsx'

import pandas as pd
import datetime
import spiceypy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,AutoMinorLocator

# Path for spice kernel set for MESSENGER
meta_kernel = '/Users/bowersch/Desktop/MESSENGER Data/msgr_spice_kernels/MESSENGER_meta_kernel.txt'

philpott_boundary_file = '/Users/bowersch/Desktop/Python_Code/Alfven_Wing/df_boundaries_philpott.pkl'

path_to_ICME_event_files = '/Users/bowersch/Desktop/Python_Code/PUB_Alfven_Wing/Pub_Bowers_Mercury_Alfven_Wing/ICME_Event_pkl_files/'

path_to_folder = '/Users/bowersch/Desktop/Python_Code/PUB_Alfven_Wing/Pub_Bowers_Mercury_Alfven_Wing/'

path_to_flux_map_folder = '/Users/bowersch/Desktop/Python_Code/PUB_Alfven_Wing/Pub_Bowers_Mercury_Alfven_Wing/ICME_Event_Flux_Maps/'

from scipy.interpolate import griddata

import matplotlib.colors as colors

# Load in spice kernels
if spiceypy.ktotal( 'ALL' )==0:
    
    spiceypy.furnsh(meta_kernel)


from os import sys
import matplotlib.colors as colors
#from lobe_id import shade_in_time_series

#

from General_Functions_Alfven_Wing import check_for_mp_bs_WS,\
    convert_to_datetime,convert_to_date_2,plot_mp_and_bs,\
    plot_MESSENGER_trange_cyl,convert_to_date_2_utc,get_mercury_distance_to_sun,\
        convert_datetime_to_string,plot_MESSENGER_trange_3ax,get_aberration_angle,\
            get_day_of_year,convert_to_date_s, read_in_Weijie_files,\
                convert_to_datetime_K, load_MESSENGER_into_tplot,\
                    append_for_run_over,append_for_run_over_FIPS,format_ticks,\
                        distance_point_to_curve, calculate_tangent_vector
                        
# Some constants
        
# mass of proton                
mp = 1.67E-27

# permitivity of free space
munaught = 4*np.pi * 1E-7

R_M = 2440

def multi_orbit_plot(ii,r1=False,r2=False,r3=False,r4=False,ranges=False,conduct=False,ganymede=False):
    
    '''FIGURE 2 in the Manuscript
    
    Example Usage: multi_orbit_plot(21)
    
    '''
    # Define alpha value for surrounding orbit data
    alpha_gray= .4
    
    # Define font size
    fs = 18
    
    
    def make_3d_mercury():
        
        ''' Create 3D Mercury plot'''
        
        # Define fontsize for plots
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
    
        
    # Create cylindrical projection
    ax_c=plot_MESSENGER_trange_cyl(['2011-09-20 00:00:00','2011-09-20 15:00:00'],plot=False)
    
    def make_Ma_plot():
        '''Make plot for estimates of MA for both the ICME orbit and the surrounding orbits'''
        
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
        ''' Add information onto the Mach number plots for each parameter'''
        
        # Which plot are we adding to?
        
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
        
        
        
        handles, labels = ax[axis].get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        
        # Plot legend with only unique labels
        
        # Uncomment if you want legend
        #ax[axis].legend(unique_labels.values(), unique_labels.keys(),loc='upper right')
        
        return
    
    def add_onto_boundary_plot(ax,df,col,siz,alph,labe,philpott=True):
        ''' Add trajectory onto cylindrical plot for each of the surrounding orbits and the ICME orbit'''
        
        # Define boundary information from Philpott File
        if philpott == True:
            
            # Load in dataframe
            
            df_boundaries = pd.read_pickle(philpott_boundary_file)
            
            # What are the magnetopause crossings? Inbound and outbound
            mp_in_1 = df_boundaries[df_boundaries.Cross_Type=='mp_in_1'].time
            mp_in_2 = df_boundaries[df_boundaries.Cross_Type=='mp_in_2'].time
            
            mp_out_1 = df_boundaries[df_boundaries.Cross_Type=='mp_out_1'].time
            mp_out_2 = df_boundaries[df_boundaries.Cross_Type=='mp_out_2'].time
            
            mp_1 = np.sort(np.append(mp_in_1,mp_out_1))
            
            mp_2 = np.sort(np.append(mp_in_2,mp_out_2))
            
            mp_time = np.stack((mp_1,mp_2),axis=1)
            
            mp_time = mp_time.astype('datetime64[s]').astype('datetime64[us]').astype(object)
            
            # Find bow shock crossings
            bs_in_1 = df_boundaries[df_boundaries.Cross_Type=='bs_in_1'].time
            bs_in_2 = df_boundaries[df_boundaries.Cross_Type=='bs_in_2'].time
            
            bs_out_1 = df_boundaries[df_boundaries.Cross_Type=='bs_out_1'].time
            bs_out_2 = df_boundaries[df_boundaries.Cross_Type=='bs_out_2'].time

            
            bs_1 = np.sort(np.append(bs_in_1,bs_out_1))
            
            bs_2 = np.sort(np.append(bs_in_2,bs_out_2))
            
            bs_time = np.stack((bs_1,bs_2),axis=1)
            
            bs_time = bs_time.astype('datetime64[s]').astype('datetime64[us]').astype(object)
            
        # Find times for magnetopause and bow shock crossings within the orbit of interest
        time = df.time
        
        gd_mp = np.where((mp_time[:,0] > time.iloc[0]) & (mp_time[:,1] < time.iloc[-1]))[0]
        
        t_mp = mp_time[gd_mp,:]
        

        gd_bs = np.where((bs_time[:,0] > time.iloc[0]) & (bs_time[:,1] < time.iloc[-1]))[0]
        
        t_bs = bs_time[gd_bs,:]
        
        # Calculate positions of magnetopause crossings and bow shock crossings
        x_mp = np.zeros((0,2))
        
        r_mp = np.zeros((0,2))
        
        x_bs = np.zeros((0,2))
        
        r_bs = np.zeros((0,2))
        
        # For each of the magnetopause crossings
        for l in range(len(t_mp)):
            
            df_0 = df[(df.time > t_mp[l,0]) & (df.time < t_mp[l,0]+datetime.timedelta(seconds=1.5))]
        
            df_1 = df[(df.time > t_mp[l,1]) & (df.time < t_mp[l,1]+datetime.timedelta(seconds=1.5))]
            
            # Calculate average position of boundaries in the cylindrical coordinate system
            
            x_mp0 = np.mean(df_0.ephx)
            
            r_mp0 = np.mean(np.sqrt(df_0.ephy**2+df_0.ephz**2))
            
            x_mp1 = np.mean(df_1.ephx)
            
            r_mp1 = np.mean(np.sqrt(df_1.ephy**2+df_1.ephz**2))
            
            x_mp = np.append(x_mp,[[x_mp0,x_mp1]],axis=0)
            
            r_mp = np.append(r_mp, [[r_mp0,r_mp1]],axis=0)
        # For each of the bow shock crossigns    
        for l in range(len(t_bs)):
        
            df_0 = df[(df.time > t_bs[l,0]) & (df.time < t_bs[l,0]+datetime.timedelta(seconds=1.5))]
        
            df_1 = df[(df.time > t_bs[l,1]) & (df.time < t_bs[l,1]+datetime.timedelta(seconds=1.5))]
        
            # Calculate average position of boundaries in the cylindrical coordinate system
        
            x_bs0 = np.mean(df_0.ephx)
            
            r_bs0 = np.mean(np.sqrt(df_0.ephy**2+df_0.ephz**2))
            
            x_bs1 = np.mean(df_1.ephx)
            
            r_bs1 = np.mean(np.sqrt(df_1.ephy**2+df_1.ephz**2))
            
            x_bs = np.append(x_bs,[[x_bs0,x_bs1]],axis=0)
            
            r_bs = np.append(r_bs, [[r_bs0,r_bs1]],axis=0)
            
        
        r_mp_range = np.abs((r_mp[:,1]-r_mp[:,0])/2)
        
        r_mp = np.mean(r_mp,axis=1)
        
        x_mp_range = np.abs((x_mp[:,1]-x_mp[:,0])/2)
        
        x_mp = np.mean(x_mp,axis=1)
        
        if labe[0]=='I':
            # We are in the ICME orbit! So we plot this with the highest zorder
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
           # We are in the ICME orbit! So we plot this with the highest zorder
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
            
            '''Estimate the Mach number based on MAG data'''
            
            # Define Parameters for magnetopause curve
            Rss = 1.45
            alpha = 0.5
            
            # Define magnetopause curve
                
            phi2 = (np.linspace(0,2*np.pi,1000))

            rho=Rss*(2/(1+np.cos(phi2)))**(alpha)

            xmp=rho*np.cos(phi2)

            ymp=rho*np.sin(phi2)

            curve=np.transpose(np.vstack((xmp,ymp)))
            
            # Constants for calculations
            
            munaught=4*np.pi*1E-7
            
            # Create empty arrays to track the relevant variables
            
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
                
                # Use the inbound portion first 
                if p == 0:
                    
                    # Define the upstream time range to get solar wind parameters
                    
                    up_range = np.array([bs_crossing[0] - datetime.timedelta(minutes=30),bs_crossing[0]])
                    
                    # Define time range that defines the magnetospheric parameters
                    mp_range = np.array([mp_crossing[1], mp_crossing[1] + datetime.timedelta(minutes=1)])
                
                # Then the outbound portion
                if p == 1:
                    
                    # Define the upstream time range to get solar wind parameters
                    
                    up_range = np.array([bs_crossing[1], bs_crossing[1] + datetime.timedelta(minutes=30)])
                    
                    # Define time range that defines the magnetospheric parameters
                    
                    mp_range = np.array([mp_crossing[0] - datetime.timedelta(minutes=1), mp_crossing[0]])
                    
                
                
                magamp=df.magamp.to_numpy()
                
                mag=df[['magx','magy','magz']].to_numpy()
                
                eph=df[['ephx','ephy','ephz']].to_numpy()
                
                
                # What is the location of the boundary?
                ea_test = np.mean(eph[(df.time > mp_range[0]) & (df.time < mp_range[1]),:],axis=0)
                
                ea_std = np.std(eph[(df.time > mp_range[0]) & (df.time < mp_range[1]),:],axis=0)
                
                
                # Define dataframe for within the magnetopause
                
                df_MP=df[(df.time > mp_range[0]) & (df.time < mp_range[1])]
                
                # Only include the top 20% of magnetic field measurements
                percentile = np.percentile(df_MP.magamp.to_numpy(), 80)
                
                df_MP=df_MP[(df.magamp > percentile)]
                
                # What is the strength of the magnetic field near the magnetopause?
                
                B_MP=np.mean(df_MP.magamp.to_numpy())
                
                B_MP_std = np.std(df_MP.magamp.to_numpy())
                
                # What is the strength of the magnetic field upstream of the bow shock?
                B_SW = np.mean(magamp[(df.time > up_range[0]) & (df.time < up_range[1])])
                
                B_SW_std = np.std(magamp[(df.time > up_range[0]) & (df.time < up_range[1])])
                
                # Orientation of magnetic field upsteram of bow shock
                B_X = np.mean(mag[(df.time > up_range[0]) & (df.time < up_range[1]),0])
                B_Y = np.mean(mag[(df.time > up_range[0]) & (df.time < up_range[1]),1])
                B_Z = np.mean(mag[(df.time > up_range[0]) & (df.time < up_range[1]),2])
                
                B_X_std = np.std(mag[(df.time > up_range[0]) & (df.time < up_range[1]),0])
                B_Y_std = np.std(mag[(df.time > up_range[0]) & (df.time < up_range[1]),1])
                B_Z_std = np.std(mag[(df.time > up_range[0]) & (df.time < up_range[1]),2])
                
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
                
                # t = calculate_tangent_vector(curve, index)
                
                # Uncomment this if you want to calculate MA from only the tangential component
                # of the magnetic field:
                
                # mag_mp = np.array([np.mean(df_MP.magx),np.mean(df_MP.magy),np.mean(df_MP.magz)])
                
                # mag_mp_cyl = np.array([mag_mp[0],np.sqrt(mag_mp[1]**2+mag_mp[2]**2)])
                
                # B_MP = np.dot(mag_mp_cyl,t)
                
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
                
                # Estimate dynamic pressure of the solar wind
                
                psw1=(B_MP**2/(2*munaught) - B_SW**2/(2*munaught))*(.88*np.cos(phi)**2)**(-1)
               
                psw_max =  ((B_MP+B_MP_std)**2/(2*munaught) - (B_SW-B_SW_std)**2/(2*munaught))*(.88*np.cos(np.max([phi_1,phi_2]))**2)**(-1)
               
                psw_min =  ((B_MP-B_MP_std)**2/(2*munaught) - (B_SW+B_SW_std)**2/(2*munaught))*(.88*np.cos(np.min([phi_1,phi_2]))**2)**(-1) 
               
                
               
                psw_std = psw_max-psw1
                # If phi is within 60 deg of the solar wind flow direction, then calculate MA
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

        return Ma,psw1*1E9,B_SW*1E9,B_IMF,pmag*1E9,180-phi*180/np.pi,Ma_std,psw_std*1E9,B_SW_std*1E9,B_IMF_std,pmag_std*1E9,phi_std*180/np.pi

        
        
    ax_3d=make_3d_mercury()
        
    ax1,ax2,ax3=make_mercury()
    

    # Load in icme data
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
    
    def make_mag_and_eph_plot(df,ylims,xlims,count,event_number,orbit_number,ganymede=False):
        
        ''' Plot relevant data onto trajectory and MA estimate plots'''
        
        diff = df.time.iloc[-1]-df.time
        
        diff_a = df.time.iloc[-1]-date_a
        
        diff_e = df.time.iloc[-1]-date_e
        
        dates =np.array([d.total_seconds() for d in diff])/60
        
        dates = np.max(dates)-dates
        

        
        dates_a = diff_a.total_seconds()/60
        
        dates_e = diff_e.total_seconds()/60
                         

        
        CME_Colors=['darkorange','goldenrod','mediumpurple','mediumturquoise','brown','green']
        
        
        if (((df.time.iloc[0] < date_a) and (df.time.iloc[-1] > date_a)) or \
            ((df.time.iloc[0] < date_e) and (df.time.iloc[-1] > date_e))) or \
            ((df.time.iloc[0] > date_a) and (df.time.iloc[-1] < date_e)):
            
            gd=((df.time > date_a) & (df.time < date_e))    
             
            
            df_cme=df[gd]   
            
    
            
            if len(df_cme)> 0.7*len(df):
                df_cme[((dates[gd]>105) & (dates[gd]<107))]=np.nan
                
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

          

        else:
            add_eph_onto_traj(ax1,ax2,ax3,df.ephx,df.ephy,df.ephz,'gray',3,alpha_gray)
                
            add_eph_onto_traj(ax_3d,ax2,ax3,df.ephx,df.ephy,df.ephz,'gray',3,alpha_gray,d3=True)
            
            Ma, psw, B_SW, B_IMF, B_MP, phi, Ma_std, psw_std, B_SW_std, B_IMF_std, B_MP_std, phi_std =  add_onto_boundary_plot(ax_c,df,'gray',50,.7, 'Surrounding Orbits',philpott=True)
            
            add_onto_MA_plot(ax_ma,Ma,psw, Ma_std,psw_std,'gray',50,.7, 'Surrounding Orbits','$P_{SW}^{dyn}$ (nPa)')
            
            add_onto_MA_plot(ax_ma,Ma,B_SW, Ma_std, B_SW_std, 'gray',50,.7, 'Surrounding Orbits', '$B_{IMF}$ (nT)')
            
            add_onto_MA_plot(ax_ma,Ma,B_MP, Ma_std, B_MP_std, 'gray',50,.7, 'Surrounding Orbits', '$P_{M}^{mag}$ (nPa)')
            
            add_onto_MA_plot(ax_ma,Ma,phi, Ma_std, phi_std, 'gray',50,.7, 'Surrounding Orbits', '$\theta$ (deg)')


            
            
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
        
        # Save data to easily access later
        import pickle
        file_path='ICME_Dictionary_'+str(event_number)+'_'+str(orbit_number)+'.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(icme_event, file)

        
        
        return axes_eph,count
    
    
    # Full Data
    
    xlims=[0,700]
    
    
    ylims=[[-140,140],[-140,140],[-140,140],[0,300],[0,50],[-1,1]]
    
    count=0
    for i in range(11):
        
        # Load in orbit for  ICME event
        df = pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Event_'+str(ii)+'_'+str(i)+'.pkl')
        
        axes_eph,count = make_mag_and_eph_plot(df,ylims,xlims,count,ii,i,ganymede = False)
        
def mhd_visualizer():
    
    
    ''' Figure 4 in the manuscript'''
    
    # Load in result
    result_file = path_to_folder + 'MHDSim_MESSENGER_ICME_20111230_B_V_PlasmaDensity_XZ_run2.dat'
    
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
        
        # Make a panel colored by the variable of interest
        
        
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
            
            df = pd.read_pickle(path_to_ICME_event_files + 'ICME_Event_'+str(21)+'_'+str(6)+'.pkl')
            
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
    
    '''
    FIGURES 5, 6, and 9 in the manuscript
    
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
        
        df = pd.read_pickle(path_to_ICME_event_files+'ICME_Event_'+str(ii)+'_'+str(i)+'.pkl')
        df_nobs = pd.read_pickle(path_to_ICME_event_files+'ICME_Event_NOBS_'+str(ii)+'_'+str(i)+'.pkl')
        df_HF = pd.read_pickle(path_to_ICME_event_files+'ICME_Event_HF_'+str(ii)+'_'+str(i)+'.pkl')
        
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
    
        
    
    # Load in MHD simulation
    mhd_run = pd.read_csv(path_to_folder+'MHDSim_MESSENGER_ICME_20111230_MAG_Density_run2.dat',index_col=False)
    
    # Rename variables
    
    df = pd.read_pickle(path_to_ICME_event_files+'ICME_Event_'+str(ii)+'_'+str(6)+'.pkl')
    
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
        
    
    icme = pd.read_pickle(path_to_ICME_event_files+'ICME_Dictionary_21_6.pkl')

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

def plot_on_spherical_mesh(flux_data,frange):
    ''' Make spherical mesh plot for flux map data'''
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
        
def generate_FIPS_flux_map_event(event_number,tag, avg=False, Na = False):
    
    '''FIGURE 7 and 10 in the manuscript
    
    Figure 7: generate_FIPS_flux_map_event([310,335],6,'dayside',avg=True)
    
    Figure 10: generate_FIPS_flux_map_event([385,410],6,'nightside',avg=True)
    
    '''
    fr = [1E8,5E12]
    
    flux_data = np.load(path_to_flux_map_folder + 'ICME_Flux_Map_'+tag+'_event_'+str(event_number)+'.npy')
    
    ax = plot_on_spherical_mesh(flux_data, fr)
    
   # breakpoint()
    ax.set_title('Event number '+str(event_number))
    
    ax.set_title('ICME Event')
    
    
    if avg == True:
        flux_data_total = np.zeros((0,18,36))
        for i in range(11):
            
            if i !=6:
                
                flux_data = np.load(path_to_flux_map_folder + 'ICME_Flux_Map_'+tag+'_event_'+str(i)+'.npy')

                    
                flux_data_total = np.append(flux_data_total,[flux_data],axis=0)
                
                #ax.set_title('Event number '+str(i))
                
                
        flux_data = np.nanmean(flux_data_total,axis=0)
        
        ax = plot_on_spherical_mesh(flux_data, fr)
        
        ax.set_title('Surrounding Orbits')

def current_sheet_analysis(event,orbit):
    '''Figure 8 in the manuscript
    
    Figure 8: current_sheet_analysis(21,np.arange(11))
    
    '''
    
    fig,ax = plt.subplots(1)

    
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
        
        if R_squared > 0.95:
            return popt[1],z0_calc,popt[0]
        
        else:
            return np.nan
        
        # #print('MAE = '+str(MAE))
        # if R_squared > 0.95:
        #     L = np.append(L,popt[1])
            
        # else:
        #     L = np.append(L,np.nan)
        

    L_all = np.array([])
    z0_all = np.array([])
    B0_all = np.array([])
    
    
    
    
    for o in orbit:
    
        df_cme = pd.read_pickle(path_to_ICME_event_files + 'ICME_Event_'+str(event)+'_'+str(o)+'.pkl')
        
        L1,z1,b1= add_to_cs_plots(df_cme,o)
        

        
        L_all = np.append(L_all,L1)
        
        z0_all = np.append(z0_all,z1)
        
        B0_all = np.append(B0_all,b1)
        
    mhd_run = pd.read_csv(path_to_folder+'MHDSim_MESSENGER_ICME_20111230_MAG_Density_run2_new.dat',index_col=False)
    
    #df_nobs
    mhd_run = mhd_run.rename(columns={'Sim_Bx [nT]':'magx',
                                      'Sim_By [nT]':'magy'})
    
    mhd_run.columns = ['time','magx','magy','magz','magamp','n','ephx','ephy','ephz']
    mhd_run.ephz = mhd_run.ephz.to_numpy()-0.19
    time_datetime = np.array([convert_to_datetime(d) for d in mhd_run.time.to_numpy()])
    
    mhd_run.time = time_datetime
    
    
    L1,z1,B1 = add_to_cs_plots(mhd_run,'model')
    
    L_all = np.append(L_all,L1)
    
    z0_all = np.append(z0_all, z1)
    
    B0_all = np.append(B0_all,B1)
    
    
    
    
    
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
                
                df = pd.read_pickle(path_to_ICME_event_files+'ICME_Event_'+str(ii)+'_'+str(i)+'.pkl')
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

    
    
    
    

    
    
    
    fig,ax = plt.subplots(1)
    sz = 30
    
    ax.scatter(z0_all, L_all, color = 'gray', alpha = 0.5,label = 'Surrounding Orbits',s=sz)
    
    ax.scatter(z0_all[6], L_all[6], color = 'darkorange', alpha = 1, label = 'ICME Orbit',s=sz)
    
    ax.scatter(z0_all[-1], L_all[-1], color = 'mediumpurple', alpha = 1, label = 'MHD Model',s=sz)
    
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
    
    '''Add trajectory information with certain color values to match the dates'''
    
    
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
            
            ax1.scatter(ephx,ephz,c=col,s=6,cmap=cmap,zorder=3)
            
            ax2.scatter(ephx,ephy,c=col,s=6,cmap=cmap,zorder=3)
            
            ax3.scatter(ephy,ephz,c=col,s=4,cmap=cmap,zorder=3)
               
def shade_in_time_series(ax,tstart,tstop,color,shade=False,alpha=.05):
    import matplotlib.pyplot as plt
    
    y=np.array([-10000000000000,100000000000000])
    
    if shade==False:
    
        ax.axvline(tstart,color=color)
            
        ax.axvline(tstop,color=color)
    
    if shade==True:
        ax.fill_betweenx(y,tstart,tstop,color=color,alpha=alpha)
     
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
            
            df = pd.read_pickle(path_to_ICME_event_files+'ICME_Event_'+str(ii)+'_'+str(i)+'.pkl')
            df_nobs = pd.read_pickle(path_to_ICME_event_files + 'ICME_Event_NOBS_'+str(ii)+'_'+str(i)+'.pkl')
            
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
    
    
def event_properties(event,parameter):
    
    p = np.array([])
    
    for i in range(11):
        
        icme_event=pd.read_pickle('/Users/bowersch/Desktop/Python_Code/Alfven_Wing/ICME_Dictionary_'+str(event)+'_'+str(i)+'.pkl')
        
        if i != 6:
        
            p = np.append(p,icme_event[parameter])
            
        if i == 6:
            
            print('ICME '+parameter+' : '+str(icme_event[parameter]))
            
            
    return p
        

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
            
    
    
    
    
    
    
    
