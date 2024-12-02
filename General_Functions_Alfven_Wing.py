#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:18:39 2024

@author: bowersch
"""

#import pytplot
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import datetime as dt
#matplotlib qt

import pandas as pd
import sys
from matplotlib.widgets import Cursor
#import pyspedas
import datetime
from matplotlib.ticker import MultipleLocator,AutoMinorLocator


#Functions

def get_mercury_distance_to_sun(date):
    # create a PyEphem observer for the Sun
    
    import ephem
    
    j = ephem.Mercury()
    
    j.compute(date,epoch='1970')

    distance_au=j.sun_distance
    
    
    
    return distance_au

def get_day_of_year(date_string):
    import datetime
    date_obj = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    return date_obj.timetuple().tm_yday

def convert_to_date_2(utc_val):
    import datetime as dt
    dt_object = dt.datetime.fromtimestamp(utc_val)
    
    dt_string=dt_object.strftime("%Y-%m-%d %H:%M:%S")
    
    return dt_string

def convert_to_date_2_utc(utc_val):
    import datetime as dt
    dt_object = dt.datetime.utcfromtimestamp(utc_val)
    
    dt_string=dt_object.strftime("%Y-%m-%d %H:%M:%S")
    
    return dt_string

def convert_to_date(utc_val):
    import datetime
    dt_string=datetime.datetime.strptime(convert_to_date_2(utc_val), '%Y-%m-%d %H:%M:%S.%f')
    return dt_string

def convert_to_utc(date):
    date_string = date
    date_obj = dt.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    timestamp = date_obj.timestamp()
    
    return timestamp

def convert_to_date_s(date):
    date_obj = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    return date_obj

def convert_to_datetime(date_string):
    import datetime
    date_obj=datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    
    return date_obj

def convert_to_datetime_K(date_string):
    import datetime
    date_obj = datetime.datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%f')
    
    return date_obj

def convert_datetime_to_string(time):
    dstring=time.strftime("%Y-%m-%d %H:%M:%S")
    return dstring
def convert_from_txt_to_date(file):
    x_in=np.loadtxt(file,usecols=(0,1,2,3,4,5))
    x_out=np.loadtxt(file,usecols=(6,7,8,9,10,11))
    date_in=np.array([])
    date_out=np.array([])
    for i in range(np.size(x_in[:,0])):
        
        if int(np.floor(x_in[i,5])) >= 60:
            x_in[i,5]=0.0
            x_in[i,4]=x_in[i,4]+1
            
        if int(np.floor(x_out[i,5])) >= 60:
            x_out[i,5]=0.0
            x_out[i,4]=x_out[i,4]+1
            
        if int(np.floor(x_out[i,5])) < 0:
            x_out[i,5]=59
            x_out[i,4]=x_out[i,4]-1
            
            if x_out[i,4]<0:
                x_out[i,3]=x_out[i,3]-1
                x_out[i,4]=59
            
        
        if int(np.floor(x_in[i,5])) < 0:
            x_in[i,5]=59
            x_in[i,4]=x_in[i,4]-1
            if x_in[i,4]<0:
                x_in[i,3]=x_in[i,3]-1
                x_in[i,4]=59
        
        date_in=np.append(date_in,str(int(np.floor(x_in[i,0])))+'-'+str(int(np.floor(x_in[i,1])))+'-'+str(int(np.floor(x_in[i,2])))+' '+str(int(np.floor(x_in[i,3])))+
                          ':'+str(int(np.floor(x_in[i,4])))+':'+str(int(np.floor(x_in[i,5]))))
        date_out=np.append(date_out,str(int(np.floor(x_out[i,0])))+'-'+str(int(np.floor(x_out[i,1])))+'-'+str(int(np.floor(x_out[i,2])))+' '+str(int(np.floor(x_out[i,3])))+
                           ':'+str(int(np.floor(x_out[i,4])))+':'+str(int(np.floor(x_out[i,5]))))
                                                            
        
        
    date=np.array([date_in,date_out])
    
    return date
def read_in_Weijie_files():
    file_mp_in='/Users/bowersch/Desktop/MESSENGER Data/Weijie Crossings/MagPause_In_Time_Duration_ver04_public_version.txt'
    file_mp_out='/Users/bowersch/Downloads/MagPause_Out_Time_Duration_public_version_WeijieSun_20230829.txt'
    file_bs_in='/Users/bowersch/Desktop/MESSENGER Data/Weijie Crossings/Bow_Shock_In_Time_Duration_ver04_public_version.txt'
    file_bs_out='/Users/bowersch/Downloads/Bow_Shock_Out_Time_Duration_public_version_WeijieSun_20230829.txt'
    
    mp_in=convert_from_txt_to_date(file_mp_in)
    mp_out=convert_from_txt_to_date(file_mp_out)
    bs_in=convert_from_txt_to_date(file_bs_in)
    bs_out=convert_from_txt_to_date(file_bs_out)
    
    return mp_in, mp_out, bs_in, bs_out  

def check_for_mp_bs_WS(time,mp_in,mp_out,bs_in,bs_out):
    
    #To get mp/bs data: mp_in, mp_out, bs_in, bs_out = read_in_Weijie_files()
    
    #Get time_range of loaded data
    
    timespan=[time[0],time[-1]]
    
    #Convert mp/bs data into UTC:
    mp_in_utc = np.empty((0,2), float)
    mp_out_utc=np.empty((0,2), float)
    
    bs_in_utc=np.empty((0,2), float)
    bs_out_utc=np.empty((0,2), float)
    
    for i in range(np.size(mp_in[0,:])):
        
        mp_in_utc=np.append(mp_in_utc,[[convert_to_date_s(mp_in[0,i]),convert_to_date_s(mp_in[1,i])]],axis=0)
        
        mp_out_utc=np.append(mp_out_utc,[[convert_to_date_s(mp_out[0,i]),convert_to_date_s(mp_out[1,i])]],axis=0)
        
    for i in range(np.size(bs_in[0,:])):

    
        bs_in_utc=np.append(bs_in_utc,[[convert_to_date_s(bs_in[0,i]),convert_to_date_s(bs_in[1,i])]],axis=0)
    
    
    for i in range(np.size(bs_out[0,:])):
        
        bs_out_utc=np.append(bs_out_utc,[[convert_to_date_s(bs_out[0,i]),convert_to_date_s(bs_out[1,i])]],axis=0)
        
    #Check in any boundaries are within the timespan
    
    #Mp_in?
    
    mp_in_ts=mp_in_utc[(mp_in_utc[:,0] >timespan[0]) & (mp_in_utc[:,1]<timespan[1])]
    
    mp_out_ts=mp_out_utc[(mp_out_utc[:,0] > timespan[0])& (mp_out_utc[:,1]<timespan[1])]
    
    bs_in_ts=bs_in_utc[(bs_in_utc[:,0] >timespan[0]) & (bs_in_utc[:,1]<timespan[1])]
    
    bs_out_ts=bs_out_utc[(bs_out_utc[:,0] > timespan[0])& (bs_out_utc[:,1]<timespan[1])]
    
        
    return mp_in_ts, mp_out_ts, bs_in_ts, bs_out_ts
def get_aberration_angle(date):
    
    import numpy as np
    
    r=get_mercury_distance_to_sun(date)*1.496E11
    
    a=57909050*1000.
    
    M=1.9891E30
    
    G=6.67430E-11
    
    v=np.sqrt(G*M*(2./r-1./a))
    
    alpha=np.arctan(v/400000)
    
    return alpha

def plot_mp_and_bs(ax1,mso=False):
    
    ''' Plot the magnetopause and bow shock on a general axis given by ax1'''
    
    y_mp=np.linspace(-100,100,100)
    z_mp=np.linspace(-100,100,100)
    x_mp=np.linspace(-10,10,100)
    
    rho=np.sqrt(y_mp**2+(z_mp)**2)
    

    
    phi=np.arctan2(rho,x_mp)
    
    
    Rss=1.45
    
    alpha=0.5
    
    phi2 = (np.linspace(0,2*np.pi,100))
    
    rho=Rss*(2/(1+np.cos(phi2)))**(alpha)
    
    xmp=rho*np.cos(phi2)
    
    ymp=rho*np.sin(phi2)
    
    if mso==True:
        
        ymp=ymp+.196
    
    ax1.plot(xmp,ymp,color='black',linestyle='--',linewidth=3)

    
    psi=1.04

    p=2.75

    L=psi*p

    x0=.5

    phi = (np.linspace(0,2*np.pi,100))
    rho = L/(1. + psi*np.cos(phi))

    xshock = x0 + rho*np.cos(phi)
    yshock = rho*np.sin(phi)
    
    if mso==True:
        
        yshock=yshock+.196

    ax1.plot(xshock,yshock,color='black',linestyle='--',linewidth=3)
    
    plt.show()

def load_MESSENGER_into_tplot(date_string,res="01",full=False,FIPS=False):
    #res can be 01,05,10,60
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
    
    #Offset due to dipole field!
    
    ephz=ephz-479
    
    #Aberration:
        
    phi=get_aberration_angle(date_string)
    
    new_magx=mag1[:,0]*np.cos(phi)-mag1[:,1]*np.sin(phi)
    
    new_magy=mag1[:,0]*np.sin(phi)+mag1[:,1]*np.cos(phi)
    
    mag1[:,0]=new_magx
    mag1[:,1]=new_magy
    
    
    new_ephx=ephx*np.cos(phi)-ephy*np.sin(phi)
    
    
    new_ephy=ephx*np.sin(phi)+ephy*np.cos(phi)
    
    ephx=new_ephx
    ephy=new_ephy
    
    eph=np.transpose(np.vstack((ephx,ephy,ephz)))
    
    if full==True:
        mag1=df[:,9:]
        eph=df[:,5:8]
        ephx=df[:,5]
        ephy=df[:,6]
        ephz=df[:,7]
    
    #Define magnetic field amplitude
    
    
    magamp=np.sqrt(mag1[:,0]**2+mag1[:,1]**2+mag1[:,2]**2)
    
    if FIPS==False:
        return time, mag1, magamp, eph
    
    if FIPS==True:
        
        year=date_string[2:4]
        
        import os
        
        
        file='/Users/bowersch/Desktop/MESSENGER Data/mess-fips '+year+'/'+str(month)+'/FIPS_ESPEC_'+year_full+doy_s+'_DDR_V01.TAB'
        
        if os.path.isfile(file)==False:
            file='/Users/bowersch/Desktop/MESSENGER Data/mess-fips '+year+'/'+str(month)+'/FIPS_ESPEC_'+year_full+doy_s+'_DDR_V03.TAB'
            
        if os.path.isfile(file)==False:
            file='/Users/bowersch/Desktop/MESSENGER Data/mess-fips '+year+'/'+str(month)+'/FIPS_ESPEC_'+year_full+doy_s+'_DDR_V02.TAB'
            
        if os.path.isfile(file)==False:
            #print('No FIPS File')
            #print(file)
            return time,mag1,magamp,eph
        
        df = np.genfromtxt(file)
        
        

        erange=[13.577,12.332,11.201,10.174,9.241,8.393,7.623,6.924,
                6.289,5.712,5.188,4.713,4.28,3.888,3.531,3.207,2.913,2.646,2.403,2.183,1.983,1.801,
                1.636,1.485,1.349,1.225,1.113,1.011,0.918,0.834,0.758,0.688,0.625,0.568,
                0.516,0.468,0.426,0.386,0.351,0.319,0.29,0.263,0.239,0.217,0.197,0.179,0.163,0.148,
                0.134,0.122,0.111,0.1,0.091,0.083,0.075,0.068,0.062,0.056,0.051,0.046,
                0.046,0.046,0.046,0.046]

        df_data=df[:,2:]

        df_dates=df[:,1]
        
        cutoff=200411609
        
        #cutoff=cutoff.timestamp()

        if df_dates[-1] > cutoff:
            

            datetime_MET=convert_to_date_s('2004-08-03 06:55:32')
            
        else: 
            
            datetime_MET=convert_to_date_s('2013-01-08 20:13:20')

        df_dates2=df_dates+datetime_MET.timestamp()

        df_datetime=[convert_to_date_2_utc(d) for d in df_dates2]

        ds=df_datetime[0]

        df_datetime=[convert_to_datetime(d) for d in df_datetime]

        df_data=np.reshape(df_data,(np.size(df[:,0]),5,64))

        H_data=df_data[:,0,:]
        
        #time_FIPS=np.array(df_datetime)-datetime.timedelta(hours=1)
        
        time_FIPS=np.array(df_datetime)
        
        
        if os.path.isfile(file)==True:
        
            return time, mag1, magamp, eph, time_FIPS, H_data
            
        if os.path.isfile(file)==False:
            return time, mag1, magamp, eph



def append_for_run_over(date_string):
    
    '''Append time series together for tranges that span more than one day '''
    
    import numpy as np
    import datetime
    time1,mag1,magamp1,eph1=load_MESSENGER_into_tplot(date_string)
    time1=np.array(time1)
    
    mp_in, mp_out, bs_in, bs_out = read_in_Weijie_files()
    
    d1=convert_to_datetime(date_string+" 00:00:00")
    
    d2=d1+datetime.timedelta(days=1)
    
    date_string2=convert_datetime_to_string(d2)[0:10]

    time2,mag2,magamp2,eph2=load_MESSENGER_into_tplot(date_string2)
    time2=np.array(time2)
    
    d3=d1-datetime.timedelta(days=1)
    
    date_string2=convert_datetime_to_string(d3)[0:10]
    
    time3,mag3,magamp3,eph3=load_MESSENGER_into_tplot(date_string2)
    time3=np.array(time3)
    
    
    
    time=np.append(time3,time1)
    
    time=np.append(time,time2)
        
    mp_in_ts, mp_out_ts, bs_in_ts, bs_out_ts=check_for_mp_bs_WS(time,mp_in,mp_out,bs_in,bs_out)
    
    mag=np.append(mag3,mag1,axis=0)
    
    mag=np.append(mag,mag2,axis=0)
    
    magamp=np.append(magamp3,magamp1)
    
    magamp=np.append(magamp,magamp2)
    
    eph=np.append(eph3,eph1,axis=0)
    
    eph=np.append(eph,eph2,axis=0)
    
    return time,mag,magamp,eph,mp_in_ts,mp_out_ts,bs_in_ts,bs_out_ts

def check_FIPS_file(date_string):
    
    from trying3 import get_day_of_year
    
    doy=get_day_of_year(date_string)
    month=date_string[5:7]
    year=date_string[2:4]
    
    year_full=date_string[0:4]
    if doy < 10: doy_s='00'+str(doy)
        
    elif (doy<100) & (doy>=10):doy_s='0'+str(doy)
        
    else: doy_s=str(doy)
    
    year=date_string[2:4]
    
    import os
    
    
    file='/Users/bowersch/Desktop/MESSENGER Data/mess-fips '+year+'/'+str(month)+'/FIPS_ESPEC_'+year_full+doy_s+'_DDR_V01.TAB'
    
    if os.path.isfile(file)==False:
        file='/Users/bowersch/Desktop/MESSENGER Data/mess-fips '+year+'/'+str(month)+'/FIPS_ESPEC_'+year_full+doy_s+'_DDR_V03.TAB'
    
    if os.path.isfile(file)==False:
        file='/Users/bowersch/Desktop/MESSENGER Data/mess-fips '+year+'/'+str(month)+'/FIPS_ESPEC_'+year_full+doy_s+'_DDR_V02.TAB'
    
    if os.path.isfile(file)==False:
        
        print('No FIPS File')
        x=False
        
    if os.path.isfile(file)==True:
        x=True
        
    return x


def append_for_run_over_FIPS(date_string):
    
    import datetime
    time1,mag1,magamp1,eph1,time_FIPS1,H_data1=load_MESSENGER_into_tplot(date_string,FIPS=True)

    time1=np.array(time1)
    time_FIPS1=np.array(time_FIPS1)
    
    d1=convert_to_datetime(date_string+" 00:00:00")
    
    d2=d1+datetime.timedelta(days=1)
    
    date_string2=convert_datetime_to_string(d2)[0:10]
    
    fcheck=check_FIPS_file(date_string2)
    
    if fcheck==True:
    
        time2,mag2,magamp2,eph2,time_FIPS2,H_data2=load_MESSENGER_into_tplot(date_string2,FIPS=True)
        time2=np.array(time2)
        time_FIPS2=np.array(time_FIPS2)
        
        time=np.append(time1,time2)

        mag=np.append(mag1,mag2,axis=0)
        
        magamp=np.append(magamp1,magamp2)
        
        eph=np.append(eph1,eph2,axis=0)
        
        time_FIPS=np.append(time_FIPS1,time_FIPS2)
        
        H_data=np.append(H_data1,H_data2,axis=0)
        
    if fcheck==False:
        
        time2,mag2,magamp2,eph2=load_MESSENGER_into_tplot(date_string2)
        time2=np.array(time2)
        
        time=np.append(time1,time2)

        mag=np.append(mag1,mag2,axis=0)
        
        magamp=np.append(magamp1,magamp2)
        
        eph=np.append(eph1,eph2,axis=0)
        
        time_FIPS=time_FIPS1
        
        H_data=H_data1
        
        
        
        
    
    
    

    
    return time,mag,magamp,eph,time_FIPS,H_data

def plot_MESSENGER_trange_3ax(trange,figure=False,plot=True):
    
    date_string=trange[0][0:10]
    
    time,mag,magamp,eph=load_MESSENGER_into_tplot(date_string)
    
    #t_sc=t_sc.strftime('%Y-%m-%d %H:%M:%S')
    
    time=np.array(time)
    
    tstart=convert_to_datetime(trange[0])
    tstop=convert_to_datetime(trange[1])
    
    gd=((time > tstart) & (time < tstop))
    
    time=time[gd]
    mag=mag[gd,:]
    eph=eph[gd,:]
    fs=20
    
    
    
    R_m=2440
    
    ephx=eph[:,0]/R_m
    
    ephy=eph[:,1]/R_m
    
    ephz=eph[:,2]/R_m
        
    fig, (ax1,ax2,ax3)=plt.subplots(1,3)
    
    #fig, ax1 = plt.subplots()
    
# =============================================================================
#     a=np.where(ephx*np.roll(ephx,1) < 0)[0]
#     ta=time[a]
#     b=np.where(np.abs(ta-np.roll(ta,1))> 10000)
#     
#     b=np.array(b)[0][:]
#     
#     o=a[b[0]][b]
#     
#     t_o=time[o]
#     
#     t_o=np.insert(t_o,0,time[0])
#     
#     t_o=np.append(t_o,time[-1])
#     
#     diff=np.abs(t_o-t_sc)
#     
#     orb_num=np.where(diff == np.min(diff))
#     
#     orb_num=orb_num[0]
#     
#     o=np.insert(o,0,0)
# 
#     o=np.append(o,np.size(time))
#     
#     if orb_num < np.size(t_o):
#             
#         gd=np.array([o[orb_num],o[orb_num+1]])
#         
#     else:
#             
#         gd=np.array([o[-2],o[-1]])
#         
#     ephxo=ephx[gd[0][0]:gd[-1][0]]
#     ephyo=ephy[gd[0][0]:gd[-1][0]]
#     ephzo=ephz[gd[0][0]:gd[-1][0]]
# =============================================================================
    
    
    
    #fig.canvas.set_window_title('MESSENGER Orbit')
    if plot==True:
        ax1.plot(ephx,ephz,color='black')
        ax2.plot(ephx,ephy,color='black')
        ax3.plot(ephy,ephz,color='black')
    
    #Plot Mercury
    
    theta = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(theta)
    y = np.sin(theta)-0.2
    
    # Plot the circle in all 3 plots
    ax1.plot(x, y, color='gray')
    ax2.plot(x,y+.2,color='gray')
    ax3.plot(x,y,color='gray')
    
    ax1.set_xlabel("$X_{MSM'}$",fontsize=fs)
    
    ax1.set_ylabel("$Z_{MSM'}$",fontsize=fs)
    
    ax2.set_xlabel("$X_{MSM'}$",fontsize=fs)
    ax2.set_ylabel("$Y_{MSM'}$",fontsize=fs)
    
    ax3.set_xlabel("$Y_{MSM'}$",fontsize=fs)
    
    ax3.set_ylabel("$Z_{MSM'}$",fontsize=fs)
    
    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    ax3.tick_params(labelsize=20)
    
    
    ax1.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax1.tick_params(axis='both', which='major', length=8)
    ax1.tick_params(axis='both', which='minor', length=4)
    
    ax2.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax2.tick_params(axis='both', which='major', length=8)
    ax2.tick_params(axis='both', which='minor', length=4)
    
    ax3.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax3.tick_params(axis='both', which='major', length=8)
    ax3.tick_params(axis='both', which='minor', length=4)
    
    plot_mp_and_bs(ax1)
    plot_mp_and_bs(ax2)
    
    # Set the limits of the plot in all 3 plots
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-8, 1.5])
    
    ax2.set_xlim([-5,5])
    ax2.set_ylim([-5,5])
    
    ax3.set_xlim([-4,6])
    ax3.set_ylim([-6,1.55])
    
    # Color the left hemisphere red and the right hemisphere gray
    ax1.fill_between(x, y, where=x<0, color='black', interpolate=True)
    ax2.fill_between(x, y+.2, where=x<0,color='black',interpolate=True)
    
    #Set equal aspect so Mercury is circular
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')
    
    if figure == False:
    
        return ax1,ax2,ax3
    
    if figure == True:
        
        return fig,ax1,ax2,ax3

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

    def is_point_inside_curve(curve_points, point):
        # Create a Path object from the curve points
        path = mpl_path.Path(curve_points)
        
        # Check if the given point lies inside the curve
        return path.contains_point(point)
    

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

def rotate_into_mso(x,y,z,t):
    
    '''rotates from msm to mso coordinates in ephemeris data'''
    z_mso=z+479
    
    #Aberration:
        
    phi=get_aberration_angle(t)
    
    x_mso=x*np.cos(-phi)-y*np.sin(-phi)
    
    y_mso=y*np.sin(-phi)+y*np.cos(-phi)
    
    return x_mso,y_mso,z_mso
    
def rotate_into_msm(x,y,z,time):
    
    '''rotates from mso to msm for non-ephemeris data'''
    #Aberration:
        
    phi=get_aberration_angle(time)
    
    x_msm=x*np.cos(phi)-y*np.sin(phi)
    
    y_msm=y*np.sin(phi)+y*np.cos(phi)
    
    return x_msm,y_msm,z


def format_ticks(ax,fs):
    
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks for the x-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.tick_params(axis='both', which='major', length=8)
    ax.tick_params(axis='both', which='minor', length=4) 
    
    ax.tick_params(labelsize=fs-5)

def calculate_tangent_vector(curve_points, index):
    # Get the neighboring points
    prev_index = (index - 1) % len(curve_points)
    next_index = (index + 1) % len(curve_points)
    prev_point = curve_points[prev_index]
    next_point = curve_points[next_index]
    
    # Calculate the tangent vector
    tangent_vector = next_point - prev_point
    
    # Normalize the tangent vector
    tangent_vector = tangent_vector/ np.linalg.norm(tangent_vector)
    
    return tangent_vector 

def plot_MESSENGER_trange_cyl(trange,plot=False):
    
    plt.rcParams.update({'font.size': 15})
    date_string=trange[0][0:10]
    
    time,mag,magamp,eph=load_MESSENGER_into_tplot(date_string)
    
    #t_sc=t_sc.strftime('%Y-%m-%d %H:%M:%S')
    
    time=np.array(time)
    
    tstart=convert_to_datetime(trange[0])
    tstop=convert_to_datetime(trange[1])
    
    gd=((time > tstart) & (time < tstop))
    
    time=time[gd]
    mag=mag[gd,:]
    eph=eph[gd,:]
    
    
    
    
    R_m=2440
    
    ephx=eph[:,0]/R_m
    
    ephy=eph[:,1]/R_m
    
    ephz=eph[:,2]/R_m
        
    fig, ax1=plt.subplots(1)
    
    r=ephy**2+ephz**2
    
    if plot==True:
        ax1.plot(ephx,r,color='black')

    
    #Plot Mercury
    
    theta = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(theta)
    y = np.sin(theta)-0.2
    
    # Plot the circle in all 3 plots
    ax1.plot(x, y, color='gray')

    
    ax1.set_xlabel("$X_{MSM\'}$ ($R_M$)",fontsize=20)
    
    ax1.set_ylabel("\u03C1$_{MSM\'}$ ($R_M$)",fontsize=20)
    
    ax1.tick_params(labelsize=20)
    
    
    plot_mp_and_bs(ax1)
    
    # Set the limits of the plot in all 3 plots
    ax1.set_xlim([-5, 3])
    ax1.set_ylim([0, 4])

    
    # Color the left hemisphere red and the right hemisphere gray
    ax1.fill_between(x, y, where = x < 0, color='black', interpolate=True)
    
    theta = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(theta)
    y = np.sin(theta)+0.2
    
    # Plot the circle in all 3 plots
    ax1.plot(x, y, color='indianred',linestyle = '--')
    
    
    #Set equal aspect so Mercury is circular
    ax1.set_aspect('equal', adjustable='box')

    
    return ax1