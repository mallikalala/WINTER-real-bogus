# get ra,dec from avro
# query ra,dec within 3 arcsec
# check output

import os
import io
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
import fastavro

from astropy.time import Time
from astropy.io import fits
import os
import glob
from scipy.stats import sigmaclip


from astropy.coordinates import SkyCoord


###### we need this to query ZTF
from penquins import Kowalski

kowalski_auth = {
"username": "tahumada",
"password": "JDJiJDEyJEVYNlY3QzJkaVlUSU5rUGtlUWhYT3UvSlphZlVSNE40b2VQMGQ3VC5Gd0NaQVY0TU1Zd3JD",
"email": "tahumada@astro.caltech.edu",
"token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoidGFodW1hZGEiLCJjcmVhdGVkX2F0IjoiMjAyMy0wNC0xMlQxNzo0ODozMy4xNzQwODArMDA6MDAifQ.tAiRyZpup6nd-QS7ihjp-BjYB8XI20zOgbrPJi59Xe8"
}

kowalski = Kowalski(token=kowalski_auth['token'])
#######

#Function definitions

def make_dataframe(packet):
    df = pd.DataFrame(packet['candidate'], index=[0])
    if len(packet['prv_candidates']) > 0:
        df_prv = pd.DataFrame(packet['prv_candidates'])
        return pd.concat([df,df_prv], ignore_index=True)
    else:
        return df

def make_dataframe_ZTF(packets):
    df = pd.DataFrame(packets[0]['candidate'], index=[0])
    for packet in packets[1:]:
        df_t = pd.DataFrame(packet['candidate'], index=[0])
        df = pd.concat([df,df_t], ignore_index=True)
    return df
    
type = []
def plot_lightcurve(dflc, ax= None, days_ago=True,telescope = 'WINTER',plot_metadata=True):
    if telescope == 'ZTF':
        filter_color = {1:'green', 2:'red', 3:'blue'}
        if days_ago:
            now = Time.now().jd
            t = dflc.jd - now
            xlabel = 'Days Ago'
        else:
            t = dflc.jd
            xlabel = 'Time (JD)'
        
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            
        for fid, color in filter_color.items():
            # plot detections in this filter:
            w = (dflc.fid == fid) & ~dflc.magpsf.isnull()
            if np.sum(w):
                ax.errorbar(t[w],dflc.loc[w,'magpsf'], dflc.loc[w,'sigmapsf'],fmt='.',color=color)
            wnodet = (dflc.fid == fid) & dflc.magpsf.isnull()
            if np.sum(wnodet):
                ax.scatter(t[wnodet],dflc.loc[wnodet,'diffmaglim'], marker='v',color=color,alpha=0.25)
    if telescope == 'WINTER':
        filter_color = {1:'black', 2:'black', 3:'black'}
        if days_ago:
            now = Time.now().jd
            t = dflc.jd - now
            xlabel = 'Days Ago'
        else:
            t = dflc.jd
            xlabel = 'Time (JD)'
        
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            
        for fid, color in filter_color.items():
            # plot detections in this filter:
            w = (dflc.fid == fid) & ~dflc.magpsf.isnull()
            if np.sum(w):
                ax.errorbar(t[w],dflc.loc[w,'magpsf'], dflc.loc[w,'sigmapsf'],fmt='.',color=color)
            wnodet = (dflc.fid == fid) & dflc.magpsf.isnull()
            if np.sum(wnodet):
                ax.scatter(t[wnodet],dflc.loc[wnodet,'diffmaglim'], marker='v',color=color,alpha=0.25)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Magnitude')
    if telescope == 'ZTF' and plot_metadata:
        metadata = ['sgscore1','distpsnr1']
        x,y = ax.get_xlim(), ax.get_ylim()
        for i,m in enumerate(metadata):
            s = m+':'+str(np.round(dflc[m][0],3))
            ax.text(x[1]-(x[1]-x[0])/2,y[0]+((y[1]-y[0])/3)-i*(y[1]-y[0])/10,s)
        if dflc['sgscore1'][0] > 0.9 and dflc['distpsnr1'][0] < 2:
            ax.text(x[0]+(x[1]-x[0])/4,y[0]+((y[1]-y[0])/3),"star")
            type.append("star")
        elif dflc['sgscore1'][0] < 0.5 and dflc['distpsnr1'][0] < 2:
            ax.text(x[0]+(x[1]-x[0])/4,y[0]+((y[1]-y[0])/3),"nuclear")
            type.append("nuclear")
        elif dflc['sgscore1'][0]<0.5 and dflc['distpsnr1'][0] > 2:
            ax.text(x[0]+(x[1]-x[0])/4,y[0]+((y[1]-y[0])/3),"offset from a galaxy")
            type.append("offset from a galaxy")
        elif dflc['distpsnr1'][0] > 2:
            ax.text(x[0]+(x[1]-x[0])/4,y[0]+((y[1]-y[0])/3),"hostless")
            type.append("hostless")
        else:
            ax.text(x[0]+(x[1]-x[0])/4,y[0]+((y[1]-y[0])/3),"not classified")
            type.append("not classified")
            
#     We should look for
# stars: high sgscore (>0.9) and low distance (<2)
# nuclear: low sgscore (<0.5) and low distance (<2)
# offset transients: high distance (>2)


def show_stamps(packet, mode='linear', telescope = 'WINTER', packet_lc=None, save=None,save_folder=None):
    #fig, axes = plt.subplots(1,3, figsize=(12,4))
    fig = plt.figure(figsize=(16,4))
    ax = fig.add_subplot(1,4,1)
    dflc_ztf = make_dataframe_ZTF(packet_lc)
    dflc_winter = make_dataframe(packet)
    plot_lightcurve(dflc_ztf,ax=ax,telescope = 'ZTF')
    plot_lightcurve(dflc_winter,ax=ax,telescope = 'WINTER')
    name = save
    ax.set_title(name)
    if save != None:
        plt.savefig(save_folder+save+'_'+telescope+'.png',dpi=250)
    return fig

# for ZTF
# we will query the avro packets from Kowalski

def get_ZTFcutout(ZTFname):
    # defining a general query
    q = {
        "query_type": "find",
        "query": {
            "catalog": "ZTF_alerts",
            "filter": {
                "objectId": ZTFname
            }  
            }
        }

    # modifications to get the lightcurve
    lc_projection = {'projection':{
                "objectId": 1,
                "candidate": 1,
                }}

    # modifications to get the cutouts
    cutout_projection = {'projection':{
                "_id": 0,
                "objectId": 1,
                'cutoutScience': 1,
                'cutoutTemplate': 1,
                'cutoutDifference': 1,
                }}
    
    # querying the lightcurves (lcs)
    q_lc = q.copy()
    q_lc['query'].update(lc_projection)
    response_lc = kowalski.query(query=q_lc)
    lc_data = response_lc.get('default').get("data") # retrieving data from one instance
    
    # querying the cutouts
    q_cutout = q.copy()
    q_cutout['query'].update(cutout_projection)
    q_cutout["kwargs"] =  {'limit': 1} # we need to limit to one so we don't load ALL the cutouts
    response_cutout = kowalski.query(query=q_cutout)
    cutout_data = response_cutout.get('default').get("data") # retrieving data from one instance

    # return the lc table and the cutouts
    return lc_data,cutout_data[0]


table_path = ['/home/msheshadri/real-bogus/sample_quality/csv/gold_matches.csv']
names = ['/home/msheshadri/real-bogus/gold_plots/gold_plots.pdf']

for ii, night in enumerate(table_path):
    print(night)
    # read table 
    table = pd.read_csv(night)
    table = table[0:20]
    with PdfPages(names[ii]) as pdf:
    
        # get WINTER information for the first row
        
        for i in range(len(table)):
            avro_path = table['file'][i]
            avro_idx = table['packet_idx'][i]
            save_folder = '/home/msheshadri/real-bogus/gold_plots/lightcurves/'
            ZTFname = table['ztf_xmatch'][i]
            WINTER_name = table['WINTER_name'][i]
            save_name = WINTER_name+'_'+ZTFname
            
            with open(avro_path,'rb') as f:
                freader = fastavro.reader(f)
                avro_content = list(freader) # this has all the packets! 
                print('in this avro file there are', len(avro_content), 'packets')
            
            # reading the correct avro package (that was crossmatched)
            xmatched_packet = avro_content[avro_idx]
            lc_data_ZTF, cutout_data_ZTF = get_ZTFcutout(ZTFname)
            
            #show_stamps(xmatched_packet,mode='sigmaclip',save=save_name,save_folder = save_folder)
            #show_stamps(cutout_data_ZTF,mode='arcsinh',telescope = 'ZTF',packet_lc=lc_data_ZTF,save=save_name, save_folder= save_folder)

            f_WINTER_ZTF = show_stamps(xmatched_packet,mode='sigmaclip',packet_lc=lc_data_ZTF, save=save_name,save_folder = save_folder)
            #f_ZTF = show_stamps(cutout_data_ZTF,mode='sigmaclip',telescope = 'ZTF',packet_lc=lc_data_ZTF,save=save_name, save_folder= save_folder)
            pdf.savefig(f_WINTER_ZTF)  # Save the first figure
            plt.close(f_WINTER_ZTF)

            