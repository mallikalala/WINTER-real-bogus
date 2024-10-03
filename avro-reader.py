f# get ra,dec from avro
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
    

def plot_lightcurve(dflc, ax= None, days_ago=True,telescope = 'WINTER',plot_metadata=True,object_type = []):
    
    filter_color = {1:'green', 2:'red', 3:'black'}
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
            object_type = "star"
        elif dflc['sgscore1'][0] < 0.5 and dflc['distpsnr1'][0] < 2:
            ax.text(x[0]+(x[1]-x[0])/4,y[0]+((y[1]-y[0])/3),"nuclear")
            object_type = "nuclear"
        elif dflc['sgscore1'][0]<0.5 and dflc['distpsnr1'][0] > 2:
            ax.text(x[0]+(x[1]-x[0])/4,y[0]+((y[1]-y[0])/3),"offset from a galaxy")
            object_type = "offset from a galaxy"
        elif dflc['distpsnr1'][0] > 2:
            ax.text(x[0]+(x[1]-x[0])/4,y[0]+((y[1]-y[0])/3),"hostless")
            object_type = "hostless"
        else:
            ax.text(x[0]+(x[1]-x[0])/4,y[0]+((y[1]-y[0])/3),"not classified")
            object_type = "not classified"
    return object_type
            
#     We should look for
# stars: high sgscore (>0.9) and low distance (<2)
# nuclear: low sgscore (<0.5) and low distance (<2)
# offset transients: high distance (>2)


def plot_cutout(stamp, fig=None, subplot=None, mode = 'linear', **kwargs):
    with gzip.open(io.BytesIO(stamp), 'rb') as f:
        
        if fig is None:
            fig = plt.figure(figsize=(4,4))
        if subplot is None:
            # subplot = (1,1,1)
            Rows,Cols,Position = 1,1,1
        else:
            Rows,Cols,Position = subplot
            
        ax = fig.add_subplot(Rows,Cols,Position)
        hdul= fits.open(io.BytesIO(f.read())) 
        im = hdul[0].data
        counting_nans = np.count_nonzero(~np.isnan(data))
        
        if mode == 'linear':
            
            ax.imshow(im)
        
        if mode == 'arcsinh':
            im = np.arcsinh(im)
            vmin,vmax,std = np.nanmin(im), np.nanmax(im),np.nanstd(im)
            ax.imshow(im,vmin=vmin-0*std, vmax=vmin+4*std)

        if mode == 'log':
            im = np.log(im)
            vmin,vmax,std = np.nanmin(im), np.nanmax(im),np.nanstd(im)
            ax.imshow(im,vmin=vmin-2*std, vmax=vmin+5*std)

        if mode == 'sigmaclip':
            # im = im #getting rid of nans
            # get the values of the values using sigmaclip
            # im = np.arcsinh(hdul[0].data)
            c, low, upp = sigmaclip(im[~np.isnan(im)])
            ax.imshow(im,vmin=low, vmax=upp)
        
    return fig, counting_nans

def show_stamps(packet, mode='linear', telescope = 'WINTER', packet_lc=None, save=None,save_folder=None):
    #fig, axes = plt.subplots(1,3, figsize=(12,4))
    fig = plt.figure(figsize=(16,4))
    ax = fig.add_subplot(1,4,1)
    if telescope == 'ZTF':
        dflc = make_dataframe_ZTF(packet_lc)
    else:
        dflc = make_dataframe(packet)
    object_type = plot_lightcurve(dflc,ax=ax,telescope = telescope)
    name = save.split('_')
    if telescope == 'ZTF':
        ax.set_title(name[1])
    else:
         ax.set_title(name[0])
    
    cutouts = ['_science','_template','_difference']
    if telescope == 'ZTF':
        cutouts = ['Science','Template','Difference']
    for i, cutout in enumerate(cutouts):
        if telescope=='ZTF':
            stamp = packet['cutout{}'.format(cutout)]['stampData']
        else: 
            stamp = packet['cutout{}'.format(cutout)]
        ffig = plot_cutout(stamp, fig=fig, subplot = (1,4,i+2),mode=mode)
        plt.title(cutout)
        
    if save != None:
        plt.savefig(save_folder+save+'_'+telescope+'.jpg',dpi=250)
    return fig, object_type

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

# getting the coordinates from the dataframe

def get_ZTFinWINTER(WINTER_detections,radius = 2): #arcsecs

    coords_arr = list(zip(WINTER_detections['ra'], WINTER_detections['dec']))
    print('hi')
    query = {"query_type": "cone_search",
                 "query": {"object_coordinates": {
                                        "radec": f"{coords_arr}",
                                        "cone_search_radius": f"{radius}",
                                        "cone_search_unit": "arcsec"
                                        },
                         "catalogs": {
                              "ZTF_alerts": {
                                             "filter": {
                                                        "candidate.drb":
                                                        {'$gt': 0.8},
                                                         },
                                             "projection": {
                                                            "objectId": 1,
                                                            }
                                               }
                               },
                 }
                 }
    # Perform the query
    r = kowalski.query(query=query)
    # reading the results:
    # we will save the WINTER coordinates and the ZTF name that is xmatched to the WINTER data

    check= 0
    coords, ztf_xmatch = [],[] 
    if r['default']['status'] == 'success': #r = results checking if query was successful
        results = r['default']['data']['ZTF_alerts'] 
        for coord in results: #look for elements with 1+ xmatches
            check = check+1
            if len(results[coord])>0:
                coords.append(coord)
                ztf_xmatch.append(results[coord][0]['objectId'])
    if len(coords) != 0:
        # we crossmatch again to get the name of the WINTER source 
        c_ztf = SkyCoord(np.array([s[1:-1].replace('_','.').split(',') for s in coords]).astype(float),unit=('deg','deg'))
        c_winter = SkyCoord(coords_arr,unit=('deg','deg'))
        idx,d2d,d3d = c_ztf.match_to_catalog_sky(c_winter)
    
        # making a table to return
        
        WINTER_in_ZTF = WINTER_detections.iloc[idx]
        WINTER_in_ZTF.reset_index(drop=True)
        WINTER_in_ZTF['ztf_xmatch'] = ztf_xmatch
        return WINTER_in_ZTF
    else:
        return None 
    
#FOR MULTIPLE NIGHTS

# reading avro packages

night_files= []
night_idx = []
m = glob.glob("/data/loki/data/winter/20*")
#for i in range(2):  # uncomment to run on just a few of the nights

for i in range(len(m)): #for all nights!
    idx = m[i].find("20")
    night_id = m[i][idx:]
    path = '/data/loki/data/winter/' + night_id +'/avro/*'
    night_files.append(glob.glob(path))
    night_idx.append(night_id)


# getting the folder where the avro files live

WINTER_detections = []

# let's get only the RA, DEC, the avro file name, and the index of the packet

for avro_files in night_files:
    # for each avro package in the folder get the folowing:
    avros = []
    ra,dec,file,packet_idx,winterID = [],[],[],[],[]
    for avro in avro_files:
        
        fname = avro
        with open(fname,'rb') as f: # read the avro package
            freader = fastavro.reader(f) # this has multiple packets
            for i,packet in enumerate(freader):
                ra.append(packet['candidate']['ra']) 
                dec.append(packet['candidate']['dec'])
                file.append(fname)
                packet_idx.append(i)
                winterID.append(packet['objectid'])
    
    # saving this in a dataframe
    
    d = {
        'WINTER_name':winterID,
        'ra':ra,
        'dec':dec,
        'file':file,
        'packet_idx':packet_idx
    }
    detections = pd.DataFrame(data=d)
    WINTER_detections.append(detections)
#print(WINTER_detections)
  
nights = {}
pdf_names = []
for k in range(len(WINTER_detections)):
    Temp_detections = WINTER_detections[k]
    frames =[]
    i = 0
    while i < len(Temp_detections):
        if i <= (len(Temp_detections) - 600):
            j = i+600
            temp= get_ZTFinWINTER(Temp_detections[i:j],radius = 2)
            print(i,j)
            frames.append(temp)
            i += 600
            
        else:
            j = len(Temp_detections)
            temp=get_ZTFinWINTER(Temp_detections[i:j],radius = 2)
            frames.append(temp)
            break
    len(frames)
    if len(frames) > 1:
        crossmatches = pd.concat(frames)
        temp = night_idx[k]
        print(k)
        nights[temp] = crossmatches
        pdf_names.append(temp + '.pdf')
    elif len(frames) == 1:
        crossmatches = pd.DataFrame(frames[0])
        temp = night_idx[k]
        print(k)
        nights[temp] = crossmatches
        pdf_names.append(temp + '.pdf')
    else:
        continue

print(pdf_names)
   
#Save the CSV files with the date of each night in the file name
table_path = []
for key in nights:
    nights_dataframe = pd.DataFrame(nights[key])
    nights_dataframe.to_csv("/home/msheshadri/real-bogus/test_files/csv/" + str(key) + ".csv")
    table_path.append('/home/msheshadri/real-bogus/test_files/csv/' + str(key) + '.csv')
    
print(nights)
print(table_path)

#THIS PART IS FOR STAMPS

type_column = {}
for ii, night in enumerate(table_path):
    print(night)
    # read table 
    table = pd.read_csv(night)
    key = pdf_names[ii]
    type_column[key] = []
    with PdfPages(pdf_names[ii]) as pdf:
    
        # get WINTER information for the first row

        for i in range(len(table)):
            avro_path = table['file'][i]
            avro_idx = table['packet_idx'][i]
            save_folder = '/home/msheshadri/real-bogus/test_files/jpg/'
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
            
            f_WINTER, object_type_winter = show_stamps(xmatched_packet,mode='sigmaclip',save=save_name,save_folder = save_folder)
            f_ZTF, object_type_ZTF = show_stamps(cutout_data_ZTF,mode='sigmaclip',telescope = 'ZTF',packet_lc=lc_data_ZTF,save=save_name, save_folder= save_folder)
            print(object_type_ZTF)

            type_column[key].append(object_type_ZTF)
                
            pdf.savefig(f_WINTER)  # Save the first figure
            pdf.savefig(f_ZTF)
            plt.close(f_WINTER)
            plt.close(f_ZTF)

    print('type column')
    print(type_column)
    print('object_type_ZTF')
    print(object_type_ZTF)
    crossmatches = pd.read_csv(table_path[ii])
    crossmatches.insert(6, "Type", type_column[key], True) 
    crossmatches.to_csv(table_path[ii])

