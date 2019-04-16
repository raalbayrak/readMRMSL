import datetime
import pytz
import pdb

import numpy as np
import numpy.ma as ma



'''
    # for numpy translations see
    #
https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64
'''

def augment_infoD(infoD, datetime_L, datetime_strL, folderL):
    infoD['dateL']     =  datetime_L
    infoD['datestrL']  =  datetime_strL
    infoD['folderName']   =  folderL 
    return infoD

# -----------------------GET__GRID_INDX------------------------
def getGriIndx(lats,lons,latv,lonv):
    #xi=np.searchsorted(lats,latv)
    #yi=np.searchsorted(lons,lonv)
    a = abs(lats-latv)+abs(lons-lonv)
    xi,yi = np.unravel_index(a.argmin(),a.shape)
    return xi,yi 


# ________________________INPUT_OUTPUT_________________________

def callSave(indx, sdate, infoD, varN = ''):
    fnp, fnh5 = get_fileNames2save(indx, sdate)
    save2meta(fnp, infoD)
    varN = folderL[indx]
    save2h5(fnh5, varN, finalM, infoD)
    return


def get_fileNames2save(indx, sdate):
    fnp  = folderL[indx] + sdate.strftime('_%Y_%m_%d') + '.p'
    fnh5 = folderL[indx] + sdate.strftime('_%Y_%m_%d') + '.h5'
    return fnp, fnh5

def save2meta(fn, DD):
    import pickle
    with open(fn, 'wb') as f:
        pickle.dump(DD, f)
    return

def loadFromMeta(fn):
    import pickle
    with open(fn,'rb') as file:
        DD = pickle.load(file)
    return DD

def save2h5(fn, varN, finalM, infoD):
    '''
        to check the content (h5ls -rd tutorial1.h5)
        ptdump -a -v new.h5   (better one)
    '''
    import h5py
    with h5py.File(fn, "w") as h5f:
        #h5f = h5py.File(fn, "w")
        dset = h5f.create_dataset(varN, data=finalM, fillvalue=int(infoD['NODATA_value']))
    return

def get_latR_lonRandmash():
    latRange = np.arange(54.995, 20, -0.01)
    #latRange = np.arange(20, 54.995, 0.01)
    lonRange = np.arange(-129.995, -60.0, 0.01)
    nhh      = 48 #; Number of Half-Hours in a day
    lon, lat = np.meshgrid(lonRange, latRange)    
    return lon, lat



# ____________________________________________________________

def collect_info4file(fn):
    import csv
    import gzip
    import io

    infoD = {}
    with gzip.open(fn, "r") as file:
        reader = csv.reader(io.TextIOWrapper(file, newline=""))
        aa = list(reader)
        for indx, iin in enumerate(aa):
            if indx == 6:
                break
            else:
                n1 = iin[0].split(' ')
                infoD[n1[0]] = n1[1]
                #print(list(reader))
    return infoD

def collect4conca(fileFormatL, checknum=np.nan):
    # checknum = 2 2nd iteration algorthm stops.
    if np.isnan(checknum):
        checknum = len(fileFormatL)    
    for indx, filename in enumerate(fileFormatL):
        print('reading {} th index'.format(indx))
        if indx == 0:
            tempM = np.genfromtxt(filename, skip_header=6)
            finalM = np.zeros([len(fileFormatL), tempM.shape[0], tempM.shape[1]])
            finalM[indx,:,:] = tempM
        else:
            if indx == checknum:
                return finalM
            mydataM1 = np.genfromtxt(filename, skip_header=6)
            finalM[indx,:,:] = mydataM1
    return finalM

def get_datelist(sdate, edate, dtime, prefix, postfix, dirn):
    datetime_L    = []
    datetime_strL = []
    fileFormatL   = []
    while sdate < edate:
        sdate = sdate + datetime.timedelta(minutes=30)
        tstr = sdate.strftime("%Y%m%d.%H%M%S")
        formated_str = dirn + '/' + prefix + tstr + postfix
        datetime_L.append(sdate)
        datetime_strL.append(str(sdate))
        fileFormatL.append(formated_str)
        
    return datetime_L, datetime_strL, fileFormatL

def str2datetime(ll):
    newL = []
    from dateutil import parser
    for dd in ll:
        dt = parser.parse(dd)
        newL.append(dt)
    return newL

def find_MRMStimeIndex(ddate, infoD):
    '''
    input: time in utc
        t = '2017-03-14T23:30:00.000Z'
    output:
        index number out of 48 halfhour list
    '''
    kk = np.array(infoD['dateL'])
    k = str2datetime([ddate])[0]
    indx = np.where(kk == k)
    return indx[0]



def plotRMRS_folium(ddata):
    '''
    maybe better tosave first si that fill values will not show with 0
    import scipy.misc
    scipy.misc.imsave('outfile.jpg', mydataM1)  
    '''
    
    import folium
    from folium import plugins
    from scipy.ndimage import imread

    #fix the negatives
    ffl = (ddata < 0.001)
    ddata[ffl] = 0.0001


    # boundary of the image on the map
    min_lon = -129.995
    max_lon = -60.0
    min_lat = 20
    max_lat = 54.995

    # create the map
    map_ = folium.Map(location=[38.2, -122],
                  tiles='Stamen Terrain', zoom_start = 8)

    ## read in png file to numpy array
    #data = imread('./ii_overlay.png')

    # Overlay the image
    map_.add_children(plugins.ImageOverlay(np.log(ddata), opacity=0.6, \
        bounds =[[min_lat, min_lon], [max_lat, max_lon]]))
    map_.save('imageover.html')
    print('plot savaed in html')
    return


def plotRMRS_basemap(ddata, lat, lon):
    day_out = 12
    from mpl_toolkits.basemap import Basemap, cm
    import matplotlib.pyplot as plt

    m = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=50,\
                llcrnrlon=-130,urcrnrlon=-60,lat_ts=20,resolution='i')
    m.drawcoastlines()
    m.drawcountries()
    #m.drawstates()
    # draw parallels and meridians.
    parallels = np.arange(-90.,91.,5.)
    # Label the meridians and parallels
    m.drawparallels(parallels,labels=[False,True,True,False])
    # Draw Meridians and Labels
    meridians = np.arange(-180.,181.,10.)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    m.drawmapboundary(fill_color='white')
    plt.title("Forecast {0} days out".format(day_out))

    #x,y = m(lon, lat)                            # This is the step that transforms the data into the map's projection
    #m.plot(x,y, 'bo', markersize=0.5)
    
    # Define a colormap
    jet = plt.cm.get_cmap('jet')
    #jet = plt.cm.get_cmap('s3pcpn')
    # Transform points into Map's projection
    x,y = m(lon, lat)
    # Color the transformed points!
    sc = plt.scatter(x,y, c=mydataM1, vmin=-2.0, vmax =2.30, cmap=jet, s=0.01)
    
    #levs = [0.10,0.20,0.30,1,1.5,2.0,3.0,3.5,3.75,4,4.25,4.50,4.75,5,5.5,6,6.5,7,8,9,10]
    #sc = m.contourf(x,y,mydataM1,clevs,cmap=cm.s3pcpn)
    
    # And let's include that colorbar
    cbar = plt.colorbar(sc, shrink = .5)
    #cbar.set_label(temp)
    plt.show()
    return





#import pandas as pd
    
#std = '2017-03-13T00:00:00Z'
#edd = '2017-03-14T23:30:00Z'

#temp1 = pd.DataFrame(columns=['NULL'], index=pd.date_range(std, edd, freq='30T'))
#temp2 = temp1.between_time('00:00','23:50').index.strftime('%Y-%m-%dT%H:%M:%SZ')
#temp3 = temp2.tolist()
#df.Date = df.Date.dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')


# for numpy translations see
#
#https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64

'''
run readMRMS.py 
finalM.shape
i,j = getGriIndx(latRange,lonRange,manhattanLat,manhattanLon)
    latRange = np.arange(54.995, 20, -0.01)                                                                      
   ...:     lonRange = np.arange(-129.995, -60.0, 0.01)                                        
   ...:     nhh      = 48 #; Number of Half-Hours in a day   
thisLat, thisLong = getGriIndx(latRange,lonRange,manhattanLat,manhattanLon) 
xi=np.searchsorted(latRange,manhattanLat) 
np.searchsorted(latRange,manhattanLat) 
lon, lat = np.meshgrid(lonRange, latRange) 
finalM[:,i,j]  
plt.imshow(finalM[0,:,:]) 
plotRMRS_basemap(mydataM1, lat, lon) 
plotRMRS_folium(mydataM1) 

    latRange = np.arange(54.995, 20, -0.01)                                                                      
   ...:     #latRange = np.arange(20, 54.995, 0.01)                                            
   ...:     lonRange = np.arange(-129.995, -60.0, 0.01)                          
   ...:     nhh      = 48 #; Number of Half-Hours in a day 
   ...:                                 
   ...:     lon, lat = np.meshgrid(lonRange, latRange)    
   ...:     manhattanLat =  39.95   # 40.758  
   ...:     manhattanLon = -75.1652 #-73.985 
   ...:                            
   ...:                           
   ...:     i,j = getGriIndx(lat,lon,manhattanLat,manhattanLon)
'''


if __name__ == "__main__":

    # %%
    #Block 1 for data retreival...
    
    sdate = datetime.datetime(2017, 3, 14, 0, 0, 0, tzinfo=pytz.utc)
    edate = datetime.datetime(2017, 3, 14, 23, 30, 0, tzinfo=pytz.utc)
    #edate = datetime.datetime(2017, 3, 14, 23, 30, 0, tzinfo=pytz.utc)
    dtime = 30 #in minutes
    
    #base_dirn    = '/media/disk/MRMS'
    base_dirn    = '/home/albayrak/Documents/prog/twitter/twitter_json/code/MRMS'
    address_dirn = '/hector.gsfc.nasa.gov/NMQ/level3-30m/2017/03/'
    dirn         = base_dirn + address_dirn
    
    #folderL      = ['1HCF', 'GCP']
    folderL      = ['GCP']
    #prefixL      = ['1HCF.', '30MGCP.']
    prefixL      = ['30MGCP.']
    postfix      = '.asc.gz'

    prodInfoD         = {}
    prodInfoD['1HCF'] = 'Ratio between the hourly gauge adjusted radar and hourly radar only'
    prodInfoD['GCP']  = 'Gauge corrected precipitation product'
    
    for indx, folderName in enumerate(folderL):
        dirnn = dirn + folderName
        # creat file name list
        datetime_L, datetime_strL, fileFormatL = get_datelist(sdate, edate, dtime, prefixL[indx],
                                              postfix, dirnn)
        infoD  =  collect_info4file(fileFormatL[indx]) 
        infoD  =  augment_infoD(infoD, datetime_L, datetime_strL, folderL[indx])
        
        # 'NODATA_value', 'nrows', 'yllcenter', 'cellsize', 'xllcenter', 'ncols'
        # 'NODATA_value', 'nrows', 'yllcenter', 'cellsize', 'xllcenter', 'ncols', 'dateL', 'datestrL', 'folderName'
        
        finalM =  collect4conca(fileFormatL)      #only 2 files (fileFormatL,2)
        callSave(indx, sdate, infoD, varN = folderL[indx])  #save the metadata and data

    # %%
 
    # Load the data from h5
    dirloc = '/home/albayrak/Documents/prog/twitter/twitter_json/code/MRMS/data/'
    h5fn   = dirloc + 'GCP_2017_03_14.h5'  
    metafn = dirloc + 'GCP_2017_03_14.p'
    infoD  = loadFromMeta(metafn)
    
    indx = find_MRMStimeIndex('2017-03-14T23:30:00.000Z', infoD)
    k = str2datetime(['2017-03-14T23:30:00.000Z'])[0]
    
    # %%   
    
    import h5py
    import numpy as np
    f = h5py.File(h5fn, "r")   
    for DS in f:
        print(DS,"--->",f[DS])
    finalM = f['GCP'][...]
    f.close()
    
    # %%
    
    lon, lat = get_latR_lonRandmash()
    
    #manhattanLat =  39.95   # 40.758 
    #manhattanLon = -75.1652 #-73.985
    manhattanLat =  40.758 
    manhattanLon = -73.985

    i,j = getGriIndx(lat,lon,manhattanLat,manhattanLon)
    
    latRange[i]
    lonRange[j]

    lat[i,j]
    lon[i,j]

    finalM[:,i,j]






#aa = str( datetime_strL[0] )
# collect information such as 

    #read example file 
    filename1 = fileFormatL[49]
    filename2 = fileFormatL[1]
    mydataM1 = np.genfromtxt(filename1, skip_header=6)

    
    latRange = np.arange(54.995, 20, -0.01)
    lonRange = np.arange(-129.995, -60.0, 0.01)
    nhh      = 48 #; Number of Half-Hours in a day
    
    lon, lat = np.meshgrid(lonRange, latRange)
    mydataM1 = ma.masked_where(finalM[40,:,:] < 0.01, finalM[40,:,:])
    lon      = ma.masked_where(finalM[40,:,:]< 0.01, lon)
    lat      = ma.masked_where(finalM[40,:,:] < 0.01, lat)

    #mydataM1 = ma.log(mydataM1)
    #plotRMRS_folium(mydataM1)         # plot with folium
    #plotRMRS_basemap(mydataM1, lat, lon)  # plot with base map



# %%

    for indx, filename in enumarate(fileFormatL):
        if indx == 0:
            finalM = np.genfromtxt(filename1, skip_header=6)
        else:
            mydataM1 = np.genfromtxt(filename1, skip_header=6)
            finalM   = np.concatenate((finalM, mydataM2), axis=0)
            stop



#, edgecolors='none'
    
    
    
# %%

        


    mydataM1 = ma.masked_where(mydataM1 < -998.0, mydataM1)
    mydataM2 = ma.masked_where(mydataM2 < -998.0, mydataM2)
    kk = np.concatenate((kk, mydataM2), axis=0)

    hdf5_path = "my_compressed_data.hdf5"









'''

pro get_level3_mrms_30m, file, data, px1,px2, py1,py2

#; read the level3 30-m MRMS file
#;

   nhd=6  ; 1st 6 lines are headers

   nx=px2-px1+1
   ny=py2-py1+1
  
   data=fltarr(nx,ny) & data[*,*]=-9.99

   openr,lun_data, file, /get_lun,/compress

   #;print,'Reading '  +file
   line=''
   nline=0L
   while (not eof(lun_data)) do begin
        nline++
        readf,lun_data,line
    
        if (nline le nhd) then begin
            a=strsplit(line,/extract)
            #;help,a, line
            if nline eq 1 then ncol=a[1]*1
            if nline eq 2 then nrow=a[1]*1
            if nline eq 3 then x11=a[1]*1.
            if nline eq 4 then y11=a[1]*1.
            if nline eq 5 then res=a[1]*1.
            if nline eq 6 then missing=a[1]/100.
        endif

        if (nline ge py1+nhd and nline le py2+nhd) then begin
            j=nline-py1-nhd 
            a=strsplit(line,/extract)

            data[0:nx-1, j]=a[px1:px2]*1.
        endif   
    endwhile

    idx=where(data lt 0, n)
    if n gt 0 then data[idx]=-9.99

    free_lun,lun_data

end

# ______________________________________________________________



function adjust_00z_to_24z, year,month,day,ihh
#;
#; This is to adjust 0000Z of Day-2 to 2400Z of Day-1
#;    
#; ihh: half-hour conuter
    
    h=ihh/2
    if ihh mod 2 eq 1 then mn='30' else mn='00'
    d=day
    m=month
    y=year

    if (ihh eq 0) then begin
        h=24
        d=day-1
        if(d eq 0) then begin
            m=month-1
            if(m eq 0) then begin  
             #; 00Z of Jan 1, adjust to 24Z of Dec 31 of previous yr
                y=year-1
                m=12 
                d=31
            endif else begin
                days_in_mon,y,m,nday
                d=nday
            endelse
        endif
    endif

    h=string(h,format='(i02)')
    d=string(d,format='(i02)')
    m=string(m,format='(i02)')
    y=string(y,format='(i04)')

    ymdh= y+'/'+m+'/'+d+'  '+h+':'+mn+'Z'

    return, ymdh

end

# ________________________________________________________


function get_numrecords,file,HEADER=header

;
; *** Determine and return the number of records in a given file
; *** Note: header = number of records to skip before ingesting data
;    

if(NOT KEYWORD_SET(header)) then header=0
spawn,'wc -l ' + file, result    
a = strsplit(result,' ',/extract)             
nrecs = a(0)*1L - header
;    print,'There are ',nrecs,' records in ',file
    
return,nrecs    

end

# ________________________________________________________


function merge_mrms_daily_to_yearly,year,month

; to run:  IDL> xx=merge_mrms_daily_to_yearly('2013','11')

;
; *** Program to merge daily MRMS files into a yearly file
;
!QUIET=1

 bmon=1
;if (year eq '2013') then bmon=10 else bmon=1
;    
;    year  = string(year, format='(I4.4)')
    mrms_yearlyfile =  'RainData/MRMS_QCed/'+year+'/MRMS.'+year+'.30m.grid'
    ;if (file_test (mrms_yearlyfile) eq 1) then spawn,'rm ' + mrms_yearlyfile
    openw,out_unit, mrms_yearlyfile, /get_lun  ;,/append

    rec = ''

for imonth=bmon, month*1 do begin

    smon=string(imonth,format='(i02)')
    the_dir = 'RainData/MRMS_QCed/'+year+'/'+smon+'/'
    wc = the_dir + 'Rain.'+year+smon+'*'
    files = file_search(wc,COUNT=nf)

;
; *** Cat the contents of these files to a yearly file 
;
    for i=0,nf-1 do begin
        file = files[i]

        print, 'Merging '+file+' to '+ mrms_yearlyfile
;
; *** Check to see how many records are in this file.  If it is 0
; *** (ignoring header line), then skip and move on.
;
        nr = get_numrecords(file)
        if(nr eq 0) then goto,next_file
        openr,unit,file,/get_lun      
        
        while(NOT EOF(unit)) do begin
            readf,unit,rec
            printf,out_unit,rec
        endwhile
        free_lun,unit

        next_file:

     endfor 

endfor  ;month


free_lun,out_unit

return,mrms_yearlyfile

end



# ________________________________________________________


;;;;;;;;;;;; main ;;;;;;;;;;;;;;;;;; 
pro read_mrms_30m_hfd, year, bmon, emon

; Get level-3 MRMS data (half-hourly rainfall) over HalfDegGauge Network
; Use RQI>=80; 0.5<=ratio<=2. and ratio=0.; no snow
; Don't use typeM, which seems to all be set to -999.99

;prod='GCP RQI  TYPE00  TYPE01  TYPE02  TYPE03  TYPE04  TYPE06  TYPE07  TYPE10  TYPE91  TYPE96  TYPEM

;year='2015' & bmon='01' & emon='01'

; 30-m data available from 31 May 2014 23:30Z
; Missing Jan-May 2016

; bad file: 1HCF.20151109.023000.asc.gz

Trqi = 80.
Tr1 =0.5
Tr2 =2.
Tlq=0.0001 ; Liquid precipitation only &  no solid preciptation (type:03,04,07)


; NW corner of NMQ data or pixel(0,0)
;lat00=55.005
lat00=54.995
lon00=-129.995


area=[-75.9, 38.4, -75.3, 37.9]   ; gauge area
; 1st grid is at NW corner of the study area
px1= ceil((area[0]-lon00)*100)
px2=floor((area[2]-lon00)*100)
py1= ceil((lat00-area[1])*100)
py2=floor((lat00-area[3])*100)

;print,px1,py1,px2,py2
nx=px2-px1+1
ny=py2-py1+1
nhh=48 ; Number of Half-Hours in a day

for imonth=bmon*1,emon*1 do begin
   month=string(imonth, format='(i02)')
   days_in_mon,year,imonth,ndays
 
   in_dir='/d2/ftp/pub/wang/NMQ/level3-30m/'+year+'/'+month+'/'

   out_dir='RainData/MRMS_QCed/'+year+'/'+month+'/'
   if(file_test(out_dir,/directory) ne 1) then file_mkdir,out_dir
 
 ;  for iday=1,ndays do begin 
   day1=1
   ;if  month eq '11' then day1=9 else day1=1
   for iday=day1,ndays do begin 

      rain=fltarr(nx,ny,nhh)
      rain[*,*,*]=-9.99
      rqi=rain &  ratio=rain
      type03=rain & type04=rain & type07=rain & typeM=rain

      sday=string(iday,format='(i02)') 
      print, year+month+sday
      wc = in_dir + 'GCP/30MGCP.'+year+month+sday+'*asc.gz'
      files = file_search(wc, COUNT=nf)

      for ifl=0,nf-1 do begin
      ;for ifl=0,1 do begin
         rain_file=files[ifl]
         rain_file_base=file_basename(rain_file,'.gz')
         a=strsplit(rain_file_base,'.',/extract,count=nc)
         ymdh=a[1]+'.'+a[2]
         day=strmid(a[1],6,2)
         hr=strmid(a[2],0,2)
         mn=strmid(a[2],2,2)
         fn=hr*2+mn/30       ; File Number 0,1,...47 <-- 0000Z,0030Z,....., 2330Z
   
         print, 'Reading '+a[1]+'.'+a[2] +'    FN= ',fn
         get_level3_mrms_30m, rain_file, data, px1,px2, py1,py2
         rain[*,*,fn]=data

         rqi_file=in_dir+'RQI/30MRQI.'+ymdh+'.asc.gz'
         if (file_test(rqi_file)) then begin
            get_level3_mrms_30m, rqi_file, data, px1,px2, py1,py2
            rqi[*,*,fn]=data
         endif else begin
            rqi[*,*,fn]=-9.99
         endelse

         ;idx=where(rqi[*,*,fn] ge Trqi, nrqi) & print,'n=',nrqi

         ratio_file=in_dir+'1HCF/1HCF.'+ymdh+'.asc.gz'
         if (file_test(ratio_file)) then begin
            get_level3_mrms_30m, ratio_file, data, px1,px2, py1,py2
            ratio[*,*,fn]=data
         endif else begin
            ratio[*,*,fn]=-9.99 
         endelse
          
;idx=where((ratio[*,*,fn] ge Tr1 and ratio[*,*,fn] le Tr2) or (ratio[*,*,fn] eq 0), nratio) & help,nratio

         type03_file=in_dir+'TYPE03/30MTYPE03.'+ymdh+'.asc.gz'
         if (file_test(type03_file)) then begin
            get_level3_mrms_30m, type03_file, data, px1,px2, py1,py2
            type03[*,*,fn]=data
         endif else begin
            type03[*,*,fn]=-9.99
         endelse
 
         type04_file=in_dir+'TYPE04/30MTYPE04.'+ymdh+'.asc.gz'
         if (file_test(type04_file)) then begin
            get_level3_mrms_30m, type04_file, data, px1,px2, py1,py2
            type04[*,*,fn]=data
         endif else begin
            type04[*,*,fn]=-9.99
         endelse 
     
         type07_file=in_dir+'TYPE07/30MTYPE07.'+ymdh+'.asc.gz'
         if (file_test(type07_file)) then begin
            get_level3_mrms_30m, type07_file, data, px1,px2, py1,py2
            type07[*,*,fn]=data
         endif else begin
            type07[*,*,fn]=-9.99
         endelse 

         ;typeM_file=in_dir+'TYPEM/30MTYPEM.'+ymdh+'.asc.gz'
         ;if (file_test(typeM_file)) then begin
         ;   get_level3_mrms_30m, typeM_file, data, px1,px2, py1,py2
         ;   typeM[*,*,fn]=data
         ;endif else begin
         ;   typeM[*,*,fn]=-9.99
         ;endelse 

      endfor  ;30-m files

      ;
      ; Write out half-hourly data (QCed gauge-corrected Liquid radar rain only)
      ;

      fmt='('+strtrim(string(nx),2)+'f6.2)'

      out_file=out_dir+'Rain.' +year+month+sday+'.30m' 
      openw,lun_30m,out_file,/get_lun 

      for ihh=0,nhh-1 do begin
         ymdh=adjust_00z_to_24z(year,month,sday,ihh)           
         printf,lun_30m, ymdh
         
         ; get bad data, set them to -9.99
         idx= where( (rqi[*,*,ihh] lt Trqi) or $
                ( (ratio[*,*,ihh] lt Tr1 and ratio[*,*,ihh] ne 0) or (ratio[*,*,ihh] gt Tr2) ) $
                 or (type03[*,*,ihh] gt Tlq) or (type04[*,*,ihh] gt Tlq), n)
                    ;    or (typeM[*,*,ihh]  gt Tlq or typeM[*,*,ihh]  lt -9) , n)
         ;print, ymdh+"  Good data pixels :", nx*ny-n
         r=rain[*,*,ihh]
         if n gt 0 then r[idx]=-9.99
         for iy=0,ny-1 do begin
              printf,lun_30m,r[*,iy],format=fmt
         endfor
      endfor
          
      free_lun,lun_30m

   endfor    ; day


endfor       ; month

; 
; Merge daily files to a yearly file
;

mrms_yearlyfile=merge_mrms_daily_to_yearly(year,emon)

;stop


end

 '''
 
