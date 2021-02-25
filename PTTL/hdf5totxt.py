import numpy as np
import h5py
import pandas as pd
#pd.read_hdf("PTTL/mail/3B-HHR-E.MS.MRG.3IMERG.20180101-S000000-E002959.0000.V06B.RT-H5")


a=[3,4,1,4,2,6,8,4]
ln = len(a)
for i in range(1,ln):
    for j in range(0,i,1):
        if a[i] < a[j]:
            temp = a[i]
            del a[i]
            a.insert(j,temp)
            




# path to input data file in HDF5 format:
hdf5_file = 'PTTL/mail/3B-HHR-E.MS.MRG.3IMERG.20180101-S000000-E002959.0000.V06B.RT-H5'
f=h5py.File(hdf5_file)
list(f)
list(f["Grid"])

def getText(strt,eachY):
        try:
                for eachP in list(eachY):
                        try:
                                strt = getText(strt,eachP)
                        except:
                                strt = strt + "\t"+str(eachY)
        except:
                strt = strt + "\t"+str(eachY)
        return strt

strt = ""
for eachX in list(f["Grid"]):
        z = f["Grid"][eachX]
        strt = strt +"\n\n\n\n"+ eachX+"\n\t"
        for eachY in list(z):
                strt = getText(strt,eachY)

        #         try:
        #                 for eachP in list(eachY):
        #                         strt = strt + "\t"+str(eachP)
        #         except:
        #                 strt = strt + "\t"+str(eachY)

file = open("output.txt","w+")
file.write(strt)
file.close()
print(strt)
# path to output data file in text format:
txt_file = 'WeidemannKahana2016_data.tsv'

# Formats for the columns in the text file:
fmts = ['%s', '%s', '%s', '%s', '%s', '%s', '%s',  '%s', '%s',
        '%s', '%s', '%s','%s', '%s', '%s','%s', '%s', '%s','%s']
len(fmts)
# Names of the columns in the text file:
colnames = ['nv', 'lonv', 'latv', 'time', 'lon', 'lat', 'time_bnds', 'lon_bnds', 'lat_bnds', 'precipitationCal', 'precipitationUncal', 'randomError', 'HQprecipitation', 'HQprecipSource', 'HQobservationTime', 'IRprecipitation', 'IRkalmanFilterWeight', 'probabilityLiquidPrecipitation', 'precipitationQualityIndex']

# Character to delimit the columns:
delimchar = '\t'


np.savetxt(txt_file, h5py.File(hdf5_file)['Grid'],
           fmt=fmts, delimiter=delimchar, header=delimchar.join(colnames))



