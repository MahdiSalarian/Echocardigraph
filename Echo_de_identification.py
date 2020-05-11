
# coding: utf-8

# In[1]:


import pydicom
import os
import numpy as np
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir
from os.path import dirname, join
from pprint import pprint


# In[59]:


get_testdata_files('/Users/Engin/Documents/Edwards/Projects/Echo/data/87568/DICOMDIR')


# In[15]:


dicom_dir


# In[17]:


for patient_record in dicom_dir.patient_records:
    if (hasattr(patient_record, 'PatientID') and
            hasattr(patient_record, 'PatientName')):
        print("Patient: {}: {}".format(patient_record.PatientID,
                                       patient_record.PatientName))


# In[18]:


dicom_dir.patient_records


# In[19]:


studies = patient_record.children
# got through each serie
for study in studies:
    print(" " * 4 + "Study {}: {}: {}".format(study.StudyID,
                                              study.StudyDate,
                                              study.StudyDescription))
    all_series = study.children
    # go through each serie
    for series in all_series:
        image_count = len(series.children)
        plural = ('', 's')[image_count > 1]

        # Write basic series info and image count

        # Put N/A in if no Series Description
        if 'SeriesDescription' not in series:
            series.SeriesDescription = "N/A"
        print(" " * 8 + "Series {}: {}: {} ({} image{})".format(
            series.SeriesNumber, series.Modality, series.SeriesDescription,
            image_count, plural))

        # Open and read something from each image, for demonstration
        # purposes. For simple quick overview of DICOMDIR, leave the
        # following out
        print(" " * 12 + "Reading images...")
        image_records = series.children
        image_filenames = [join(base_dir, *image_rec.ReferencedFileID)
                           for image_rec in image_records]

        datasets = [pydicom.dcmread(image_filename)
                    for image_filename in image_filenames]

        patient_names = set(ds.PatientName for ds in datasets)
        patient_IDs = set(ds.PatientID for ds in datasets)

        # List the image filenames
        print("\n" + " " * 12 + "Image filenames:")
        print(" " * 12, end=' ')
        pprint(image_filenames, indent=12)

        # Expect all images to have same patient name, id
        # Show the set of all names, IDs found (should each have one)
        print(" " * 12 + "Patient Names in images..: {}".format(
            patient_names))
        print(" " * 12 + "Patient IDs in images..: {}".format(
            patient_IDs))


# In[39]:


# get the pixel information into a numpy array
data = ds.pixel_array
print('The image has {} x {} voxels'.format(data.shape[0],
                                            data.shape[1]))

data_downsampling = data[::8, ::8]
print('The downsampled image has {} x {} voxels'.format(
    data_downsampling.shape[0], data_downsampling.shape[1]))
ds.PixelData = data_downsampling.tobytes()
# update the information regarding the shape of the data array
ds.Rows, ds.Columns = data_downsampling.shape

# print the image information given in the dataset
print('The information of the data set after downsampling: \n')
print(ds)


# In[44]:


import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

# load the DICOM files
files = []
print('glob: {}'.format(sys.argv[1]))
for fname in glob.glob(sys.argv[1], recursive=False):
    print("loading: {}".format(fname))
    files.append(pydicom.dcmread(fname))

print("file count: {}".format(len(files)))

# skip files with no SliceLocation (eg scout views)
slices = []
skipcount = 0
for f in files:
    if hasattr(f, 'SliceLocation'):
        slices.append(f)
    else:
        skipcount = skipcount + 1

print("skipped, no SliceLocation: {}".format(skipcount))

# ensure they are in the correct order
slices = sorted(slices, key=lambda s: s.SliceLocation)

# pixel aspects, assuming all slices are the same
ps = slices[0].PixelSpacing
ss = slices[0].SliceThickness
ax_aspect = ps[1]/ps[0]
sag_aspect = ps[1]/ss
cor_aspect = ss/ps[0]

# create 3D array
img_shape = list(slices[0].pixel_array.shape)
img_shape.append(len(slices))
img3d = np.zeros(img_shape)

# fill 3D array with the images from the files
for i, s in enumerate(slices):
    img2d = s.pixel_array
    img3d[:, :, i] = img2d

# plot 3 orthogonal slices
a1 = plt.subplot(2, 2, 1)
plt.imshow(img3d[:, :, img_shape[2]//2])
a1.set_aspect(ax_aspect)

a2 = plt.subplot(2, 2, 2)
plt.imshow(img3d[:, img_shape[1]//2, :])
a2.set_aspect(sag_aspect)

a3 = plt.subplot(2, 2, 3)
plt.imshow(img3d[img_shape[0]//2, :, :].T)
a3.set_aspect(cor_aspect)

plt.show()


# In[46]:


for record in onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


# In[690]:


import cv2
def convert_ybr_to_rgb(arr):
    if len(arr.shape) == 4:
        return np.vstack([convert_ybr_to_rgb(a)[np.newaxis] for a in arr])
    else:
        temp = arr[..., 1].copy()
        arr[..., 1] = arr[..., 2]
        arr[..., 2] = temp
        return cv2.cvtColor(arr, cv2.COLOR_YCR_CB2RGB)


def get_pixel_array_rgb(ds):
#     try:
#         dk=ds.pixel_array
        if ds.PhotometricInterpretation in ['YBR_FULL', 'YBR_FULL_422']:
            return convert_ybr_to_rgb(ds.pixel_array)
#         else: dk
#     except:
        return None
                


def load_records(path):
    files=os.listdir(path)
    records = [pydicom.dcmread(path + '/' + s) for s in files]
    return records,files


# In[691]:


# filename='/Users/Engin/Documents/Edwards/Projects/Echo/data/87625/IMAGES/IM40'
# db = pydicom.dcmread(filename)
# data=get_pixel_array_rgb(db)


# In[667]:


rec[0x0008,0x1090].keyword


# In[657]:


rec


# In[670]:


dd=rec.pixel_array


# In[675]:


import pandas as pd
lst = ['filename','Model Name', 'Manufacture', 'Photometric Interpretation', 'x0', 'x1', 'y0', 'y1']

df = pd.DataFrame(columns=lst)

print(df)

folder_path='/Users/Engin/Documents/Edwards/Projects/Echo/data/87709'
filepath = os.path.join(folder_path,'IMAGES')
records,files=load_records(filepath)
Imgs=[]
feats=[]
for i,rec in enumerate(records):
    file_name=files[i]
    data=get_pixel_array_rgb(rec) 
    
    ind=0
    if len(data.shape)==4:
        ind=1
        print('The image has {} x {} voxels'.format(data.shape[0+ind],data.shape[1+ind]))
        obj =rec.SequenceOfUltrasoundRegions[0]
        x0=obj.RegionLocationMinX0
        x1=obj.RegionLocationMaxX1
        y0=obj.RegionLocationMinY0
        y1=obj.RegionLocationMaxY1
        Model_Name=rec.ManufacturerModelName
        Manufacture=rec.Manufacturer
        Phot0Int=rec.PhotometricInterpretation
        df.append({'filename':file_name,'Model Name':Model_Name,'Manufacture':Manufacture,'Photometric Interpretation':Phot0Int,'x0':x0,
                   'x1':x1,'y0':y0,'y1':y1}, ignore_index=True)
        
        roi.append([x0,x1,y0,y1])
        

        

# slicess = [s for s in slices if 'SliceLocation' in s]


# In[741]:


def show_result(data,inx):
    for j in range(data.shape[0]):
        img=data[j,:,:,:] 
        print(img.shape)
        size1=tuple(int(s/40) for s in img.shape[:2])
        plt.figure(figsize=size1, dpi= 100, facecolor='w', edgecolor='k')
        plt.imshow(img,cmap=plt.cm.bone)
        plt.show()
        plt.figure(figsize=.5*np.array(size1), dpi= 100, facecolor='w', edgecolor='k')
        print(inx[2],inx[3],inx[0],inx[1])
        plt.imshow(img[inx[2]:inx[3],inx[0]:inx[1],:],cmap=plt.cm.bone)
        plt.show()


# In[742]:


'PhotometricInterpretation' in rec


# In[747]:


def convert_color(rec):
    if 'PhotometricInterpretation' in rec :
        if rec.PhotometricInterpretation in ['YBR_FULL', 'YBR_FULL_422']:
                return convert_ybr_to_rgb(rec.pixel_array)
    return rec.pixel_array 
# df = pd.DataFrame(columns=lst)

folder_path='/Users/Engin/Documents/Edwards/Projects/Echo/data/87651/IMAGES'
file_name='IM2'
filepath = os.path.join(folder_path,file_name)
rec=pydicom.dcmread(filepath)
Imgs=[]
feats=[]
data=convert_color(rec)
lst=[]
if data.any:
    print(data.shape)
    if len(data.shape)==3:
        lst.append(data)
    data=np.array(lst)
    print(data.shape)
    print('The image has {} x {} voxels'.format(data.shape[1],data.shape[2]))
    obj =rec.SequenceOfUltrasoundRegions[0]
    x0=obj.RegionLocationMinX0
    x1=obj.RegionLocationMaxX1
    y0=obj.RegionLocationMinY0
    y1=obj.RegionLocationMaxY1
    lst2=[x0,x1,y0,y1]
    print(lst2)
    show_result(data,lst2)
    
# #         Model_Name=rec.ManufacturerModelName
# #         Manufacture=rec.Manufacturer
# #         Phot0Int=rec.PhotometricInterpretation
# #         lst2=[file_name,Model_Name,Manufacture,Phot0Int,x0,x1,y0,y1]



# In[738]:


rec.SequenceOfUltrasoundRegions[0]


# In[739]:


rec.SequenceOfUltrasoundRegions[1]


# In[696]:


rec


# In[689]:


np.array(dd).shape


# In[678]:


filename='/Users/Engin/Documents/Edwards/Projects/Echo/data/87709/IMAGES/IM0'
db = pydicom.dcmread(filename)
data=get_pixel_array_rgb(db)
# data = db.pixel_array
ind=0
if len(data.shape)==4:
    ind=1
print('The image has {} x {} voxels'.format(data.shape[0+ind],data.shape[1+ind]))


# In[661]:


# 


# In[ ]:


# for rec in records:
#     data=get_pixel_array_rgb(rec)
    
#     ind=0
#     if len(data.shape)==4:
#         ind=1
#         print('The image has {} x {} voxels'.format(data.shape[0+ind],data.shape[1+ind]))


# In[618]:


data.shape


# In[619]:


db.PhotometricInterpretation


# In[620]:


# db[0x0018, 0x6011].keyword


# In[621]:


db.SequenceOfUltrasoundRegions[0]


# In[622]:


roi=[]
for obj in db.SequenceOfUltrasoundRegions:
    x0=obj.RegionLocationMinX0
    print(x0)
    x0=obj.RegionLocationMinX0
    x1=obj.RegionLocationMaxX1
    y0=obj.RegionLocationMinY0
    y1=obj.RegionLocationMaxY1
    roi.append([x0,x1,y0,y1])
# i=0
# x0=db.SequenceOfUltrasoundRegions[i].RegionLocationMinX0
# x1=db.SequenceOfUltrasoundRegions[i].RegionLocationMaxX1
# y0=db.SequenceOfUltrasoundRegions[i].RegionLocationMinY0
# y1=db.SequenceOfUltrasoundRegions[i].RegionLocationMaxY1
# print(x0,x1,y0,y1)


# In[623]:


roi


# In[624]:


data.shape


# In[626]:



for j in range(data.shape[0]):
    img=data[j,:,:,:] 
    k=0
    inx=roi[k]
    # img=data

    print(img.shape)
    size1=tuple(int(s/40) for s in img.shape[:2])
    plt.figure(figsize=size1, dpi= 100, facecolor='w', edgecolor='k')
    plt.imshow(img,cmap=plt.cm.bone)
    plt.show()
    plt.figure(figsize=.5*np.array(size1), dpi= 100, facecolor='w', edgecolor='k')
    plt.imshow(img[inx[2]:inx[3],inx[0]:inx[1],:],cmap=plt.cm.bone)
    plt.show()


# In[631]:


len(records)


# In[632]:


records[0]


# In[32]:


import cv2
plt.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
plt.show()


# In[ ]:


print(__doc__)
path='/Users/Engin/Documents/Edwards/Projects/Echo'
# FIXME: add a full-sized MR image in the testing data
filename = get_testdata_files('MR_small.dcm')[0]
ds = pydicom.dcmread(filename)

# get the pixel information into a numpy array
data = ds.pixel_array
print('The image has {} x {} voxels'.format(data.shape[0],
                                            data.shape[1]))
data_downsampling = data[::8, ::8]
print('The downsampled image has {} x {} voxels'.format(
    data_downsampling.shape[0], data_downsampling.shape[1]))

# copy the data back to the original data set
ds.PixelData = data_downsampling.tobytes()
# update the information regarding the shape of the data array
ds.Rows, ds.Columns = data_downsampling.shape

# print the image information given in the dataset
print('The information of the data set after downsampling: \n')
print(ds)

