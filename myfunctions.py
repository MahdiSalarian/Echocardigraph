
import pydicom
import fnmatch
import os
import numpy as np
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir
from scipy.spatial.distance import cdist
from os.path import dirname, join
from ipywidgets.widgets import * 
import ipywidgets as widgets
import cv2
import collections
from pprint import pprint
import matplotlib.pyplot as plt
import sys
import glob
import csv


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
                

def convert_color(rec):
    if 'PhotometricInterpretation' in rec :
        if rec.PhotometricInterpretation in ['YBR_FULL', 'YBR_FULL_422']:
                return convert_ybr_to_rgb(rec.pixel_array)
    return rec.pixel_array 



def show_result(data,inx):
    cm=data.shape[-1]==3
    for j in range(data.shape[0]):
        img=data[j,:,:,:] if cm else data[j,:,:]
        print(img.shape)
        size1=tuple(int(s/40) for s in img.shape[:2])
        plt.figure(figsize=.5*np.array(size1), dpi= 100, facecolor='w', edgecolor='k')
        plt.imshow(img,cmap=plt.cm.bone)
        plt.show()
        plt.figure(figsize=.5*np.array(size1), dpi= 100, facecolor='w', edgecolor='k')
#         print(inx[2],inx[3],inx[0],inx[1])
        if cm: plt.imshow(img[inx[2]:inx[3],inx[0]:inx[1],:],cmap=plt.cm.bone)           
        else:plt.imshow(img[inx[2]:inx[3],inx[0]:inx[1]],cmap=plt.cm.bone)
        plt.show()


def read_record(filepath,file_name,folder_name):
    rec=pydicom.dcmread(filepath)
    Imgs=[]
    feats=[]
    try:
        data=convert_color(rec)
    except: 
        return False,False,False
    lst=[]
    if data.any:
        b1=0
        if len(data.shape)==2  or (len(data.shape)==3 and data.shape[2]==3):
    #         lst.append(data)
    #         data=np.array(lst)
              print('Record ' + file_name +   ' located in folder ' + folder_name + ' is not a video record')
              return False,False,False
        else:   
#             if len(data.shape) > 2:b1=1
#             print(data.shape)
            print('The image has {} x {} voxels'.format(data.shape[1],data.shape[2]))
            valid=True
            try:
                rec.SequenceOfUltrasoundRegions[0]
                obj =rec.SequenceOfUltrasoundRegions[0]
                x0=obj.RegionLocationMinX0
                x1=obj.RegionLocationMaxX1
                y0=obj.RegionLocationMinY0
                y1=obj.RegionLocationMaxY1
                lst2=[x0,x1,y0,y1]
            except:
                x0,x1,y0,y1=-1,-1,-1,-1
                print('help lines are not available')
            
            Model_Name=rec.ManufacturerModelName
            Manufacture=rec.Manufacturer
            Phot0Int=rec.PhotometricInterpretation
            features=[Model_Name,Manufacture,Phot0Int,x0,x1,y0,y1]
            return data,features,valid
def get_generated_mask(mask_folder):
    
    list1=collections.defaultdict(list)    
    make_model=[]
    generated_folder=[]
    for msk in glob.glob(mask_folder+"/*.png"):
        print(msk)
        split_name=msk.split('/')[-1].split('_')
        model_f="_".join(split_name[0:2])
        make_model.append(model_f)       
        generated_folder.append(split_name[-2])
        mask_recorded = cv2.imread(msk)
        graymask=cv2.cvtColor(mask_recorded, cv2.COLOR_BGR2GRAY) if len(mask_recorded.shape)==3 else mask_recorded
        list1[model_f].append(graymask)
        
        
    return make_model,generated_folder,list1

def get_mask_path(mask_folder):
    makemodel_path={}
    for msk in glob.glob(mask_folder+"/*.png"):
#         print(msk)
        split_name=msk.split('/')[-1].split('_')
        makemodel_path["_".join(split_name[0:2])]=msk
#         generated_folder.append(split_name[-2])
    return makemodel_path
    
#----------------------------------------------------------

CANNY_THRESH_1 = 20
CANNY_THRESH_2 = 200
def find_overlap_area(grayimage,all_masks_rel):
#     plt.figure()
#     plt.imshow(grayimg)
#     for i,Mask_n in enumerate(rel_masks):
#         plt.figure()
#         plt.imshow(Mask_n)
#     all_masks_rel=rel_masks
    edges = cv2.Canny(grayimage, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
#     edges = cv2.erode(edges, None)

    # #-- Find contours in edges, sort by area --
    contour_info = []
    _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    mask_res = np.zeros(edges.shape)
    for i in range(6):
        if contour_info[i][2]> 10:
#         cv2.drawContours(mask_res, [contour_info[i][0]], -1, (255), 2)
          cv2.fillConvexPoly(mask_res, contour_info[i][0], (255))

#     plt.figure()
#     plt.imshow(mask_res,cmap=plt.cm.bone)
    mask_s = cv2.erode(mask_res, np.ones([5,5]), iterations=1)
    mask_s=mask_s.astype('float32') / 255.0 
#     plt.figure()
#     plt.imshow(mask_s,cmap=plt.cm.bone)
    new_edge=edges*(1.0-mask_s)

#     plt.figure()
#     plt.imshow(new_edge,cmap=plt.cm.bone)
    mask_res=new_edge
    # ----
    # # cv2.fillConvexPoly(mask_res, max_contour[0], (255))
    # plt.figure()
    # plt.imshow(mask_res,cmap=plt.cm.bone)

    mask_erode = cv2.erode(mask_res, np.ones([7,7]), iterations=1)
    mask_dial = cv2.dilate(mask_res, np.ones([7,7]), iterations=1)
    mask_dial=mask_dial.astype('float32') / 255.0
    mask_erode = mask_erode.astype('float32') / 255.0  

    # dial_masked=create_dilated_mask(mask,1)
    # dial_masked=dial_masked.astype('float32') / 255.0
    # mask2 = mask.astype('float32') / 255.0  

    border_mask=((1.0-mask_erode)+mask_dial)-1.0
#     plt.figure()
#     plt.imshow(border_mask,cmap=plt.cm.bone)

    # plt.figure()
    # plt.imshow(mask_res)
    mres=border_mask
    mres=mres.astype('float32') / 255.0
    pixel_over=[]

    for i,Mask_n in enumerate(all_masks_rel):

                    mask_e = cv2.erode(Mask_n, np.ones([8,8]), iterations=1)
                    mask_d = cv2.dilate(Mask_n, np.ones([8,8]), iterations=1)
                    mask_d=mask_d.astype('float32') / 255.0
                    mask_e = mask_e.astype('float32') / 255.0 
                    border=((1.0-mask_e)+mask_d)-1.0

    #   when a smaller ROI is selected
    #                 maskout_e = cv2.dilate(Mask_n, np.ones([30,30]), iterations=1) 
    #                 maskin_e = cv2.dilate(Mask_n, np.ones([16,16]), iterations=1)
    #                 maskout_e=maskout_e.astype('float32') / 255.0 
    #                 maskin_e=maskin_e.astype('float32') / 255.0 
    #                 out_border=((1.0-maskin_e)+maskout_e)-1.0
    #                 plt.figure()
    #                 plt.imshow(out_border)
    #                 print( out_border.shape,mres.shape)
    #                 outband= out_border*mres
    #                 wrong_overlap=len(np.where(outband !=0)[0])
    #                 plt.figure()
    #                 plt.imshow(outband)

                    dest_xor= border*mres

                    overlap=len(np.where(dest_xor!=0)[0])
    #                 pixel_over.append(overlap-wrong_overlap)
                    pixel_over.append(overlap)
    #                     print(overlap)
#                     plt.figure()
#                     plt.imshow(dest_xor)
    best_mask_inx=np.argmax(pixel_over)
#     print(max_score,indexx)
    best_m=all_masks_rel[best_mask_inx]
    return pixel_over[best_mask_inx],best_mask_inx

#--------------------------------------------------------------------
# # ROI selection
# Get points on image from mouse click 
MASK_COLOR = (0.0,0.0,1.0)
refPt = []
def click_save(event, x, y, flags, param):
#  grab references to the global variables
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))

# Use selected points by users to create the contour and clean up

def create_contour(im_tst,refPt,thickness,size_d,size_e):
    augm_arc=0
    poly = np.zeros(im_tst.shape[0:2])
#     r_extra=thickness*augm_arc+size_d-size_e
    r_extra=4
    (x1,y1),(x2,y2),(x3,y3),(x4,y4)=refPt[0],refPt[1],refPt[2],refPt[3]
    # Make top line smoother for dialation part.
#     in order to get smooth boundries in top lef and right corners, a term (int(thickness/2)) is added to mid point
#     mid1=int((y1+y3)/2)+int(thickness/2)
#     cv2.line(poly,(x1,mid1),(x3,mid1),(255),thickness)
    lower=int(min(y1,y3))
    cv2.line(poly,(x1,lower),(x3,lower),(255),thickness)
    
    px=((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    py=((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
#------smooth top line----
#     poly[0:mid1,:]=0
#-------------------------
    px,py=int(px),int(py)
    r1=np.sqrt((y2-py)**2 + (x2-px)**2)
    r2=np.sqrt((y4-py)**2 + (x4-px)**2)
    r=max(r1,r2)
    if len(refPt)==5:
        (x5,y5)=refPt[4]        
        r3=np.sqrt((y5-py)**2 + (x5-px)**2)
        r=max(r,r3)
    r-=r_extra      
    pleft=[]
    pleft.append([x2,y2])
    pright=[]
    pright.append([x4,y4])
    points=[]
    for t in np.arange(0.0, 360.0, 0.01):
        xt=int(r*np.cos(t)+px)
        yt=int(r*np.sin(t)+py)
        points.append((xt,yt))
    #     points=np.array(points)
    intersec_l = points[cdist(np.array(pleft), points, metric='euclidean').argmin()]
    intersec_r = points[cdist(np.array(pright), points, metric='euclidean').argmin()]
#     cv2.line(poly,(x1,mid1),intersec_l,(255),thickness) 
#     cv2.line(poly,(x3,mid1),intersec_r,(255),thickness)
    cv2.line(poly,(x1,lower),intersec_l,(255),thickness) 
    cv2.line(poly,(x3,lower),intersec_r,(255),thickness)    
    for pnt in points:
        p1=augm_arc
        if pnt[0] <= intersec_r[0] and pnt[0] >= intersec_l[0] and pnt[1] > min(intersec_l[1],intersec_r[1])-3:
#             if  intersec_r[0] - pnt[0]  < 9 or pnt[0] - intersec_l[0] > 9:       
#                 p1=int(augm_arc/2)
            poly[pnt[1]:pnt[1]+thickness+p1,pnt[0]:pnt[0]+thickness+p1]=255
            
    # poly[py:py+2,px:px+2]=255
    return poly.astype(np.uint8)


# ----preprocessing on contour and make the mask

def create_mask(edges,MASK_DILATE_ITER,MASK_ERODE_ITER,thickness):

    height , width ,  =  edges.shape[:2]      
#     edges=cv2.cvtColor(poly, cv2.COLOR_BGR2GRAY) if is_color else poly
#     edges = cv2.dilate(edges,np.ones(size_dial), None)
#     edges = cv2.erode(edges,np.ones(size_erod), None)
    
    # Find contours in created shape   
    contour_info = []
    _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]
    mask = np.zeros(edges.shape)
    mask=cv2.fillConvexPoly(mask, max_contour[0], (255))
    # #-- Smooth mask, then blur it --------------------------------------------------------
#     mask = cv2.dilate(mask, np.ones([3,3]), iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, np.ones([thickness,thickness]), iterations=MASK_ERODE_ITER)
#     mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    return mask

def create_dilated_mask(mask,mask_shape,iter1):
    mask_dial = cv2.dilate(mask, np.ones([mask_shape[0],mask_shape[1]]), iterations=iter1)
    return mask_dial

def evaluate_mask(mask2,gray2,thr):
    
    dial_masked=create_dilated_mask(mask2,[15,15],1)
    dial_masked=dial_masked.astype('float32') / 255.0
    
    mask2=create_dilated_mask(mask2,[3,3],1)
    mask2 = mask2.astype('float32') / 255.0  
    border_mask=((1.0-mask2)+dial_masked)-1.0
    
    plt.figure()
    plt.imshow(border_mask,cmap=plt.cm.bone)
    
    masked_border=border_mask*gray2
    
    plt.figure()
    plt.imshow(masked_border,cmap=plt.cm.bone)
    
    res2 = cv2.erode(masked_border,np.ones(5), 1)
    plt.figure()
    plt.imshow(res2,cmap=plt.cm.bone)
    
    res2=255.0*(res2.astype('float32')/np.max(res2))
    num_outliers=len(np.where(res2>thr)[0])
    return num_outliers
    
# ----show mask on image

def create_overlay(im_tst,mask,is_color):
    
    alpha=0.3
    mask_stack =mask if len(mask.shape)==3 else np.dstack([mask]*3) # Create 3-channel mask
    corpimg = im_tst if is_color else np.dstack([im_tst]*3)
    mask_stack  = mask_stack.astype('float32') / 255.0  
    norm_img    = corpimg.astype('float32') / 255.0    
#     masked_img = alpha*(mask_stack * norm_img) + (1-mask_stack) * MASK_COLOR*(1-alpha) # Blend
    masked_img = (mask_stack * alpha) + ((1-alpha) * norm_img) # Blend
#     masked_img = (mask_stack * norm_img) + ((1-mask_stack) * MASK_COLOR) # Blend
    masked_img = (masked_img * 255).astype('uint8') 
#     plt.imshow(masked_img,cmap=plt.cm.bone)
    return masked_img


def apply_mask(im_tst,mask,is_color):
    mask_stack =mask if len(mask.shape)==3 else np.dstack([mask]*3)  # Create 3-channel mask if not
    corpimg = im_tst if is_color else np.dstack([im_tst]*3)
    mask_stack  = mask_stack.astype('float32') / 255.0  
    norm_img    = corpimg.astype('float32') / 255.0
    masked_img= (mask_stack * norm_img) + ((1-mask_stack) * MASK_COLOR)
#   masked_img  masked_img = (mask_stack * norm_img)
    masked_img = (masked_img * 255).astype('uint8') 
#     plt.imshow(masked_img,cmap=plt.cm.bone)
    return masked_img

def show_result(res_mask,overlay_res):
        plt.subplot(1, 2, 1)
        plt.title('Mask and overlay on image') 
        plt.imshow(res_mask,cmap=plt.cm.bone)
        plt.subplot(1, 2, 2)
        plt.imshow(overlay_res,cmap=plt.cm.bone)

        
       
