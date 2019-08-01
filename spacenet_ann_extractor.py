import os
import numpy as np
import pandas as pd
import skimage.io
from skimage import exposure
import rasterio
import cv2
import matplotlib.pyplot as plt
import json
import geojson
import re
import time
from tqdm import tqdm
from osgeo import gdal 


def load_polypixel_coordinates(folder_path, scene_filename, img_no, bldg_id):
    
    """
    Load pixel coordinates as a list of list from the csv file
    
    Returns polygon pixel coordinates in the following format:
    [[484.768,
      900.0,
      485.379,
      898.427,
      490.289,
      895.327,
      498.722,
      894.38,
      499.076,
      898.811,
      499.346,
      900.0,
      484.768,
      900.0]]
      
    coco_seg
    
    [[484.768, 900.0],
     [485.379, 898.427],
     [490.289, 895.327],
     [498.722, 894.38],
     [499.076, 898.811],
     [499.346, 900.0],
     [484.768, 900.0]]
    
    
    """
    
    scene = scene_filename
    img_id = scene + "_" + img_no
    df = pd.read_csv(os.path.join(folder_path, scene + "_Train.csv")) 

    # Drop existing index column
    if 'Unnamed: 0' in df.columns:
    	df = df.drop(['Unnamed: 0'],axis=1)

    # Get number of buildings in the image 
    bldg_count = len(df.loc[df['ImageId'] == img_id])
    
    
    # Get pixel coordinates of the building
    s = str(df.loc[df['ImageId'] == img_id][df['BuildingId'] == bldg_id]['PolygonWKT_Pix'].values)
    #s = s[12:-4] # Strip out POLYGON and other stuff
    s = re.search("\((.+?)\)",s).group(0)[2:-1] # Strip out POLYGON and other stuff
    
    #print(s)
   
    bldg_pix_co = [[float(i) for i in x.split()] for x in s.split(',')]
    #coco_seg = [[x for t in bldg_pix_co for x in t]]
    
    return bldg_pix_co
	
	
def get_bbox_coordinates(polypix,bbox_format="XYWH"):
    """
    Get bbox coordinates from the minimum rectangle that encloses the polygon i.e. get the bbox of a single building in the scene
    
    polypix     : Polygon Pixels: list of lists format - #bldgpixco 
    bbox_format : XYWH  -> Displays bbox coordinates in the format    : [xmin, ymin, width, height]
                : LTRB  -> Displays bbox coordinates in the format    : [xmin, ymax, xmax, ymin]
                : CWH  -> Displays bbox coordinates in the format    : [xcenter, ycenter, width, height]
                : all   -> Displays all bbox coordinates in the format: [(x1,y1),(x2,y2),(x3,y3),(x1,y1)]
    
    Returns bbox coordinates of the format: [xmin,ymin,width,height]
    """

    gpbc = np.array(poly2rect(polypix))
  
    xmin = np.min(gpbc[:,0])
    xmax = np.max(gpbc[:,0])
    ymin = np.min(gpbc[:,1])
    ymax = np.max(gpbc[:,1])

    xcenter = (xmin + xmax)/2
    ycenter = (ymin + ymax)/2

    width = round(xmax - xmin,3)
    height = round(ymax - ymin,3)
    
    if bbox_format == "XYWH":
        bbox_coordinates = [xmin, ymin, width, height]
    elif bbox_format == "LTRB":
        bbox_coordinates = [xmin, ymax, xmax, ymin]
    elif bbox_format == "CWH":
        bbox_coordinates = [xcenter, ycenter, width, height]
    elif bbox_format == "all":
        bbox_coordinates = [gpbc[0][0],gpbc[0][1],gpbc[1][0],gpbc[1][1],gpbc[2][0],gpbc[2][1],gpbc[3][0],gpbc[3][1]]

    return bbox_coordinates
	
def poly2rect(polypixlist):
    """
    Converts polygon coordinates to minimum bounding rectangle coordinates
    
    polypixlist : polygon coordinates (list of lists) - #bldgpixco
    Returns : minimum bounding rectangle coordinates (list of lists)
    
    
    For example,
    
    polypixlist = [[484.768, 900.0],
                   [485.379, 898.427],
                   [490.289, 895.327],
                   [498.722, 894.38],
                   [499.076, 898.811],
                   [499.346, 900.0],
                   [484.768, 900.0]]
                   
    Returns value  = [[499.3460000000001, 894.38],
                      [484.7680000000001, 894.38],
                      [484.7680000000001, 900.0],
                      [499.3460000000001, 900.0]]
    
    """

    geom = polypixlist
    mabr = minimum_bounding_rectangle(np.array(geom))
    return mabr.tolist()
    
def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    TODO: Incorporate orientation

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    
    from scipy.spatial import ConvexHull
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval
	
	
def PolyArea(x,y):
    """
    Calculates area of the polygon
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
	
	
	

	
def coco_ann(imgcat_id, bldg_set_id,bldg_id,bldg_dir, bldg_set_file,category_id, polypix,obj_id):
    
    """
    Returns a dict of annotations in the MS COCO format
    """
    
    # Area
    p = np.array(polypix)
    x = p[:,0]
    y = p[:,1]
    area = PolyArea(x,y)
    
    # BBox Coordinates
    bbox = get_bbox_coordinates(polypix)
    
    id_sc = obj_id#imgcat_id
    image_id = imgcat_id + "_" + bldg_set_id
    iscrowd = 0
    category_id = category_id
    
    poly_segmentation = [[x for t in polypix for x in t]]
     
    
    ann = {'area': area,
        'bbox': bbox,
        'category_id': category_id,
        'id': id_sc,
        'image_id': image_id,
        'iscrowd': 0,
        'segmentation': poly_segmentation}
    
    return ann
	

def get_img_info(img_dir, img_files):
    
    iminfo = []

    for filename in img_files:
        iminfo.append({'coco_url': 'aws s3 cp s3://spacenet-dataset/Spacenet_Off-Nadir_Dataset/SpaceNet-Off-Nadir_Sample/SpaceNet-Off-Nadir_Sample.tar.gz',
            'date_captured': '2013-11-14 11:18:45',
            'file_name': filename,
            'flickr_url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg',
            'height': 900,
            'width': 900,
            'id': filename[:-4],
            'license': 3})
    return iminfo


# def gtif2jpg(src_dir_name,dest_dir_name,adapt_hist_eq=False):
    
#     """
#     Converts the satellite images from geotif to jpg
    
#     """

#     ROOT_DIR = os.getcwd()    
#     os.mkdir(ROOT_DIR + os.sep + "{}".format(dest_dir_name))
#     img_path = os.path.join(ROOT_DIR,src_dir_name)
#     jpg_path = os.path.join(ROOT_DIR,dest_dir_name)
#     img_dirs = [im[1] for im in os.walk(img_path)][0]
    
    
#     for each_img_dir in img_dirs:
#         os.mkdir(jpg_path + os.sep + each_img_dir)
#         dir_path = os.path.join(jpg_path,each_img_dir)
#         os.mkdir(dir_path + os.sep + "Pan-Sharpen")
#         dest_path = os.path.join(dir_path, "Pan-Sharpen")
#         print("Converting " + each_img_dir + " files")
#         img_files = [i for ind in os.walk(os.path.join(img_path,each_img_dir,"Pan-Sharpen")) for i in ind[2] if bool(re.search('.aux.xml',i)) is False and bool(re.search('.tif',i)) is True]
        
#         for each_img in tqdm(img_files):
#             dest = os.path.join(dest_path, each_img[:-4] +".jpg")
#             src = os.path.join(img_path,each_img_dir, "Pan-Sharpen", each_img)

#             raster = rasterio.open(src)

#             # Normalize bands into 0.0 - 1.0 scale
#             def normalize(array):
#                 array_min, array_max = array.min(), array.max()
#                 return ((array - array_min)/(array_max - array_min))

#             # Convert to numpy arrays
#             red = raster.read(3)
#             green = raster.read(2)
#             blue = raster.read(1)

#             # Normalize band DN
#             redn = normalize(red)
#             greenn = normalize(green)
#             bluen = normalize(blue)

#             # Stack bands
#             nrg = np.dstack(( redn, greenn, bluen))

#             data = 255 * nrg # Now scale by 255
#             img = data.astype(np.uint8)

#             if adapt_hist_eq == False:
#                 cv2.imwrite(dest, img)
#             elif adapt_hist_eq == True:
#                 img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.02)
#                 skimage.io.imsave(dest, img_adapteq)


#             # gdal.Translate(destName=dest, 
#             #                srcDS=src, 
#             #                options=translate_options)


#     print("Conversion Complete")


def tifconv(src_dir_name,dest_dir_name,ext_choice='.jpg'):

    dest_dir_name = "images_jpg_gdalver"

    """
    Converts satellite images from geotif to jpg format
    """

    ROOT_DIR = os.getcwd()    
    os.mkdir(ROOT_DIR + os.sep + "{}".format(dest_dir_name))
    img_path = os.path.join(ROOT_DIR,src_dir_name)
    jpg_path = os.path.join(ROOT_DIR,dest_dir_name)
    img_dirs = [im[1] for im in os.walk(img_path)][0]
    

    for each_img_dir in img_dirs:
        os.mkdir(jpg_path + os.sep + each_img_dir)
        dir_path = os.path.join(jpg_path,each_img_dir)
        os.mkdir(dir_path + os.sep + "Pan-Sharpen")
        dest_path = os.path.join(dir_path, "Pan-Sharpen")
        print("Converting " + each_img_dir + " files")
        img_files = [i for ind in os.walk(os.path.join(img_path,each_img_dir,"Pan-Sharpen")) for i in ind[2] if bool(re.search('.aux.xml',i)) is False and bool(re.search('.tif',i)) is True]
        
        for each_img in tqdm(img_files):
            dest = os.path.join(dest_path, each_img[:-4] + ext_choice)
            src = os.path.join(img_path,each_img_dir, "Pan-Sharpen", each_img)
            
            gimg = gdal.Open(src)
            scale=[]
            for i in range(3):
                arr = gimg.GetRasterBand(i+1).ReadAsArray()
                scale.append([np.percentile(arr, 1), np.percentile(arr, 99)])

            gdal.Translate(dest, src, options=gdal.TranslateOptions(outputType=gdal.GDT_Byte, scaleParams=scale))

    print("Conversion Complete")



	
def main():
    # For all objects in all images

    ROOT_DIR = os.getcwd()

    train_path = os.path.join(ROOT_DIR,"summaryData")
    bldg_dir = os.path.join(ROOT_DIR,"geojson","spacenet-buildings")
    img_path = os.path.join(ROOT_DIR,"images")

    #img_dir = os.path.join(ROOT_DIR,imgcat_id,"Pan-Sharpen") 

    train_files = [t[2] for t in os.walk(train_path)][0]
    bldg_files = [b[2] for b in os.walk(bldg_dir)][0]
    img_dirs = [im[1] for im in os.walk(img_path)][0]


    print("Number of scenes = ",len(img_dirs))
    print("\n")

    space_coco_ann = {}
    space_ann = []


    ###########################################

    # Convert tif images to jpg images

    #dest_dir_name = "images_jpg_train"
    dest_dir_name = "images_jpg_gdalver"
    src_dir_name = "images"

    print("Existence of JPEG directory:",os.path.isdir(ROOT_DIR + os.sep + dest_dir_name))
    print(ROOT_DIR + os.sep + dest_dir_name)

    # Check if jpeg image directory exists
    if os.path.isdir(ROOT_DIR + os.sep + dest_dir_name) is False:
        #gtif2jpg(src_dir_name,dest_dir_name,adapt_hist_eq=True)
        tifconv(src_dir_name,dest_dir_name,ext_choice='.jpg')
    else:
        print("JPEG image directory exists")

    k1 = str(input("Press Y to continue:"))
    if k1.lower() != 'y':
        exit()

    img_path = os.path.dirname(img_path)
    img_path = os.path.join(img_path,dest_dir_name)


    # Rename images from Pan-Sharpen_Atlanta_nadir10_catid_1030010003CAF100_740801_3726489.jpg to Atlanta_nadir10_catid_1030010003CAF100_740801_3726489.jpg
    print("Renaming Images\n")
    print("WARNING: Currently renaming images only aims to cut out the first 12 chars of the name. It's intended to mainly convert images with names like Pan-Sharpen_Atlanta_nadir10_catid_1030010003CAF100_740801_3726489.jpg to Atlanta_nadir10_catid_1030010003CAF100_740801_3726489.jpg \n")

    rc = str(input("Do you want to rename images? (Y/n)"))

    if rc.lower() == "y":
        for path, subdirs, files in os.walk(img_path):
            for f in files:
                if f.endswith('.jpg'):
                #if f.endswith('.tif'):
                    # Cutting out the Pan-Sharpen_ part of the name
                    os.rename(os.path.join(path,f),os.path.join(path,f[12:]))
    else:
        print("No Renaming")

    print("COCO style annotation generation")
    k2 = str(input("Press Y to continue:"))
    if k2.lower() != 'y':
        exit()

    obj_id = 900000 # Annotation ID


    img_files = [i for ind in os.walk(img_path) for i in ind[2] if i.endswith('.jpg')]

    print("Loading COCO style annotations for",len(img_files),"images")

    for each_img in tqdm(img_files):
        img_id = each_img[:-4] # Atlanta_nadir44_catid_1030010003CCD700_745301_3733239
        bldg_set_id = img_id[-14:]#re.search("_\d+_\d+",img_id).group()[1:] # 745301_3733239
        imgcat_id = img_id[0:-15]#re.sub(bldg_set_id, '', img_id)[:-1] # Atlanta_nadir44_catid_1030010003CCD700
            
        # Get the correct train csv file
        train_csv = imgcat_id + "_Train.csv"

            
        # Load the dataframe
        df = pd.read_csv(os.path.join(train_path, train_csv))
            
        # Drop existing index column
        if 'Unnamed: 0' in df.columns:
        	#print("Removing Index Column")
        	df = df.drop(['Unnamed: 0'],axis=1)

        #print(imgcat_id+"_"+bldg_set_id)
            
        df = df[df['ImageId']== img_id]
            
        for index, row in df.iterrows():
            obj_id += 1
            dfimid = row['ImageId'] # Atlanta_nadir44_catid_1030010003CCD700_743051_3735939
            bldg_set_id = dfimid[-14:]#re.search("_\d+_\d+",dfimid).group()[1:] # 743051_3735939
            bldg_set_file = 'spacenet-buildings_' + bldg_set_id + '.geojson'
            dfbid = row['BuildingId'] # 0
                
            dfpolypix = load_polypixel_coordinates(train_path,imgcat_id,bldg_set_id,dfbid)
                
            if (df.ImageId == imgcat_id + "_" + bldg_set_id).any() == True:
                # Buildings present. Set category_id of annotation to building/1
                # Go through each building and update annotation
                category_id = 1
            else:
                # No building present. Set category_id of annotation to BG/0
                category_id = 0
                    
            ann = coco_ann(imgcat_id,bldg_set_id,dfbid,bldg_dir,bldg_set_file,category_id,dfpolypix,obj_id)
            space_ann.append(ann)

        #with open(each_img +'.json', 'w') as fw:
        #	json.dump(space_ann, fw) 


    ###########################################

    # Write annotations to file
    space_coco_ann['annotations'] = space_ann
    space_coco_ann['info'] = {'description': 'SpaceNet Off Nadir Dataset',
            'url': 'https://spacenetchallenge.github.io/datasets/spacenet-OffNadir-summary.html',
            'version': '1.0',
            'year': 2009,
            'contributor': 'SpaceNet',
            'date_created': '2009/12/22'}
    space_coco_ann['categories'] = [{'supercategory': 'building', 'id': 1, 'name': 'building'}] 
    space_coco_ann['images'] = get_img_info(img_path,img_files)



    with open('space_coco_annotations.json', 'w') as fp:
        json.dump(space_coco_ann, fp)      


        
if __name__ == '__main__':
    main()
