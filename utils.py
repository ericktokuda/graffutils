#!/usr/bin/env python3
"""Remove duplicate points from via format csv

"""

import os
import argparse
import json
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt 
import json
import imageio
import skimage
import skimage.draw

def args_ok(args):
    for arg in vars(args):
        if getattr(args, arg) == None: return False
    return True

def validate_metadata(metadatadir, coordsfile):
    print('Not downloaded points metadata')

    files =  set(os.listdir(metadatadir))

    fh = open(coordsfile)

    for l in fh:
        lon, lat = l.strip().split(',')
        filename = 'im_{}_{}_0.json'.format(lat, lon)
        if filename not in files:
            print(l.strip())
    fh.close()

def validate_files(imgdir, coordsfile):
    df = pd.read_csv(coordsfile)
    lon = set(df['lon'])
    lat = set(df['lat'])
    for f in os.listdir(imgdir):
        if not f.endswith('.jpg'):
            print('{} not jpeg'.format(f))
            return
        arr = f.split('_')
        flat = float(arr[1])
        flon = float(arr[2])

        if not flat in lat or not flon in lon:
            print(f)
            continue

        #print(len(df[(df['lat'] == flat) & (df['lon'] == flon)]))


def filter_and_save_json(origjson, delkeys, outpath, listpath):
    """Delete unwanted keys and save to file

    Args:
    origjson(dict): json dictionary
    delkeys(list): list of keys to delete
    outpath(str): output path
    """

    mycopy = dict(origjson)
    for entry in delkeys:
        mycopy.pop(entry, None)

    with open(outpath, 'w') as fh:
        json.dump(mycopy, fh)

    with open(listpath, 'w') as fh:
        for k, v in mycopy.items():
              fh.write('{}\n'.format(v['filename']))

def include_region_attribute(inputjson, attribkey, attribvalue, outjson):
    """Include a region attribute to all regions

    Args:
    inputjson(str): path to the input json
    attribkey(str): attribute key to be included
    attribvalue(str): attribute value to be included
    outjson(str): output path
    """

    fh = open(inputjson)
    myjson = json.load(fh)

    for k, v in myjson.items():
        if not v['regions'].keys(): continue

        for kk, region in v['regions'].items():
            region['region_attributes'][attribkey] = attribvalue

    with open(outjson, 'w') as fh:
        json.dump(myjson, fh)

    fh.close()


def split_train_val(inputjson, outdir, ratiotrain=0.8):
    """Split input json into train and val in VIA json format

    Args:
    inputjson(str): path to the input json
    outdir(str): output dir
    ratiotrain(float): Proportion of files in the training set
    """

    fh = open(inputjson)
    myjson = json.load(fh)

    entries = list(myjson.keys())
    nentries = len(entries)

    train = list(entries)
    val = list(entries)
    ntrain = int(nentries*ratiotrain)

    random.shuffle(entries)
    train = entries[:ntrain]
    val = entries[ntrain:]

    trainjsonpath = os.path.join(outdir, 'train.json')
    trainlistpath = os.path.join(outdir, 'train.lst')
    valjsonpath = os.path.join(outdir, 'val.json')
    vallistpath = os.path.join(outdir, 'val.lst')

    filter_and_save_json(myjson, val, trainjsonpath, trainlistpath)
    filter_and_save_json(myjson, train, valjsonpath, vallistpath)

def save_without_duplicates(inputjson, outdir):
    """Save via format without reigon duplicates

    Args:
    inputjson(str): path to the input json
    outdir(str): output dir
    """

    fh = open(inputjson)
    myjson = json.load(fh)

    for k, v in myjson.items():
        if not v['regions'].keys(): continue

        buffer = set()
        repeatedindices = []

        for kk, region in v['regions'].items():
            if str(myjson[k]['regions'][kk]['shape_attributes']['all_points_x']) in buffer:
                repeatedindices.append(kk)
                continue

            buffer.add(str(myjson[k]['regions'][kk]['shape_attributes']['all_points_x']))
        for j in repeatedindices:
            v['regions'].pop(j, None)

    with open(os.path.join(outdir, 'out.json'), 'w') as f:
        json.dump(myjson, f)
    fh.close()

def round_csv_values_to_8digits(incsv):
    fh = open(incsv)
    print(fh.readline().strip())

    for l in fh:
        lon, lat = [ float(x) for x in l.strip().split(',') ]
        print('{:.8f},{:.8f}'.format(lon, lat))
    fh.close()


def generate_gridgsvpoint_queries(jsondir):
    """Generate the insert queries into the graffitidb based on google street view json format

    Args:
    jsondir(str): path to the csvs directory
    """

    #print('/*Insert into GridGsvpoint*/')
    for f in os.listdir(jsondir):
        if not f.endswith('.json'): continue
        fullpath = os.path.join(jsondir, f)
        fh = open(fullpath)
        myjson = json.load(fh)

        if 'location' not in myjson.keys():
            # no correspondance
            continue

        fetchedlat = round(myjson['location']['lat'], 8)
        fetchedlon = round(myjson['location']['lng'], 8)
        _, gridpointlat, gridpointlon = f.replace('.json','').split('_')

        query = '''INSERT INTO GridGsvpoint (gridid, gsvpointid) VALUES ''' \
            ''' ((SELECT Grid.id from Grid WHERE lon={} and lat={} AND ''' \
            '''grid.gridtypeid={}), ''' \
            ''' (SELECT Gsvpoint.id FROM Gsvpoint WHERE lon={} and lat={}));'''.\
            format(gridpointlon, gridpointlat, 1, fetchedlon, fetchedlat)
        print(query)

        fh.close()

def generate_shell_rename_commands(jsondir):
    """Generate the insert queries into the graffitidb based on google street view json format

    Args:
    jsondir(str): path to the csvs directory
    """

    #print('/*Insert into GridGsvpoint*/')
    for f in os.listdir(jsondir):
        if not f.endswith('.json'): continue
        fullpath = os.path.join(jsondir, f)
        fh = open(fullpath)
        myjson = json.load(fh)

        if 'location' not in myjson.keys():
            # no correspondance
            continue

        fetchedlat = round(myjson['location']['lat'], 8)
        fetchedlon = round(myjson['location']['lng'], 8)
        _, gridpointlat, gridpointlon = f.replace('.json','').split('_')

        oldcoords = '_{:.8f}_{:.8f}'.format(round(float(gridpointlat), 8),
                                        round(float(gridpointlon), 8))
        newcoords = '_{}_{}'.format(round(fetchedlat, 8), round(fetchedlon, 8))
        if oldcoords == newcoords: continue
        print('for G in $(locate --database=/home/keiji/.locatebbox.db   "{}"); do mv $(basename $G) ../newbboxes/$(basename ${{G/{}/{}}}); done 2>/dev/null'.format(oldcoords, oldcoords, newcoords))

        fh.close()

def generate_gsvpoint_queries(jsondir, schema, crs):
    """Generate the insert into gsvpoint table queries

    Args:
    jsondir(str): path to the csvs directory

    """
    print('/*Insert into Gsvpoint*/')
    for f in os.listdir(jsondir):
        if not f.endswith('.json'): continue
        fullpath = os.path.join(jsondir, f)
        fh = open(fullpath)
        myjson = json.load(fh)

        if 'location' not in myjson.keys():
            # no correspondance
            continue

        fetchedlat = round(float(myjson['location']['lat']), 8)
        fetchedlon = round(float(myjson['location']['lng']), 8)
        #provider = round(float(myjson['location']['lng']), 8)
        if 'Google' in myjson['copyright']:
            fromgoogle = 'true'
        else:
            fromgoogle = 'false'

        if 'date' in myjson.keys():
            capturedon = "{}-01".format(myjson['date'])
        else:
            capturedon = '2017-06-01' #Adding default date

        query = '''INSERT INTO {}.gsvpoint (lon, lat, capturedon, geom, fromgoogle) VALUES '''\
            ''' ({},{}, '{}', ST_SetSRID(ST_MakePoint({}, {}), '''\
            '''{}), {});'''. \
            format(schema, fetchedlon, fetchedlat, capturedon,
                   fetchedlon, fetchedlat,
                   crs, fromgoogle)
        print(query)
        fh.close()

def generate_gsvpoint_update_false_rows(jsondir):
    """Update queries containing provider

    Args:
    jsondir(str): path to the csvs directory

    """
    print('/*Insert into Gsvpoint*/')
    for f in os.listdir(jsondir):
        if not f.endswith('.json'): continue
        fullpath = os.path.join(jsondir, f)
        fh = open(fullpath)
        myjson = json.load(fh)

        if 'location' not in myjson.keys():
            # no correspondance
            continue
        
        fetchedlat = round(float(myjson['location']['lat']), 8)
        fetchedlon = round(float(myjson['location']['lng']), 8)
        if 'Google' in myjson['copyright']:
            continue

        #capturedon = "{}-01".format(myjson['date'])
        #{
               #"copyright" : " Agncia GoVirtual - Sua empresa em destaque!",
               #"date" : "2017-04",
               #"location" : {
                         #"lat" : -23.56037118804005,
                         #"lng" : -46.65909251334561
                      #},
               #"pano_id" : "CAoSLEFGMVFpcE9CdGJJSllMQTZJN2Nsc2dlZTd2d3lPSkJVOGQyX09DU1hDWlhX",
               #"status" : "OK"
        #}
           #"copyright" : " Google, Inc.",

        query = '''UPDATE gsvpoint SET fromgoogle=false '''\
            ''' WHERE lon={} AND lat={}; '''. \
            format(fetchedlon, fetchedlat)
        print(query)
        fh.close()

def generate_bbox_queries(csvdir):
    """Generate Bbox queries based on the csv files

    Args:
    csvdir(str):
    """
    #classid = 1
    pitch = 20

    for f in os.listdir(csvdir):
        if not f.endswith('.csv'): continue
        fullpath = os.path.join(csvdir, f)
        fh = open(fullpath)

        #print(f)

        _, lat, lon, heading = f.replace('.csv','').split('_')
        for l in fh:
            # coords, origarea, c, s
            xmin, ymin, xmax, ymax, maskarea, classid, score = l.strip().split(',')
            w = int(xmax) - int(xmin)
            h = int(ymax) - int(ymin)
            bboxarea = w*h

            query = '''INSERT INTO Bbox (minx,miny,maxx,maxy,bboxarea,maskarea,score,classid,''' \
                '''imageid) '''\
                ''' VALUES ({}, {}, {}, {}, {}, {}, {}, {}, ''' \
                ''' (SELECT image.id FROM Image WHERE heading={} AND ''' \
                '''pitch={} AND gsvpointid=''' \
                '''(SELECT id FROM Gsvpoint WHERE  lon={} AND lat={})));'''.\
                format(xmin, ymin, xmax, ymax, bboxarea, maskarea, score, classid,
                       heading, pitch, lon, lat)
            print(query)

        fh.close()

def generate_image_queries(imdir):
    """Generate Insert queries based on the images

    Args:
    imdir(str):
    """
    classid = 1
    pitch = 20

    for f in os.listdir(imdir):
        if not f.endswith('.jpg'): continue
        fullpath = os.path.join(imdir, f)
        _, lat, lon, heading = os.path.splitext(f)[0].split('_')

        query = '''INSERT INTO Image (heading, pitch, gsvpointid)''' \
            ''' VALUES ({}, {}, ''' \
            ''' (SELECT id FROM Gsvpoint WHERE ''' \
            ''' lon={} AND lat={}));'''.\
            format(heading, pitch, lon, lat)
        print(query)

def check_empty_region_attributes(inputjson):
    """Analyze json in VIA format and search for empty region_attributes fields.
    Print the file ids of the entries matching

    Args:
    inputjson(str): path to the input json in VIA format
    """

    fh = open(inputjson)
    myjson = json.load(fh)
    for k, v in myjson.items():
        noclass = False
        for kk, vv in v['regions'].items():
            if (vv['region_attributes'] == {}):
                noclass = True
                break
        if noclass:
            print(k)
    fh.close()

def standardize_dir(indir):
    """Standardize filese in indir to 8 digits precision and copy the file to /tmp

    Args:
    indir(str): path containing the files 'im_XXXXXXXX_YYYYYYYYY.zwy'
    """

    for f in os.listdir(indir):
        if not 'im' in f: continue
        arr = f.split('_')
        latstr = '{:.8f}'.format(float(arr[1]))
        lonstr = '{:.8f}'.format(float(arr[2]))
        newf = f.replace(arr[1], latstr).replace(arr[2], lonstr)
        newf = newf.replace('im_', '')

        outdir = '/home/keiji/results/graffiti/20180511-gsv_liberdade/img/'
        shutil.copy(os.path.join(indir, f), '/tmp/' + newf)
        #shutil.move(os.path.join(indir, f), outdir + newf)

def standardize_csv(incsv):
    """Standardize rows in @incsv to 8 digits precision and print the results

    Args:
    incsv(str): path containing the input csv
    """

    fh = open(incsv)
    print(fh.readline().strip())
    for l in fh:
        lon, lat = l.strip().split(',')
        print('{:.8f},{:.8f}'.format(float(lon), float(lat)))

    fh.close()

def plot_dates_hist(datescsv):
    """Plot histogram on capturedon

    Args:
    datescsv(str): fullpath of the file

    """

    x = pd.read_csv(datescsv)
    plt.bar(x['yyyy'], x['count'], width=1, tick_label=x['yyyy'],
            color='dimgray', linewidth=1, edgecolor=['k']*9)
    plt.locator_params(axis='y', nbins=6)
    plt.show()

def generate_square_grid(minlat, maxlat, minlon, maxlon, delta, outfile):
    """Generate grid and output to file

    Args:
    minlat(float): min latitude
    maxlon(float): max latitude
    minlon(float): min longitude
    maxlon(float): max longitude
    """
    fh = open(outfile, 'w')
    d = delta
    nlat = int((maxlat - minlat)/d) + 1
    nlon = int((maxlon - minlon)/d) + 1

    fh.write('id,lat,lon\n')
    id = 1
    for i in range(nlat):
        lat = round(minlat + i*d, 8)
        for j in range(nlon):
            lon = round(minlon + j*d, 8)
            fh.write('{},{:.8f},{:.8f}\n'.format(id, lat, lon))
            id += 1

    fh.close()

def overlay_objects_all(im1dir, annot1path, im2dir, outdir='/tmp'):
    im2paths = []
    imgs = os.listdir(im2dir)
    #random.shuffle(imgs)
    for f in imgs:
        if not f.endswith('.jpg'): continue
        im2paths.append(os.path.join(im2dir, f))

    with open(annot1path, 'rb') as fh:
        annot1json = json.load(fh)

    annotoutjson = annot1json.copy()
    nim2paths = len(im2paths)

    i = 0
    for f in os.listdir(im1dir):
        if not f.endswith('.jpg'): continue
        im1path = os.path.join(im1dir, f)
        im2path = im2paths[i%nim2paths]
        overlay_objects(im1path, annot1json, im2path, outdir)
        outimg = os.path.join(outdir, f)
        annotoutjson = update_json(annotoutjson, outimg)
        i += 1

    outjson = os.path.join(outdir, 'via_region_data.json')
    with open(outjson, 'w') as fh:
        json.dump(annotoutjson, fh)

def update_json(annotoutjson, newimg):
    filename = os.path.basename(newimg)

    imgkey = None
    for k in annotoutjson.keys():
        if filename in k:
            imgkey = k
            break

    sz = os.path.getsize(newimg)
    newkey = filename + str(sz)
    annotoutjson[newkey] = annotoutjson.pop(imgkey)
    annotoutjson[newkey]['size'] = sz
    return annotoutjson

def overlay_objects(im1path, annot1json, im2path, outdir):
    """Overlay image1 over image2

    Args:
    im1path(str): image1 containing the object
    annot1path(str): annotation (json) in via format for the objects in image1
    im2path(str): image2 chosen image to be the background
    """
    filename = os.path.basename(im1path)
    outpath = os.path.join(outdir, filename)

    imgkey = None
    for k in annot1json.keys():
        if filename in k:
            imgkey = k
            break

    if not imgkey:
        print('Image does not contain annotation in the file provided.')
        return

    inputimg = imageio.imread(im1path)
    finalimg = imageio.imread(im2path)

    for _, v in annot1json[imgkey]['regions'].items():
        rr, cc = skimage.draw.polygon(v['shape_attributes']['all_points_y'], v['shape_attributes']['all_points_x'])
        finalimg[rr, cc, :] = inputimg[rr, cc, :]

    imageio.imwrite(outpath, finalimg)

##########################################################
def main():
    #im1dir = '/home/frodo/temp/signboard/kaistscenetext_augmented/'
    #annot1path = '/home/frodo/temp/signboard/kaistscenetext_augmented/via_region_data.json'
    #im1dir = '/home/frodo/temp/20180801-gsv_annotated/del/'
    #annot1path = '/home/frodo/temp/20180801-gsv_annotated/del/all.json'
    #im2dir = '/home/frodo/temp/20180801-backgrounds//'

    #overlay_objects(im1path, annot1path, im2path)
    #overlay_objects_all(im1dir, annot1path, im2dir)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputjson', required=True, help='Input json in via format')
    parser.add_argument('--outdir', required=True, help='Input json in via format')
    args = parser.parse_args()

    #save_without_duplicates(args.inputjson)
    #split_train_val(args.inputjson, args.outdir)
    #include_region_attribute(args.inputjson, 'class', 'tag', '/tmp/out.json')

    ##########################################################
    #imgdir = '/home/keiji/results/graffiti/20180511-gsv_spcity/img/'
    #metadatadir = '/home/keiji/results/graffiti/20180511-gsv_spcity/metadata/'
    #coordsfile = '/home/keiji/results/graffiti/20180511-gsv_spcity/points001.csv'

    #generate_square_grid(-34.70521111, -34.52751111, -58.53151111, -58.33511111,
                         #0.0005, '/tmp/buenosairesgrid.csv')
    generate_square_grid(-21.37000000, -21.05800000, -47.99000000, -47.64500000,
                         0.001, '/tmp/ribeirao_grid.csv')
    #-47.99000000 | -47.64500000 | -21.37000000 | -21.05800000 

    #generate_square_grid(51.39921111, 51.52911111, -2.71461111, -2.44001111,
                         #0.0005, '/home/keiji/temp/bristolgrid.csv')
    #generate_gsvpoint_update_false_rows(metadatadir)
    #generate_gsvpoint_queries('/misc/users/keiji/partition1/20180511-gsv_spcity/metadata/')
    #generate_gsvpoint_queries('/home/keiji/temp/20180625-mexico_metadata/')
    #generate_gridgsvpoint_queries('/home/keiji/temp/newjson/')
    #generate_gridgsvpoint_queries('/misc/users/keiji/partition1/20180511-gsv_spcity/metadata/')
    #generate_image_queries('/misc/users/keiji/partition1/20180511-gsv_spcity/img')
    #generate_bbox_queries('/home/keiji/results/graffiti/20180511-gsv_spcity/newbboxes/')
    #generate_bbox_queries('/misc/users/keiji/partition1/20180511-gsv_spcity/newbboxes/')
    #generate_shell_rename_commands('/misc/users/keiji/partition1/20180511-gsv_spcity/metadata/')
    #generate_shell_rename_commands('/home/keiji/temp/newjson/')
    #generate_gsvpoint_queries(jsondir):
    #validate_metadata(metadatadir, coordsfile)
    #validate_files(imgdir, coordsfile)

if __name__ == "__main__":
    main()

