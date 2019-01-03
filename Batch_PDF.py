"""
TimeResolved_main.py

Definition of the script. What does it do

Please see
  Github: https://github.com/EmilSkaaning 
  ----------------------------------------------------------------------
  Author:       Emil Thyge Skaaning Kjaer (rsz113@alumni.ku.dk)
  Supervisor:   Kirsten M. OE. Jensen
  Department:   Nanoscience Center and Department of Chemistry
  Institute:    University of Copenhagen
  Date:         03/01/2019
  ----------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import errno
import pickle
import ConfigParser
import h5py
import heapq
import sys
import ast
import itertools

from prompter import yesno
from tqdm import tqdm
from diffpy.pdfgetx import PDFGetter, loadPDFConfig, findfiles
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#---------------------------------------------------------------------------------------------------
#   
# Functions
#
#---------------------------------------------------------------------------------------------------

def fields_view(arr, fields):
    dtype2 = np.dtype({name:arr.dtype.fields[name] for name in fields})
    return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)

def neg_check(mat, start, owd, gen_pic, totime):
    neg_files = []
    neg_vals  = []

    for i in range(len(mat)):
        ph = 0
        total_neg = 0
        files_neg = 0


        for j in range(len(mat[i])):
            if ph == 1 and mat[i][j] < 0:
                total_neg += 1

            elif ph == 0 and mat[i][j] < 0:
                total_neg += 1
                files_neg += 1
                ph = 1
                neg_files.append(i)

            if j == len(mat[i])-1:
                neg_vals.append(total_neg)
    
    fig, ax = plt.subplots(figsize=(14,6))
    x = np.arange(start, start+len(mat))*totime

    ax.plot(x, neg_vals, 'bo-', label='Negative Values')
    ax.set_ylabel('Negative Values', fontsize=16)
    ax.set_xlabel('Time [m]', fontsize=16)
    ax.set_xlim(x[0], x[-1])
    ax.set_title('Negative Values in Datasets', fontsize=20)
    ax.legend(loc='best', fontsize=13)
    ax.tick_params(axis='x', labelsize = 16)
    ax.tick_params(axis='y', labelsize = 16)

    if gen_pic:
        pic_dir(owd, 'Pictures_')
        fig.savefig('Negative_Values' + '.png')
        plt.clf()

    else:
        plt.draw()
        plt.pause(0.1)
        raw_input("<Hit Enter To Close>")
        plt.close(fig)

    return neg_files, neg_vals

def pic_dir(owd, folder_name):
    """ 
    pic_dir takes two inputs:
        owd, is the directory where the script is.
        folder_name, is the name of the folder where the files will be put.

    If a folder with folder_name does not exist, then a new folder will be made.
    The function than changes the directory.
    """
    os.chdir(owd)  # Goes back to the directory where the script is
    #dir_path = os.path.dirname(os.path.realpath(__file__))  # Ta
    dir_path = owd
    dir_path = dir_path + '/' + str(folder_name) + str(dict['file_name'])

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    os.chdir(dir_path)
    print 'Directory has been changed:'
    print os.getcwd()   

def read_data(frame_sumstart, nr_files, file_name, file_type, line_skip, load_str, seq=True):
    """

    """
    x_values = []
    y_values = []
    min_val  = []
    max_val  = []
    names    = []
    dim      = 1
    lendat1  = 0
    
    if seq == True:
        print 'Loading '+str(load_str)+':'
        for i in tqdm(range(frame_sumstart, frame_sumstart + nr_files)):
            frame = file_name + str.zfill(str(i), 5) + str(file_type)
            names.append(frame)
            frame_data = np.loadtxt(frame, skiprows = line_skip)
            x_values.append(frame_data[:,0])
            y_values.append(frame_data[:,1])
            min_val.append(frame_data[0,0])
            max_val.append(frame_data[-1,0])
            
            lendat2 = len(y_values[i - frame_sumstart])
            
            if lendat2 > lendat1:
                lendat1 = lendat2
    else:
        print 'Loading '+str(load_str)+':'
        index = 0
        for i in range(0, len(nr_files), 2):  # Takes every second element in list
            frame = file_name + str.zfill(str(nr_files[i]), 5) + str(file_type)
            names.append(frame)
            frame_data = np.loadtxt(frame, skiprows = line_skip)
            x_values.append(frame_data[:,0])
            y_values.append(frame_data[:,1])
            min_val.append(frame_data[0,0])
            max_val.append(frame_data[-1,0])

            lendat2 = len(y_values[index])
            index += 1
            if lendat2 > lendat1:
                lendat1 = lendat2
                
    if not all(len(i) == len(x_values[0]) for i in x_values):
        print ('Not all lists have same length! You must have used Dioptas ...')
    else:
        x_values = x_values[0]
        dim = 0

    min_val = np.array(min_val)
    max_val = np.array(max_val)

    min_val = np.amax(min_val)
    max_val = np.amin(max_val)
        
    return x_values, y_values, dim, min_val, max_val, lendat1, names

def lets_plot(x_data, y_data, x_bg, y_bg, x_diff, y_diff, x_data1, y_data1, x_bg1, y_bg1, x_diff1, y_diff1, gen_pic, owd):
    fig2, ((ax), (ax1)) = plt.subplots(2, 1, figsize = (14,6))

    ax.plot(x_data, y_data, label='Data')
    ax.plot(x_bg, y_bg, label='Bg')
    ax.plot(x_diff, y_diff, label='Data - bg')

    ax.set_xlabel("Q [$\AA$$^-$$^1$]", fontsize=16) # the label of the y axis
    ax.set_ylabel('Intensity [a.u.]', fontsize=16)  # the label of the y axis
    ax.set_title("Background Subtraction, First Frame", fontsize=20) # the title of the plot
    ax.tick_params(axis='x', labelsize = 16)
    ax.tick_params(axis='y', labelsize = 16)
    ax.legend(loc='best', fontsize = 13)
    ax.set_xlim(x_data[0], x_data[-1])
    ax.yaxis.set_major_locator(plt.NullLocator())

    ax1.plot(x_data1, y_data1, label='Data')
    ax1.plot(x_bg1, y_bg1, label='Bg')
    ax1.plot(x_diff1, y_diff1, label='Data - bg')

    ax1.set_xlabel("Q [$\AA$$^-$$^1$]", fontsize=16) # the label of the y axis
    ax1.set_ylabel('Intensity [a.u.]', fontsize=16)  # the label of the y axis
    ax1.set_title("Background Subtraction, Last Frame", fontsize=20) # the title of the plot
    ax1.tick_params(axis='x', labelsize = 16)
    ax1.tick_params(axis='y', labelsize = 16)
    ax1.legend(loc='best', fontsize = 13)
    ax1.yaxis.set_major_locator(plt.NullLocator())
    ax1.set_xlim(x_data[0], x_data[-1])
    plt.tight_layout(pad=2, w_pad=1.0, h_pad=1.0)

    if gen_pic:
        pic_dir(owd, 'Pictures_')
        fig2.savefig('Background' + '.png')
        plt.clf()
    else:
        plt.draw()
        plt.pause(0.1)
        raw_input("<Hit Enter To Close>")
        plt.close(fig2)

    return 

def interpol(xmin, xmax, steps, xdata, ydata):
    xnew = np.linspace(xmin, xmax, num=steps, endpoint=True)
    ynew = np.interp(xnew, xdata, ydata)
    return xnew, ynew

def sum_data(nr_sum, sumstep, data):
    
    summing = nr_sum/sumstep
        
    maaske = nr_sum - (summing * sumstep)
    placeholder = sumstep
    sum_mat = []
    constant = sumstep - (nr_sum - (sumstep * summing))
    
    for i in range(summing):
        sum_mat.append(np.sum(data[int(placeholder - sumstep):int(placeholder)], axis=0))
        
        if placeholder == summing * sumstep and constant != sumstep:
            print 'WARNING!!! Could not sum frames in pairs of ' + str(sumstep) + '. To correct for this, the intensity of last data set is multiplied by ' + str(constant+1) + ' in the last frame.' 
            last_mat = data[placeholder::, 0::]
            exstend_mat = data[-1::, 0::]

            for i in range(constant):
                last_mat = np.concatenate((last_mat, exstend_mat), axis = 0)

            sum_mat.append(np.sum(last_mat, axis=0))

        placeholder += sumstep

    sum_mat = np.array(sum_mat)
    return sum_mat  

#---------------------------------------------------------------------------------------------------
#   
# Config file and Dictionary
#
#---------------------------------------------------------------------------------------------------

if os.path.isfile('Batch_PDF_config.ini'):
    timeResCon = True
    parser     = ConfigParser.ConfigParser()
    parser.read('Batch_PDF_config.ini')
    load_dict  = parser.getboolean('Dictionary', 'load_dict')
    imp_dict   = parser.get('Dictionary', 'imp_dict')
else:
    load_dict  = False
    timeResCon = False
    imp_dict   = 'Batch_PDF_directory.py'
    print "No Batch_PDF_config.ini exists within current working directory!"
    print "Looking for directory name Batch_PDF_directory."  # XXXXXX
    if load_dict:
        print 'Script is using an imported dictionary!'
    else:
        print 'Neither a config file og dictionary was imported. Script should be setup in main code!!!'

#---------------------------------------------------------------------------------------------------
#   
# Define Dictionary
#
#--------------------------------------------------------------------------------------------------- 

if load_dict:
    dict = pickle.load(open(imp_dict, "rb"))
    print '\n' + 'Printing values for imported Dictionary: ' + '\n'

    print '[Main]'
    print '{0:12s} {1} {2}'.format('load_data', '= ', dict['load_data'])
    print '{0:12s} {1} {2}'.format('data_magic', '= ', dict['data_magic'])
    print '{0:12s} {1} {2}'.format('save_data', '= ', dict['save_data'])
    print '{0:12s} {1} {2}'.format('PDF', '= ', dict['PDF'])
    print '{0:12s} {1} {2}'.format('gen_PDF_file', '= ', dict['gen_PDF_file'])
    print '{0:12s} {1} {2}'.format('gen_fq_file', '= ', dict['gen_fq_file'])
    print '{0:12s} {1} {2}'.format('gen_iq_file', '= ', dict['gen_iq_file'])
    print '{0:12s} {1} {2}'.format('Nyquist', '= ', dict['Nyquist'])

    print '\n[Data]'
    print '{0:12s} {1} {2}'.format('file_name', '= ', dict['file_name'])
    print '{0:12s} {1} {2}'.format('file_type', '= ', dict['file_type'])
    print '{0:12s} {1} {2}'.format('first_file', '= ', dict['first_file'])
    print '{0:12s} {1} {2}'.format('nr_files', '= ', dict['nr_files'])
    print '{0:12s} {1} {2}'.format('line_skip', '= ', dict['line_skip'])

    print '\n[Background_Data]'
    print '{0:12s} {1} {2}'.format('bg_file', '= ', dict['bg_file'])
    print '{0:12s} {1} {2}'.format('bg_type', '= ', dict['bg_type'])
    print '{0:12s} {1} {2}'.format('first_bg', '= ', dict['first_bg'])
    print '{0:12s} {1} {2}'.format('nr_bg_files', '= ', dict['nr_bg_files'])
    print '{0:12s} {1} {2}'.format('bgline_skip', '= ', dict['bgline_skip'])
    print '{0:12s} {1} {2}'.format('seq_bg', '= ', dict['seq_bg'])
    print '{0:12s} {1} {2}'.format('bg_info', '= ', dict['bg_info'])

    print '\n[Scaling]'
    print '{0:12s} {1} {2}'.format('calib_bg', '= ', dict['calib_bg'])
    print '{0:12s} {1} {2}'.format('auto', '= ', dict['auto'])
    print '{0:12s} {1} {2}'.format('sumstep', '= ', dict['sumstep'])
    print '{0:12s} {1} {2}'.format('bg_scaling', '= ', dict['bg_scaling'])
    print '{0:12s} {1} {2}'.format('qnorm', '= ', dict['qnorm'])

    print '\n[Directories]'
    print '{0:12s} {1} {2}'.format('data_dir', '= ', dict['data_dir'])
    print '{0:12s} {1} {2}'.format('bg_dir', '= ', dict['bg_dir'])
    print '{0:12s} {1} {2}'.format('cfg_dir', '= ', dict['cfg_dir'])

    print '\n[PDFgetX3]'
    print '{0:12s} {1} {2}'.format('make_cfg', '= ', dict['make_cfg'])
    print '{0:12s} {1} {2}'.format('cfg_file', '= ', dict['cfg_file'])

    print '\n[Plotting]'
    print '{0:12s} {1} {2}'.format('show_PDF', '= ', dict['show_PDF'])
    print '{0:12s} {1} {2}'.format('show_all', '= ', dict['show_all'])
    print '{0:12s} {1} {2}'.format('save_pics', '= ', dict['save_pics'])
    print '{0:12s} {1} {2}'.format('pdf_file', '= ', dict['pdf_file'])
    print '{0:12s} {1} {2}'.format('timeframe', '= ', dict['timeframe'])
    print '{0:12s} {1} {2}'.format('PDF_name', '= ', dict['PDF_name'])

    print '\n[3D_Plot]'
    print '{0:12s} {1} {2}'.format('3D_plot', '= ', dict['3D_plot'])
    print '{0:12s} {1} {2}'.format('3D_title', '= ', dict['3D_title'])
    print '{0:12s} {1} {2}'.format('3D_cmap', '= ', dict['3D_cmap'])

    print '\nThe sections [Dictionary] and [Save_Dictionary] are not stored within the Dictionary!'

    yesno('Do you agree with imported dictionary?')
    print "Imported Dictionary has been verfied. Proceeding!"
    save_dict = False

elif timeResCon:
    parser = ConfigParser.ConfigParser()
    dict = {}
    parser.read('Batch_PDF_config.ini')

    # [Dictionary]
    load_dict           = parser.getboolean('Dictionary', 'load_dict')
    imp_dict            = parser.get('Dictionary', 'imp_dict') 

    # [Main]
    dict['load_data']   = parser.getboolean('Main', 'load_data')
    dict['data_magic']  = parser.getboolean('Main', 'data_magic')
    dict['save_data']   = parser.getboolean('Main', 'save_data')
    dict['PDF']         = parser.getboolean('Main', 'PDF')
    dict['gen_PDF_file']= parser.getboolean('Main', 'gen_PDF_file')
    dict['gen_fq_file'] = parser.getboolean('Main', 'gen_fq_file')
    dict['gen_iq_file'] = parser.getboolean('Main', 'gen_iq_file')
    dict['Nyquist']     = parser.getboolean('Main', 'Nyquist')

    # [Data]
    dict['file_name']   = parser.get('Data', 'file_name')
    dict['file_type']   = parser.get('Data', 'file_type') 
    dict['first_file']  = parser.getint('Data', 'first_file')
    dict['nr_files']    = parser.getint('Data', 'nr_files')
    dict['line_skip']   = parser.getint('Data', 'line_skip')

    # [Background_data]
    dict['bg_file']     = parser.get('Background_Data', 'bg_file')
    dict['bg_type']     = parser.get('Background_Data', 'bg_type') 
    dict['first_bg']    = parser.getint('Background_Data', 'first_bg')
    dict['nr_bg_files'] = parser.getint('Background_Data', 'nr_bg_files')
    dict['bgline_skip'] = parser.getint('Background_Data', 'bgline_skip')                                                                                                     
    dict['seq_bg']      = parser.getboolean('Background_Data', 'seq_bg') 
    dict['bg_info']     = ast.literal_eval(parser.get('Background_Data', 'bg_info'))  # Gets a list
    
    # [Scaling]
    dict['calib_bg']    = parser.getboolean('Scaling', 'calib_bg')  
    dict['auto']        = parser.getboolean('Scaling', 'auto') 
    dict['sumstep']     = parser.getint('Scaling', 'sumstep')
    dict['bg_scaling']  = parser.getfloat('Scaling', 'bg_scaling')
    dict['qnorm']       = parser.get('Scaling', 'qnorm')

    # [Directories]
    dict['data_dir']    = parser.get('Directories', 'data_dir')                                     
    dict['bg_dir']      = parser.get('Directories', 'bg_dir')
    dict['cfg_dir']     = parser.get('Directories', 'cfg_dir')

    # [PDFgetX3]
    dict['make_cfg']    = parser.getboolean('PDFgetX3', 'make_cfg')
    dict['cfg_file']    = parser.get('PDFgetX3', 'cfg_file')

    # [Plotting]
    dict['show_PDF']    = parser.getboolean('Plotting', 'show_PDF')
    dict['show_all']    = parser.getboolean('Plotting', 'show_all') 
    dict['save_pics']   = parser.getboolean('Plotting', 'save_pics')
    dict['pdf_file']    = parser.getint('Plotting', 'pdf_file') 
    dict['timeframe']   = parser.getfloat('Plotting', 'timeframe')
    dict['PDF_name']    = parser.get('Plotting', 'PDF_name')

    # [3D Plot]
    dict['3D_plot']     = parser.getboolean('3D_Plot', '3D_plot')
    dict['3D_title']    = parser.get('3D_Plot', '3D_title')
    dict['3D_cmap']     = parser.get('3D_Plot', '3D_cmap')
    
    # [Save_Dictionary]
    save_dict           = parser.getboolean('Save_Dictionary', 'save_dict')
    dict_name           = parser.get('Save_Dictionary', 'dict_name')

else:
    dict = {}

    # [Dictionary]
    load_dict           = False
    imp_dict            = dictionary.py

    # [Main]
    dict['load_data']   = False  # Load hdf5.file
    dict['data_magic']  = True                                                                                  #Check all data, and makes sure it matches in lengths and size
    dict['save_data']   = False                                                                                 #Should save data in right format, does nothing at the moment
    dict['PDF']         = True                                                                                  #Calculates PDF
    dict['gen_PDF_file']= True
    dict['gen_fq_file'] = True
    dict['gen_iq_file'] = True
    dict['Nyquist']     = True
    # [Data]
    dict['file_name']   = 'BA_WCl6_160_p-'                                                                      #Starting name of files you want inported, e.g. 'BA_WCl6_200-', full name 'BA_WCl6_200-00001'.
    dict['file_type']   = '.xy'                                                                                 #Type of file you want imported. Remember '.' in front, e.g. '.xy' 
    dict['first_file']  = 0                                                                                     #First file, 0 = 1.
    dict['nr_files']    = 6                                                                                 #Number of files you want to import
    dict['line_skip']   = 16                                                                                    #Amount of header lines that will be skiped, 16 for .xy- and 4 for .chi-files

    # [Background_Data]
    dict['bg_file']     = 'BA_BKG_160_p-'                                                                       #Name of the background file. So far only possible to take one file, so full file name
    dict['bg_type']     = '.xy'                                                                                 #Type of file you want imported. Remember '.' in front, e.g. '.xy' 
    dict['first_bg']    = 0                                                                                     #First file, 0 = 1.
    dict['nr_bg_files'] = 3                                                                                 #Number of files you want to import
    dict['bgline_skip'] = 16    
    dict['seq_bg']      = False
    dict['bgline_skip'] = [1,2]    

    #[Scaling]
    dict['calib_bg']    = True                                                                                  #If false, autoscale at qnorm   
    dict['auto']        = True
    dict['sumstep']     = 1                                                                                     #Summing files to increase intensity. If = 1, then no summation will be done                                                                                    
    dict['bg_scaling']  = 0.98                                                                                  #Constant scaling of bagground
    dict['qnorm']       = 22                                                                                    #Define the point in q, where the background should line up with the data

    # [Directories]
    dict['data_dir']    = 'C:\Users\opadn\Documents\Skolefiler\KU_Nanoscience\Kandidat\Masters\Data_WO\BA_WCl6_160_p'   #Insert path to a new directory                                     
    dict['bg_dir']      = 'C:\Users\opadn\Documents\Skolefiler\KU_Nanoscience\Kandidat\Masters\Data_WO\BA_BKG_160_p'
    dict['cfg_dir']     = 'C:\Users\opadn\Documents\Skolefiler\KU_Nanoscience\Kandidat\Masters\TimeResolved_scripts'

    # [PDFgetX3]
    dict['make_cfg']    = False
    dict['cfg_file']    = 'pdfgetx3.cfg'

    # [Plotting]
    dict['show_PDF']    = True                                                                                  #Shows PDF
    dict['show_all']    = True                                                                                  #Shows iq, sq, fq and Gr for a specific file, pdf_file
    dict['save_pics']   = True
    dict['pdf_file']    = -1                                                                                    #Show the pdf of a specific file        
    dict['timeframe']   = 2                                                                                     # Time for every measurement in seconds
    dict['PDF_name']    = 'WCl6_160_p'

    # [3D_Plot]
    dict['3D_plot']     = True
    dict['3D_title']    = '3D plot'
    dict['3D_cmap']     = 'Spectral'

    # [Save_Dictionary]
    save_dict           = True
    dict_name           = str('XXX_directory')

#---------------------------------------------------------------------------------------------------
#   
# Checking settings
#
#---------------------------------------------------------------------------------------------------

error = 0

if dict['seq_bg'] and dict['make_cfg'] == False:
    if dict['bg_info'][1] != 1:
        print 'bg_info is wrong!'
        print 'Second element needs to be 1 but is ' + str(dict['bg_info'][1]) + '!\n' 
        error = 1

    if dict['bg_info'][-1] > dict['first_file'] + dict['nr_files']:
        print 'bg_info is wrong!'
        print str(dict['bg_info'][-1]) + ' is larger than ' + str(dict['first_file'] + dict['nr_files'])
        error = 1

    if len(dict['bg_info']) % 2 != 0:
        print 'bg_info is wrong!'
        print 'The length needs to be even but is currently ' + str(len(dict['bg_info'])) + '!\n'
        error = 1
    
    j = 1
    for i in range(1, len(dict['bg_info']), 2):
        if dict['bg_info'][i] < dict['bg_info'][j]:
            print 'bg_info is wrong!'
            print str(dict['bg_info'][i]) + ' must be larger than ' + str(dict['bg_info'][j]) + '!\n'
            error = 1
        j = i

if error == 1:
    sys.exit()

#---------------------------------------------------------------------------------------------------
#   
# Code
#
#--------------------------------------------------------------------------------------------------- 

owd = os.getcwd()
totime = dict['timeframe']/60

#---------------------------------------------------------------------------------------------------
#   
# Import Data and Construct Arrays
#
#---------------------------------------------------------------------------------------------------

if dict['load_data'] == False and dict['make_cfg'] == False:
    print '\nLoading HDF5 files:'
    pic_dir(dict['cfg_dir'], 'data_binary_')
    
    hdf5_file = h5py.File('raw_data.hdf5', 'r')

    xdata_set   = hdf5_file['xdata'].value
    ydata_set   = hdf5_file['ydata'].value
    xbg_set     = xdata_set
    ybg_set     = hdf5_file['ybgdata'].value

    hdf5_file.close()

    dict['data_magic'] = False
    dict['sumstep'] = 1
    dict['nr_bg_files'] = dict['nr_files']
    same_len = 0
    steps = len(xdata_set)
    dict['save_data'] = False
elif dict['make_cfg'] == False:
    print '\nInitilazing import of files!'
   
    if dict['seq_bg'] == False:           
        os.chdir(dict['data_dir'])
        print 'Directory has been changed:'
        print os.getcwd()
        xdata_set, ydata_set, data_dim, min_val_data, max_val_data, data_len, data_name = read_data(dict['first_file'], dict['nr_files'], dict['file_name'], dict['file_type'], dict['line_skip'], 'Data files')
        print data_len 
        os.chdir(dict['bg_dir'])
        print 'Directory has been changed:'
        print os.getcwd()
        xbg_set, ybg_set, bg_dim, min_val_bg, max_val_bg, bg_len, bg_name = read_data(dict['first_bg'], dict['nr_bg_files'], dict['bg_file'], dict['bg_type'], dict['bgline_skip'], 'Background files')  

    else:
        os.chdir(dict['data_dir'])
        print 'Directory has been changed:'
        print os.getcwd()     
        xdata_set, ydata_set, data_dim, min_val_data, max_val_data, data_len, data_name = read_data(dict['first_file'], dict['nr_files'], dict['file_name'], dict['file_type'], dict['line_skip'], 'Data files')
          
        os.chdir(dict['bg_dir'])
        print 'Directory has been changed:'
        print os.getcwd()
        xbg_set, ybg_set, bg_dim, min_val_bg, max_val_bg, bg_len, bg_name = read_data(dict['first_bg'], dict['bg_info'], dict['bg_file'], dict['bg_type'], dict['bgline_skip'], 'Background files', seq=False)  
   
    if bg_len > data_len:
        steps = bg_len * 2
    elif bg_len < data_len:
        steps = data_len * 2
    else:
        steps = data_len*2  # Data_len

    if dict['data_magic']:  # Find highest min value and lowest max value for interpolation
        xmin = 0
        xmax = 0

        if min_val_data > min_val_bg:
            xmin = min_val_data
        else:
            xmin = min_val_bg

        if max_val_data < max_val_bg:
            xmax = max_val_data
        else:
            xmax = max_val_bg   

        ydata_set_int = np.zeros((dict['nr_files'], steps))
        if dict['seq_bg'] == False:
            ybg_set_int = np.zeros((dict['nr_bg_files'], steps))
        else:
            ybg_set_int = np.zeros((len(dict['bg_info'])/2, steps))

        same_len = 1
        if data_dim == 0 and bg_dim == 1:
            print '\n', 'Background files vary in length.'
            print 'Interpolating data files:'
            for i in tqdm(range(0, dict['nr_files'])):
                xdata_set_int, y_int = interpol(xmin, xmax, steps, xdata_set, ydata_set[i])
                ydata_set_int[i] = y_int

            print 'Interpolating background files:'
            if dict['seq_bg'] == False:
                for i in tqdm(range(0, dict['nr_bg_files'])):    
                    _, ybg_int = interpol(xmin, xmax, steps, xbg_set[i], ybg_set[i])
                    ybg_set_int[i] = ybg_int
            else:
                for i in tqdm(range(0, len(dict['bg_info'])/2)):    
                    _, ybg_int = interpol(xmin, xmax, steps, xbg_set[i], ybg_set[i])
                    ybg_set_int[i] = ybg_int

        elif data_dim == 1 and bg_dim == 0:
            print '\n', 'Data files vary in length.'
            print 'Interpolating data files:'
            for i in tqdm(range(0, dict['nr_files'])):
                xdata_set_int, y_int = interpol(xmin, xmax, steps, xdata_set[i], ydata_set[i])
                ydata_set_int[i] = y_int

            print 'Interpolating background files:'
            if dict['seq_bg'] == False:
                for i in tqdm(range(0, dict['nr_bg_files'])):    
                    _, ybg_int = interpol(xmin, xmax, steps, xbg_set, ybg_set[i])
                    ybg_set_int[i] = ybg_int
            else:
                for i in tqdm(range(0, len(dict['bg_info'])/2)):    
                    _, ybg_int = interpol(xmin, xmax, steps, xbg_set, ybg_set[i])
                    ybg_set_int[i] = ybg_int
 
        elif data_dim == 1 and bg_dim == 1:
            print '\n', 'Size of data and background array does not match.'
            print 'Interpolating data files:'
            for i in tqdm(range(0, dict['nr_files'])):
                xdata_set_int, y_int = interpol(xmin, xmax, steps, xdata_set[i], ydata_set[i])
                ydata_set_int[i] = y_int

            print 'Interpolating bachground files:'
            if dict['seq_bg'] == False:
                for i in tqdm(range(0, dict['nr_bg_files'])):    
                    _, ybg_int = interpol(xmin, xmax, steps, xbg_set[i], ybg_set[i])
                    ybg_set_int[i] = ybg_int
            else:
                for i in tqdm(range(0, len(dict['bg_info'])/2)):    
                    _, ybg_int = interpol(xmin, xmax, steps, xbg_set[i], ybg_set[i])
                    ybg_set_int[i] = ybg_int
          
        else:
            print '\nAll data have same dimension'
            
            if np.array_equal(xdata_set, xbg_set) == True:
                steps = data_len
                xdata_set_int = xdata_set
                ydata_set_int = ydata_set
                xdata_set = xbg_set
                ybg_set_int = ybg_set

                print 'No need for interpolation'
            else:
                print '\nData got same length but different x values.'
                print 'Data will be interpolated to have same x values.' 
                for i in tqdm(range(0, dict['nr_files'])):
                    print 
                    xdata_set_int, y_int = interpol(xmin, xmax, steps, xdata_set, ydata_set[i])
                    ydata_set_int[i] = y_int

                print 'Interpolating background files:'
                if dict['seq_bg'] == False:
                    for i in tqdm(range(0, dict['nr_bg_files'])):    
                        _, ybg_int = interpol(xmin, xmax, steps, xbg_set, ybg_set[i])
                        ybg_set_int[i] = ybg_int
                else:
                    for i in tqdm(range(0, len(dict['bg_info'])/2)):    
                        _, ybg_int = interpol(xmin, xmax, steps, xbg_set, ybg_set[i])
                        ybg_set_int[i] = ybg_int

        xdata_set = xdata_set_int
        ydata_set = ydata_set_int
        xbg_set = xdata_set
        ybg_set = ybg_set_int  

    if len(dict['bg_info'])/2 < dict['nr_files'] and dict['seq_bg'] == True:  # Makes a new matrix with backgrounds
        test_bg   = []
        extend    = dict['bg_info'][3::2]  # Starts at third index and then takes every second
        bg_index  = 0
        start_val = 0
        for i in extend:
            for j in range((i-start_val)-1):
                test_bg.append(ybg_set[bg_index])
            bg_index += 1
            start_val = i-1
            if i == extend[-1]:
                for i in range((dict['nr_files']-extend[-1])+1):
                    test_bg.append(ybg_set[-1])    
                
        ybg_set = np.array(test_bg)

    if dict['sumstep'] > 1 and dict['seq_bg'] == False:
        ydata_set = sum_data(dict['nr_files'], dict['sumstep'], ydata_set)
        ybg_set = sum_data(dict['nr_bg_files'], dict['sumstep'], ybg_set)
        dict['nr_files'] = (dict['nr_files']/dict['sumstep'])+1
        dict['nr_bg_files'] = (dict['nr_bg_files']/dict['sumstep'])+1

    if same_len == 1 and dict['seq_bg'] == False:
        ybg_set = np.reshape(ybg_set, (dict['nr_bg_files'], steps))

    if dict['nr_files'] > dict['nr_bg_files'] and dict['seq_bg'] == False:  # If there are less background files the data files exstend bg matrix with last background row til they match
        add_bgy = ybg_set[-1]
        add_bgy = np.reshape(add_bgy, (1, steps))

        print '\n', 'Extending background matrix:'
        for i in tqdm(range(abs(dict['nr_files'] - dict['nr_bg_files']))):
            ybg_set = np.concatenate((ybg_set, add_bgy), axis = 0)
     
if dict['save_data'] and dict['make_cfg'] == False:
    print '\nSaving data!'
    print '\tSaved data is not background subtrackted.\n'
    print '\tSaved data does not contain headers!'
   
    pic_dir(dict['cfg_dir'], 'data_binary_')
   
    hdf5_data = h5py.File('raw_data.hdf5', 'w') 

    hdf5_data.create_dataset('xdata', data=xdata_set)
    hdf5_data.create_dataset('ydata', data=ydata_set)   
    hdf5_data.create_dataset('ybgdata', data=ybg_set)   

    hdf5_data.close()   

#---------------------------------------------------------------------------------------------------
#   
# Cfg
#
#---------------------------------------------------------------------------------------------------

if dict['make_cfg'] and load_dict == False:
    # Values for autogen sfg
    cfg_name    = str(dict['cfg_file'])

    dataformat  = 'QA'
    outputtypes = 'iq, sq, fq, gr'
    composition = 'W Cl6'

    qmaxinst    = 19.0
    qmin        = 0.8
    qmax        = 15.0

    rmin        = 0.0
    rmax        = 30.0
    rstep       = 0.01

    rpoly = 0.9
    
    print '\nNew cfg file has been created'
    os.chdir(dict['cfg_dir'])
    print 'Directory has been changed:'
    print os.getcwd()
    NAMES  = np.array(['[DEFAULT]','dataformat', 'outputtypes', 'composition', 'qmaxinst', 'qmin', 'qmax', 'rmin', 'rmax', 'rstep', 'rpoly'])
    FLOATS = np.array(['',dataformat, outputtypes, composition, qmaxinst, qmin, qmax, rmin, rmax, rstep, rpoly])
    DAT =  np.column_stack((NAMES, FLOATS))
    np.savetxt(cfg_name, DAT, delimiter=" = ", fmt="%s") 
    print 'A cfg file has been created, ' + str(dict['cfg_file'])
    sys.exit()

elif dict['PDF']:
    print '\nCfg file is being importet.'
    print '\t Values are needed for computation!!!'
    os.chdir(dict['cfg_dir'])
    print 'Directory has been changed:'
    print os.getcwd()

    cfg = loadPDFConfig(dict['cfg_file'])
    th_q_low  = cfg.qmin * 10
    th_q_high = cfg.qmax * 10
else:
    print 'Lowest q (in AA) value that will be tested for negative values:'
    while True:
        th_q_low  = input()
        if type(th_q_low) is not float:
            print "Answer needs to be a float"
        else:
            break

    print 'Highest q value (in AA) that will be tested for negative values:'
    while True:
        th_q_high  = input()
        if type(th_q_high) is not float:
            print "Answer needs to be a float"
        else:
            break

#---------------------------------------------------------------------------------------------------
#   
# Calibrating Background
#
#---------------------------------------------------------------------------------------------------

if dict['calib_bg']:
    print '\nCalib_bg is set to True!'
    '''
    If calib_bg = True, then all frames / measurements will be scaled with the same constant. 
    For determining the scalingsfactor for background subtration two methods can be done. It 
    can be done manually or automaticly. To do it automaticly set auto = True. If auto is set
    to true then the scaling factor defined in the init file does not do anything. If one 
    wants to manually subtract background it is recommended to save files binary to increase
    computation speed.
    '''
    if dict['auto']:
        '''
        Largest diviation is found and used to calculate the scaling factor. 
        The scaling factor is multiplied with 0.99 to ensure that there are no
        negative values.
        '''
        scale = 9999
        print 'Calculating scaling factor.'
        print 'Predetermined Scaling factor is ignored!'

        for li1, li2 in tqdm(zip(ydata_set, ybg_set)):
            diff_index = [] + heapq.nsmallest(len(li1), xrange(len(li1)), key=lambda i: ((li1[i] - li2[i])/(li1[i]+0.00001)))  # Finds index for largest difference 
            scale_ph = 9999
            scan_ph = 1  # Search til it finds a value within qmin and qmax
            i       = 0 
            while scan_ph == 1:

                if th_q_low < xdata_set[diff_index[i]] and th_q_high > xdata_set[diff_index[i]] and li1[diff_index[i]] != 0:
                    scale_ph = (li1[diff_index[i]] / li2[diff_index[i]]+0.00001) * 0.99  # Scales the background a bit further down
                    scan_ph = 0
                elif i == 1000:
                    print "Check data file! 1000 values are equal to 0."
                    sys.exit()               
                else:
                    i += 1

            if scale_ph < scale:
                scale = scale_ph
            del diff_index[:]
        
        print '\tScaling factor = ', scale
        scale = np.array(scale)
        scaled_bg = ybg_set.T * scale
        scaled_bg = scaled_bg.T
        y_diff = ydata_set[:] - scaled_bg
        y_diff = np.array(y_diff)    

        for i in range(dict['nr_files']):
            for j in range(len(y_diff[i])):
                if y_diff[i][j] <= 0 and th_q_low < xdata_set[j] and th_q_high > xdata_set[j] :
                    print 'Frame:', i,' X-val :',xdata_set[j], ' neg val: ', y_diff[i][j]

        lets_plot(xdata_set, ydata_set[0], xbg_set, scaled_bg[0], xdata_set, y_diff[0], xdata_set, ydata_set[-1], xbg_set, scaled_bg[-1], xdata_set, y_diff[-1], dict['save_pics'], dict['cfg_dir'])

     
###################
#    
#    if dict['auto']:  # Subtracts all points with a constant scaling so that no values are negative within the specified range
#        #y_diff = ydata_set[:] - 0#ybg_set[:]
#        #y_diff = np.array(y_diff)
#        
#        print np.amin(y_diff), 'heeeeeeeeeeeeeeej'
#
#        strc = np.zeros(len(y_diff), dtype=[('Neg Values', float), ('List', int), ('Index', int)])
#
#        for k in range(len(y_diff)):
#            lowest_neg = 0
#            #print 'k\n', k
#            #print y_diff[k]
#            
#            neg_vals = [j for j, i in enumerate(y_diff[k]) if i < 0]  # Find all negative values
#            #zero_vals = [j for j, i in enumerate(y_diff[k]) if i = 0]  # Find all negative values
#            #positive_vals = [j for j, i in enumerate(y_diff[k]) if i > 0]  # Find all negative values
#            
#            #print 'neg_vals\n', neg_vals
#            #for i in neg_vals:
#                #print xdata_set[i]
#                #continue
#            try: 
#                lowest_neg = np.amin([y_diff[k][i] for i in neg_vals if xdata_set[i] > th_q_low and xdata_set[i] < th_q_high])  # Find the largest negative values
#                #print 'lowest_ne\n', lowest_neg
#            except ValueError:
#                continue
#                #print 'No negative values between '+str(th_q_low)+ ' and ' + str(th_q_high) + ' AA.'
#            if lowest_neg > 0:
#                tallet = [i for i in neg_vals if y_diff[k][i] == lowest_neg]  # Returns index for largest negative value
#                print y_diff[k][tallet], xdata_set[tallet]
#                #print 'tallet\n', tallet
#                print 'Negative values between '+str(th_q_low)+ ' and ' + str(th_q_high) + ' AA at {:6.1f}'.format(k * totime) + ' m.'
#
#                v1 = fields_view(strc, ['Neg Values', 'List', 'Index'])
#                v1[k] = lowest_neg, k, tallet[0] 
#
#        strc.sort(order='Neg Values'#
 
    else:  # Takes a constant scaling factor and subtracts bg from data. The data is then checked for negative values     
        scale = dict['bg_scaling']
        scaled_bg = (ybg_set[:] * scale)
        y_diff = ydata_set[:] - scaled_bg
        y_diff = np.array(y_diff)
      
        neg_files, neg_vals = neg_check(y_diff, dict['first_file'], dict['cfg_dir'], dict['save_pics'], totime)
        if sum(neg_vals) > 0:
            print '\tWARNING! '
            if sum(neg_vals) > 1000:
                print '\tOver ' + str(int(sum(neg_vals)/1000)) +'000 values are negative.'
                print '\tConsider optimizing background subtraction'
            else: 
                print '\t'+sum(neg_vals) + ' are negative.'
        else: 
            print '\tNo values are negative'

        lets_plot(xdata_set, ydata_set[0], xbg_set, scaled_bg[0], xdata_set, y_diff[0], xdata_set, ydata_set[-1], xbg_set, scaled_bg[-1], xdata_set, y_diff[-1], dict['save_pics'], dict['cfg_dir'])

else:
    '''
    Generating Multiple scaling factors.
    '''
    print '\nGenerating multiple scaling factors.'
    if dict['auto']:
        '''
        Auto scaling for each frame must be implemented here.
        A plot over the scaling factor should be produced, so that the user can see if unnatural jumps occur
        ''' 
        count = 0
        scale_list = []
        for li1, li2 in zip(ydata_set, ybg_set):
            scale = 0
            diff_index = [] + heapq.nsmallest(len(li1), xrange(len(li1)), key=lambda i: ((li1[i] - li2[i])/(li1[i]+0.00001)))  # Finds index for largest difference between li1 and li2
            scan_ph = 1  # Search til it finds a value within qmin and qmax
            i       = 0 
            while scan_ph == 1:
                if th_q_low < xdata_set[diff_index[i]] and th_q_high > xdata_set[diff_index[i]] and li1[diff_index[i]] != 0:
                    scale = (li1[diff_index[i]] / li2[diff_index[i]]+0.00001) * 0.99  # Scales the background a bit further down
                    #print (li1[diff_index[i]]) ,' / ', (li2[diff_index[i]]+0.00001) 
                    scan_ph = 0
                
                i += 1
            scale_list.append(scale)
            count += 1
            del diff_index[:]
        y_diff = np.zeros((dict['nr_files'], steps))
        scaled_bg = np.zeros((dict['nr_files'], steps))
        for i in range(len(scale_list)):
            scaled_bg[i] = ybg_set[i] * scale_list[i] 
            y_diff[i] = ydata_set[i] - scaled_bg[i]
    
        y_diff = np.array(y_diff)
        np.set_printoptions(threshold=np.nan)
        for i in range(dict['nr_files']):
            print i,') ', scale_list[i]

        for i in range(dict['nr_files']):
            for j in range(len(y_diff[i])):
                if y_diff[i][j] <= 0 and th_q_low < xdata_set[j] and th_q_high > xdata_set[j] :
                    print 'Frame:', i,' X-val :',xdata_set[j], ' neg val: ', y_diff[i][j]
        #np.savetxt('Scaling'+str(dict['nr_files'])+'.txt', scale_list)            
     
        lets_plot(xdata_set, ydata_set[0], xbg_set, scaled_bg[0], xdata_set, y_diff[0], xdata_set, ydata_set[-1], xbg_set, scaled_bg[-1], xdata_set, y_diff[-1], dict['save_pics'], dict['cfg_dir'])


    else:
        scale_list = np.zeros(dict['nr_files'])
        y_diff       = []
        scaled_bg    = []

        print '\n', 'Scaling background:'
        for i in range(dict['nr_files']):
            a = []
            bound = 0.0
            while not a:
                scale_index = np.where((xdata_set > (float(dict['qnorm'])-bound)) & (xdata_set < (float(dict['qnorm'])+bound)))
                bound += 0.0001
                scale_index = np.array(scale_index)
                if np.isnan(scale_index) == False:
                    a.append(scale_index[0])
                    y_scaling = ydata_set[i][scale_index]
                    auto_scale = y_scaling / ybg_set[i][scale_index]
                    scale_list[i] = auto_scale

            scaled_bg.append(ybg_set[i] * scale_list[i])
            y_diff.append(ydata_set[i] - scaled_bg[i]) 

        neg_files, neg_vals = neg_check(y_diff, dict['first_file'], dict['cfg_dir'], dict['save_pics'], totime)
        if sum(neg_vals) > 0:
            print '\tWARNING! '
            if sum(neg_vals) > 1000:
                print '\tOver ' + str(int(sum(neg_vals)/1000)) +'000 values are negative.'
                print '\tConsider optimizing background subtraction'
            else: 
                print '\t'+sum(neg_vals) + ' are negative.'
        else: 
            print '\tNo values are negative'

        x = np.arange(dict['first_file'], (dict['first_file'] + dict['nr_files']))
        x = x * totime

        fig2, ax = plt.subplots(figsize = (14,6))
        ax.plot(x, scale_list, 'bo-', label='Scale Factor')

        ax.set_xlabel("Time [m]", fontsize=16) # the label of the y axis
        ax.set_ylabel('Scale factor', fontsize=16)  # the label of the y axis
        ax.set_title("Scale factor for each frame", fontsize=20) # the title of the plot
        ax.tick_params(axis='x', labelsize = 16)
        ax.tick_params(axis='y', labelsize = 16)
        ax.legend(loc='best', fontsize = 13)
        ax.set_xlim(dict['first_file']*totime, (dict['first_file'] + dict['nr_files'])*totime)

        if dict['save_pics']:
            pic_dir(dict['cfg_dir'], 'Pictures_')
            fig2.savefig('Scale_factor' + '.png')
            plt.clf()
        else:
            plt.draw()
            plt.pause(0.1)
            raw_input("<Hit Enter To Close>")
            plt.close(fig2)

        lets_plot(xdata_set, ydata_set[0], xbg_set, scaled_bg[0], xdata_set, y_diff[0], xdata_set, ydata_set[-1], xbg_set, scaled_bg[-1], xdata_set, y_diff[-1], dict['save_pics'], dict['cfg_dir'])

#---------------------------------------------------------------------------------------------------
#   
# Calculate PDF
#
#---------------------------------------------------------------------------------------------------

if dict['PDF']:
    if dict['make_cfg']:
        os.chdir(dict['cfg_dir'])
        print 'Directory has been changed:'
        print os.getcwd()
        NAMES  = np.array(['[DEFAULT]','dataformat', 'outputtypes', 'composition', 'qmaxinst', 'qmin', 'qmax', 'rmin', 'rmax', 'rstep', 'rpoly'])
        FLOATS = np.array(['',dataformat, outputtypes, composition, qmaxinst, qmin, qmax, rmin, rmax, rstep, rpoly])
        DAT =  np.column_stack((NAMES, FLOATS))
        np.savetxt(cfg_name + '.cfg', DAT, delimiter=" = ", fmt="%s") 
        cfg = loadPDFConfig(cfg_name + '.cfg')
    else:
        os.chdir(dict['cfg_dir'])
        print 'Directory has been changed:'
        print os.getcwd()

        cfg = loadPDFConfig(dict['cfg_file'])   

    pg = PDFGetter(config=cfg)
    
    if dict['Nyquist'] == True:
        pg.config.rstep = round(np.pi / cfg.qmax,3)
    
    q_matrix = np.ones((len(ydata_set), len(ydata_set[0])))
    q_matrix = q_matrix[:] * xdata_set

    q_iq= []
    iq  = []
    q_sq= []
    sq  = []
    q_fq= []
    fq  = [] 
    r   = []
    gr  = []

    for i in range(len(y_diff)):
        data_gr = pg(q_matrix[i], y_diff[i]) 

        q_iq.append(pg.iq[0])
        iq.append(pg.iq[1])

        q_sq.append(pg.sq[0])
        sq.append(pg.sq[1])         
        
        q_fq.append(pg.fq[0])
        fq.append(pg.fq[1])

        r.append(data_gr[0])
        gr.append(data_gr[1])        

    q_iq = np.array(q_iq)
    iq = np.array(iq)

    q_sq = np.array(q_sq)
    sq = np.array(sq)           
    
    q_fq = np.array(q_fq)
    fq = np.array(fq)   

    r = np.array(r)
    gr = np.array(gr)   

    if dict['gen_PDF_file']:
        print "\nGenerating G(r) files!"
        pic_dir(dict['cfg_dir'], 'Gr_')
        head_name  = np.array(['# Composition', '# qmaxinst', '# qmin', '# qmax', '# rmin', '# rmax', '# Nyquist', '# rstep', '# rpoly', '# Data', '# Background','# Scaling', '#'])
        index = 0
        k     = 0
        for i in tqdm(range(dict['nr_files'])):            
            if (dict['calib_bg'] == False and dict['auto'] == True) or dict['calib_bg'] == False:
                if dict['seq_bg']:
                    if i == extend[index] -1 and not i == extend[-1] -1:
                        index += 1
                        k += 1
                    elif i == extend[-1] -1:
                        k = len(extend)
                    head_vals  = np.array([cfg.composition, cfg.qmaxinst, cfg.qmin, cfg.qmax, cfg.rmin, cfg.rmax, dict['Nyquist'], cfg.rstep, cfg.rpoly, data_name[i], bg_name[k], scale_list[i],''])
                
                else: 
                    k = i
                    if i > dict['nr_bg_files'] - 1:
                        k = dict['nr_bg_files'] - 1
                    head_vals  = np.array([cfg.composition, cfg.qmaxinst, cfg.qmin, cfg.qmax, cfg.rmin, cfg.rmax, dict['Nyquist'], cfg.rstep, cfg.rpoly, data_name[i], bg_name[k], scale_list[i],''])
            
            else:
                if dict['seq_bg']:
                    if i == extend[index] -1 and not i == extend[-1] -1:
                        index += 1
                        k += 1
                    elif i == extend[-1] -1:
                        k = len(extend)
                    head_vals  = np.array([cfg.composition, cfg.qmaxinst, cfg.qmin, cfg.qmax, cfg.rmin, cfg.rmax, dict['Nyquist'], cfg.rstep, cfg.rpoly, data_name[i], bg_name[k], scale,''])
                
                else: 
                    k = i
                    if i > dict['nr_bg_files'] - 1:
                        k = dict['nr_bg_files'] - 1
                    head_vals  = np.array([cfg.composition, cfg.qmaxinst, cfg.qmin, cfg.qmax, cfg.rmin, cfg.rmax, dict['Nyquist'], cfg.rstep, cfg.rpoly, data_name[i], bg_name[k], scale,''])
 
            header     = np.column_stack((head_name, head_vals))
            saving_dat = np.column_stack((r[i],gr[i])) 
            saving_dat = (np.vstack(((header).astype(str), (saving_dat).astype(str))))
            np.savetxt(dict['file_name'] + str(i).zfill(3) +'.gr', saving_dat, fmt='%s')

    if dict['gen_fq_file']:
        print "\nGenerating F(q) files!"
        pic_dir(dict['cfg_dir'], 'Fq_')
        index = 0
        k     = 0
        for i in tqdm(range(dict['nr_files'])):
            if (dict['calib_bg'] == False and dict['auto'] == True) or dict['calib_bg'] == False:
                if dict['seq_bg']:
                    if i == extend[index] -1 and not i == extend[-1] -1:
                        index += 1
                        k += 1
                    elif i == extend[-1] -1:
                        k = len(extend)
                    head_vals  = np.array([cfg.composition, cfg.qmaxinst, cfg.qmin, cfg.qmax, cfg.rmin, cfg.rmax, dict['Nyquist'], cfg.rstep, cfg.rpoly, data_name[i], bg_name[k], scale_list[i],''])
                
                else: 
                    k = i
                    if i > dict['nr_bg_files'] - 1:
                        k = dict['nr_bg_files'] - 1
                    head_vals  = np.array([cfg.composition, cfg.qmaxinst, cfg.qmin, cfg.qmax, cfg.rmin, cfg.rmax, dict['Nyquist'], cfg.rstep, cfg.rpoly, data_name[i], bg_name[k], scale_list[i],''])
            
            else:
                if dict['seq_bg']:
                    if i == extend[index] -1 and not i == extend[-1] -1:
                        index += 1
                        k += 1
                    elif i == extend[-1] -1:
                        k = len(extend)
                    head_vals  = np.array([cfg.composition, cfg.qmaxinst, cfg.qmin, cfg.qmax, cfg.rmin, cfg.rmax, dict['Nyquist'], cfg.rstep, cfg.rpoly, data_name[i], bg_name[k], scale,''])
                
                else: 
                    k = i
                    if i > dict['nr_bg_files'] - 1:
                        k = dict['nr_bg_files'] - 1
                    head_vals  = np.array([cfg.composition, cfg.qmaxinst, cfg.qmin, cfg.qmax, cfg.rmin, cfg.rmax, dict['Nyquist'], cfg.rstep, cfg.rpoly, data_name[i], bg_name[k], scale,''])

            header     = np.column_stack((head_name, head_vals))           
            saving_dat = np.column_stack((q_fq[i],fq[i])) 
            saving_dat = (np.vstack(((header).astype(str), (saving_dat).astype(str))))
            np.savetxt(dict['file_name'] + str(i).zfill(3) +'.fq', saving_dat, fmt='%s')

    if dict['gen_iq_file']:
        print "\nGenerating I(q) files!"
        pic_dir(dict['cfg_dir'], 'Iq_')
        index = 0
        k     = 0
        for i in tqdm(range(dict['nr_files'])):
            if (dict['calib_bg'] == False and dict['auto'] == True) or dict['calib_bg'] == False:
                if dict['seq_bg']:
                    if i == extend[index] -1 and not i == extend[-1] -1:
                        index += 1
                        k += 1
                    elif i == extend[-1] -1:
                        k = len(extend)
                    head_vals  = np.array([cfg.composition, cfg.qmaxinst, cfg.qmin, cfg.qmax, cfg.rmin, cfg.rmax, dict['Nyquist'], cfg.rstep, cfg.rpoly, data_name[i], bg_name[k], scale_list[i],''])
                
                else: 
                    k = i
                    if i > dict['nr_bg_files'] - 1:
                        k = dict['nr_bg_files'] - 1
                    head_vals  = np.array([cfg.composition, cfg.qmaxinst, cfg.qmin, cfg.qmax, cfg.rmin, cfg.rmax, dict['Nyquist'], cfg.rstep, cfg.rpoly, data_name[i], bg_name[k], scale_list[i],''])
            
            else:
                if dict['seq_bg']:
                    if i == extend[index] -1 and not i == extend[-1] -1:
                        index += 1
                        k += 1
                    elif i == extend[-1] -1:
                        k = len(extend)
                    head_vals  = np.array([cfg.composition, cfg.qmaxinst, cfg.qmin, cfg.qmax, cfg.rmin, cfg.rmax, dict['Nyquist'], cfg.rstep, cfg.rpoly, data_name[i], bg_name[k], scale,''])
                
                else: 
                    k = i
                    if i > dict['nr_bg_files'] - 1:
                        k = dict['nr_bg_files'] - 1
                    head_vals  = np.array([cfg.composition, cfg.qmaxinst, cfg.qmin, cfg.qmax, cfg.rmin, cfg.rmax, dict['Nyquist'], cfg.rstep, cfg.rpoly, data_name[i], bg_name[k], scale,''])
            header     = np.column_stack((head_name, head_vals))    
            saving_dat = np.column_stack((q_iq[i],iq[i])) 
            saving_dat = (np.vstack(((header).astype(str), (saving_dat).astype(str))))
            np.savetxt(dict['file_name'] + str(i).zfill(3) +'.iq', saving_dat, fmt='%s')

    timeresolved = (np.array(range(len(y_diff))) + dict['first_file'])*totime 
    timeresolved_q = (np.array(range(len(fq))) + dict['first_file'])*totime

    fig = plt.figure(figsize=(12, 6))
    print '\nMaking PDF Grid Plot'
    #Plot f(Q)
    plt.subplot(211)
    X, Y = np.meshgrid(q_fq[0], timeresolved_q)

    Z=(fq)
    plt.pcolormesh(X, Y, Z)
    plt.ylabel('Time [m]', fontsize=18)
    plt.xlabel('Q [$\AA^-$$^1$]', fontsize=18)
    plt.axis([np.amin(q_fq[0]), np.amax(q_fq[0]), dict['first_file']*totime, ((dict['nr_files'] + dict['first_file'])-1)*totime])
    plt.title('F(Q)', fontsize=20)
    plt.colorbar()

    # Now the PDFs.
    plt.subplot(212)
    X, Y = np.meshgrid(r[0], timeresolved)
    Z = (gr)
    plt.pcolormesh(X, Y, Z) 
    plt.ylabel('Time [m]', fontsize=18)
    plt.xlabel('r [$\AA$]', fontsize=18)
    plt.axis([np.amin(r[0]), np.amax(r[0]), dict['first_file']*totime, ((dict['nr_files'] + dict['first_file'])-1)*totime])
    plt.title('G(r)', fontsize=20)
    plt.colorbar()
    plt.tight_layout()

    fig.subplots_adjust(hspace=0.65)

    if dict['save_pics']:
        pic_dir(dict['cfg_dir'], 'Pictures_')
        fig.savefig(dict['PDF_name'] + '.png')
        plt.clf()
    else:
        plt.draw()
        plt.pause(0.1)
        raw_input("<Hit Enter To Close>")
        plt.close()

    if dict['show_PDF']: 
        print '\nSaving PDF plot of ' + str(dict['first_file'] + dict['pdf_file'])
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(r[0], gr[dict['pdf_file']], 'o-', label='Data')

        ax1.set_xlabel("G(r) [$\AA^-$$^1$]", fontsize=16) # the label of the y axis
        ax1.set_ylabel('Intensity [a.u.]', fontsize=16)  # the label of the y axis
        ax1.set_title("PDF of frame " + str(dict['pdf_file']), fontsize=20) # the title of the plot
        ax1.tick_params(axis='x', labelsize = 16)
        ax1.tick_params(axis='y', labelsize = 16)
        ax1.legend(loc='best', fontsize = 13)

        if dict['save_pics']:
            pic_dir(dict['cfg_dir'], 'Pictures_')
            fig1.savefig('Single_PDF' + '.png')
            plt.clf()
        else:
            plt.draw()
            plt.pause(0.1)
            raw_input("<Hit Enter To Close>")
            plt.close()

    if dict['show_all']:
        print '\nSaving I(q), f(q), S(q) and G(r) plot of ' + str(dict['pdf_file'])
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (16,8))

        ax1.plot(q_iq[0], iq[(dict['pdf_file'] - 1)-dict['first_file']], label='Frame ' + str(dict['pdf_file'] + 1), color='blue')
        ax1.set_xlabel('Q [$\AA^-$$^1$]', fontsize=16)
        ax1.set_ylabel('Intensity [a.u.]', fontsize=16)
        ax1.legend(loc='best', fontsize=14)
        ax1.tick_params(axis='x', labelsize = 16)
        ax1.tick_params(axis='y', labelsize = 16)
        ax1.set_title('i(Q)', fontsize=20)
        ax1.set_xlim(q_iq[0][0], q_iq[0][-1])

        ax2.plot(q_sq[0], sq[(dict['pdf_file'] - 1)-dict['first_file']], label='Frame ' + str(dict['pdf_file'] + 1), color='green')
        ax2.set_xlabel('Q [$\AA^-$$^1$]', fontsize=16)
        #ax2.set_ylabel('Intensity [a.u.]', fontsize=16)
        ax2.legend(loc='best', fontsize=14)
        ax2.tick_params(axis='x', labelsize = 16)
        ax2.tick_params(axis='y', labelsize = 16)
        ax2.set_title('s(Q)', fontsize=20)
        ax2.set_xlim(q_sq[0][0], q_sq[0][-1])

        ax3.plot(q_fq[0], fq[(dict['pdf_file'] - 1)-dict['first_file']], label='Frame ' + str(dict['pdf_file'] + 1), color='black')
        ax3.set_xlabel('Q [$\AA^-$$^1$]', fontsize=16)
        ax3.set_ylabel('Intensity [a.u.]', fontsize=16)
        ax3.legend(loc='best', fontsize=14)
        ax3.tick_params(axis='x', labelsize = 16)
        ax3.tick_params(axis='y', labelsize = 16)
        ax3.set_title('f(Q)', fontsize=20)
        ax3.set_xlim(q_fq[0][0], q_fq[0][-1])

        ax4.plot(r[0], gr[(dict['pdf_file'] - 1)-dict['first_file']], label='Frame ' + str(dict['pdf_file'] + 1), color='red')
        ax4.set_xlabel('r [$\AA$]', fontsize=16)
        #ax4.set_ylabel('Intensity [a.u.]', fontsize=16)
        ax4.legend(loc='best', fontsize=14)
        ax4.tick_params(axis='x', labelsize = 16)
        ax4.tick_params(axis='y', labelsize = 16)
        ax4.set_title('G(r)', fontsize=20)
        ax4.set_xlim(r[0][0], r[0][-1])

        plt.tight_layout(pad=2, w_pad=1.0, h_pad=1.0)       
        
        if dict['save_pics']:
            pic_dir(dict['cfg_dir'], 'Pictures_')
            fig2.savefig('All' + '.png')
            plt.clf()
        else:
            plt.draw()
            plt.pause(0.1)
            raw_input("<Hit Enter To Close>")
            plt.close()

#---------------------------------------------------------------------------------------------------
#   
# 3D Plot
#
#---------------------------------------------------------------------------------------------------

    if dict['3D_plot']:
        print '\nCreating 3D figure - be patient!'
        X = np.array(r[0])
        Y = np.arange(0,len(timeresolved))
        X, Y = np.meshgrid(X, Y * totime)
        z = gr
        Z = np.array(gr)
       
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=str(dict['3D_cmap']), linewidth=0, antialiased=False)

        ax.set_zticklabels([])

        cbar = plt.colorbar(surf, shrink=0.5, aspect=5)
        cbar.set_ticks([])
        cbar.set_ticklabels([])
        cbar.set_label('Intensity [a.u.]')
        
        plt.title(str(dict['3D_title']))
        
        ax.set_xlabel('r [$\AA$]')
        ax.set_ylabel('Time [m]')

        ax.set_xlim(np.amin(r[0]), np.amax(r[0]))
        ax.set_zlim(np.amin(Z), np.amax(Z))

        fig.tight_layout()

        if dict['save_pics']:
            pic_dir(dict['cfg_dir'], 'Pictures_')
            fig.savefig('3d' + '.png')
            plt.clf()
        else:
            plt.draw()
            plt.pause(0.1)
            raw_input("<Hit Enter To Close>")
            plt.close()

#---------------------------------------------------------------------------------------------------
#   
# Save Dictionary
#
#---------------------------------------------------------------------------------------------------

if save_dict:
    pic_dir(dict['cfg_dir'], 'Pictures_')
    pickle.dump(dict, open(dict_name + ".py", "wb" ))
