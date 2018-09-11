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
  Date:         11/09/2018
  ----------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import errno
import pickle
import ConfigParser
from prompter import yesno
from tqdm import tqdm
from diffpy.pdfgetx import PDFGetter, loadPDFConfig, findfiles

#---------------------------------------------------------------------------------------------------
#	
# Functions
#
#---------------------------------------------------------------------------------------------------

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
	dir_path = dir_path + '\\' + str(folder_name) + str(dict['file_name'])

	if not os.path.exists(dir_path):
		os.makedirs(dir_path)

	os.chdir(dir_path)

def read_data(frame_sumstart, nr_files, file_name, file_type, line_skip):
	"""

	"""
	x_values = []
	y_values = []
	min_val	 = []
	max_val	 = []
	dim = 1
	lendat1 = 0

	print 'Loading files:'
	for i in tqdm(range(frame_sumstart, nr_files)):
		frame = file_name + str.zfill(str(i+1), 5) + str(file_type)
		frame_data = np.loadtxt(frame, skiprows = line_skip)
		x_values.append(frame_data[:,0])
		y_values.append(frame_data[:,1])
		min_val.append(frame_data[0,0])
		max_val.append(frame_data[-1,0])

		lendat2 = len(y_values[i])
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
		
	return x_values, y_values, dim, min_val, max_val, lendat1

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
		sum_mat.append(np.sum(data[placeholder - sumstep:placeholder, 0::], axis=0))

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

if os.path.isfile('TimeResolved_config.ini'):
	timeResCon = True
	parser 	   = ConfigParser.ConfigParser()
	parser.read('TimeResolved_config.ini')
	load_dict  = parser.getboolean('Dictionary', 'load_dict')
	imp_dict   = parser.get('Dictionary', 'imp_dict')
else:
	load_dict  = False
	timeResCon = False
	imp_dict   = 'XXX_directory.py'
	print "No TimeResolved_config.ini exists!"
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
	print '{0:12s} {1} {2}'.format('data_magic', '= ', dict['data_magic'])
	print '{0:12s} {1} {2}'.format('save_data', '= ', dict['save_data'])
	print '{0:12s} {1} {2}'.format('PDF', '= ', dict['PDF'])
	print '{0:12s} {1} {2}'.format('gen_PDF_file', '= ', dict['gen_PDF_file'])
	print '{0:12s} {1} {2}'.format('gen_fq_file', '= ', dict['gen_fq_file'])
	print '{0:12s} {1} {2}'.format('gen_iq_file', '= ', dict['gen_iq_file'])

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

	print '\n[Scaling]'
	print '{0:12s} {1} {2}'.format('calib_bg', '= ', dict['calib_bg'])
	print '{0:12s} {1} {2}'.format('sumstep', '= ', dict['sumstep'])
	print '{0:12s} {1} {2}'.format('bg_scaling', '= ', dict['bg_scaling'])
	print '{0:12s} {1} {2}'.format('qnorm', '= ', dict['qnorm'])

	print '\n[Directories]'
	print '{0:12s} {1} {2}'.format('change_dir', '= ', dict['change_dir'])
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
	print '{0:12s} {1} {2}'.format('PDF_name', '= ', dict['PDF_name'])

	print '\nThe sections [Dictionary] and [Save_Dictionary] are not stored within the Dictionary!'

	yesno('Type yes to confirm Directory and no to terminate')

	print "Imported Dictionary has been verfied. Proceeding!"
	save_dict = False

elif timeResCon:
	parser = ConfigParser.ConfigParser()
	dict = {}
	parser.read('TimeResolved_config.ini')

	# [Main]
	dict['data_magic']	= parser.getboolean('Main', 'data_magic')
	dict['save_data']	= parser.getboolean('Main', 'save_data')
	dict['PDF'] 		= parser.getboolean('Main', 'PDF')
	dict['gen_PDF_file']= parser.getboolean('Main', 'gen_PDF_file')
	dict['gen_fq_file'] = parser.getboolean('Main', 'gen_fq_file')
	dict['gen_iq_file'] = parser.getboolean('Main', 'gen_iq_file')

	# [Data]
	dict['file_name'] 	= parser.get('Data', 'file_name')
	dict['file_type']	= parser.get('Data', 'file_type') 
	dict['first_file']	= parser.getint('Data', 'first_file')
	dict['nr_files']	= parser.getint('Data', 'nr_files')
	dict['line_skip']	= parser.getint('Data', 'line_skip')

	# [Background_data]
	dict['bg_file']		= parser.get('Background_Data', 'bg_file')
	dict['bg_type']		= parser.get('Background_Data', 'bg_type') 
	dict['first_bg']	= parser.getint('Background_Data', 'first_bg')
	dict['nr_bg_files']	= parser.getint('Background_Data', 'nr_bg_files')
	dict['bgline_skip']	= parser.getint('Background_Data', 'bgline_skip')																																		

	# [Scaling]
	dict['calib_bg']	= parser.getboolean('Scaling', 'calib_bg')	
	dict['sumstep'] 	= parser.getint('Scaling', 'sumstep')
	dict['bg_scaling'] 	= parser.getfloat('Scaling', 'bg_scaling')
	dict['qnorm'] 		= parser.get('Scaling', 'qnorm')

	# [Directories]
	dict['change_dir']	= parser.getboolean('Directories', 'change_dir')
	dict['data_dir'] 	= parser.get('Directories', 'data_dir')										
	dict['bg_dir']		= parser.get('Directories', 'bg_dir')
	dict['cfg_dir']		= parser.get('Directories', 'cfg_dir')

	# [PDFgetX3]
	dict['make_cfg'] 	= parser.getboolean('PDFgetX3', 'make_cfg')
	dict['cfg_file']	= parser.get('PDFgetX3', 'cfg_file')

	# [Plotting]
	dict['show_PDF'] 	= parser.getboolean('Plotting', 'show_PDF')
	dict['show_all']	= parser.getboolean('Plotting', 'show_all')	
	dict['save_pics'] 	= parser.getboolean('Plotting', 'save_pics')
	dict['pdf_file'] 	= parser.getint('Plotting', 'pdf_file')	
	dict['PDF_name']	= parser.get('Plotting', 'PDF_name')

	# [Save_Dictionary]
	save_dict 			= parser.getboolean('Save_Dictionary', 'save_dict')
	dict_name 			= parser.get('Save_Dictionary', 'dict_name')

else:
	dict = {}

	# [Main]
	dict['data_magic']	= True																					#Check all data, and makes sure it matches in lengths and size
	dict['save_data']	= False																					#Should save data in right format, does nothing at the moment
	dict['PDF'] 		= True																					#Calculates PDF
	dict['gen_PDF_file']= True
	dict['gen_fq_file'] = True
	dict['gen_iq_file'] = True

	# [Data]
	dict['file_name'] 	= 'BA_WCl6_160_p-'																		#Starting name of files you want inported, e.g. 'BA_WCl6_200-', full name 'BA_WCl6_200-00001'.
	dict['file_type']	= '.xy'																					#Type of file you want imported. Remember '.' in front, e.g. '.xy' 
	dict['first_file']	= 0																						#First file, 0 = 1.
	dict['nr_files']	= 6																					#Number of files you want to import
	dict['line_skip']	= 16																					#Amount of header lines that will be skiped, 16 for .xy- and 4 for .chi-files

	# [Background_Data]
	dict['bg_file']		= 'BA_BKG_160_p-'																		#Name of the background file. So far only possible to take one file, so full file name
	dict['bg_type']		= '.xy'																					#Type of file you want imported. Remember '.' in front, e.g. '.xy' 
	dict['first_bg']	= 0																						#First file, 0 = 1.
	dict['nr_bg_files']	= 3																					#Number of files you want to import
	dict['bgline_skip']	= 16	


	#[Scaling]
	dict['calib_bg']	= True																					#If false, autoscale at qnorm	
	dict['sumstep'] 	= 1																						#Summing files to increase intensity. If = 1, then no summation will be done																					
	dict['bg_scaling'] 	= 0.98																					#Constant scaling of bagground
	dict['qnorm'] 		= 22																					#Define the point in q, where the background should line up with the data

	# [Directories]
	dict['change_dir']	= True 																					#True if you want to change directory
	dict['data_dir'] 	= 'C:\\Users\\opadn\\Documents\\Skolefiler\\KU_Nanoscience\\Kandidat\\Masters\\Data_WO\\BA_WCl6_160_p'	#Insert path to a new directory										
	dict['bg_dir']		= 'C:\Users\opadn\Documents\Skolefiler\KU_Nanoscience\Kandidat\Masters\Data_WO\BA_BKG_160_p'
	dict['cfg_dir']		= 'C:\Users\opadn\Documents\Skolefiler\KU_Nanoscience\Kandidat\Masters\TimeResolved_scripts'

	# [PDFgetX3]
	dict['make_cfg'] 	= False
	dict['cfg_file']	= 'pdfgetx3.cfg'

	# [Plotting]
	dict['show_PDF'] 	= True																					#Shows PDF
	dict['show_all']	= True																					#Shows iq, sq, fq and Gr for a specific file, pdf_file
	dict['save_pics'] 	= True
	dict['pdf_file'] 	= -1																					#Show the pdf of a specific file		
	dict['PDF_name']	= 'WCl6_160_p'

	# [Save_Dictionary]
	save_dict 			= True
	dict_name 			= str(XXX_directory)

#---------------------------------------------------------------------------------------------------
#	
# Create cfg file
#
#--------------------------------------------------------------------------------------------------- 

if dict['make_cfg'] and load_dict == False:
	cfg_name    = 'pdfgetx3_new'

	dataformat  = 'QA'
	outputtypes = 'iq, sq, fq, gr'
	composition = 'W Cl6'

	qmaxinst    = 19.0
	qmin 		= 0.8
	qmax 		= 15.0

	rmin 		= 0.0
	rmax 		= 30.0
	rstep 		= 0.01

	rpoly 		= 0.9

#---------------------------------------------------------------------------------------------------
#	
# Code
#
#--------------------------------------------------------------------------------------------------- 

owd = os.getcwd()

#---------------------------------------------------------------------------------------------------
#	
# Import Data and Construct Arrays
#
#---------------------------------------------------------------------------------------------------

if dict['change_dir']:
	os.chdir(dict['data_dir'])
	print '\nDirectory has been changed:'
	print os.getcwd()

xdata_set, ydata_set, data_dim, min_val_data, max_val_data, data_len = read_data(dict['first_file'], dict['nr_files'], dict['file_name'], dict['file_type'], dict['line_skip'])

if dict['change_dir']:
	os.chdir(dict['bg_dir'])
	print '\nDirectory has been changed:'
	print os.getcwd()

xbg_set, ybg_set, bg_dim, min_val_bg, max_val_bg, bg_len = read_data(dict['first_bg'], dict['nr_bg_files'], dict['bg_file'], dict['bg_type'], dict['bgline_skip'])	

if bg_len >= data_len:
	steps = bg_len * 2
else:
	steps = data_len * 2

if dict['data_magic']:																						#Find highest min value and lowest max value for interpolation
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
	ybg_set_int = np.zeros((dict['nr_bg_files'], steps))

	if data_dim == 0 and bg_dim == 1:
		print '\n', 'Background files vary in length.'
		print 'Interpolating data files:'
		for i in tqdm(range(0, dict['nr_files'] - dict['first_file'])):
			xdata_set_int, y_int = interpol(xmin, xmax, steps, xdata_set, ydata_set[i])
			ydata_set_int[i] = y_int

		print 'Interpolating background files:'
		for i in tqdm(range(0, dict['nr_bg_files'] - dict['first_bg'])):	
			_, ybg_int = interpol(xmin, xmax, steps, xbg_set[i], ybg_set[i])
			ybg_set_int[i] = ybg_int

		xdata_set = xdata_set_int
		ydata_set = ydata_set_int
		xbg_set = xdata_set
		ybg_set = ybg_set_int  

	elif data_dim == 1 and bg_dim == 0:
		print '\n', 'Data files vary in length.'
		print 'Interpolating data files:'
		for i in tqdm(range(0, dict['nr_files'] - dict['first_file'])):
			xdata_set_int, y_int = interpol(xmin, xmax, steps, xdata_set[i], ydata_set[i])
			ydata_set_int[i] = y_int

		print 'Interpolating background files:'
		for i in tqdm(range(0, dict['nr_bg_files'] - dict['first_bg'])):	
			_, ybg_int = interpol(xmin, xmax, steps, xbg_set, ybg_set[i])
			ybg_set_int[i] = ybg_int

		xdata_set = xdata_set_int
		ydata_set = ydata_set_int
		xbg_set = xdata_set
		ybg_set = ybg_set_int 

	elif data_dim == 1 and bg_dim == 1:
		print '\n', 'Size of data and background array does not match.'
		print 'Interpolating data files:'
		for i in tqdm(range(0, dict['nr_files'] - dict['first_file'])):
			xdata_set_int, y_int = interpol(xmin, xmax, steps, xdata_set[i], ydata_set[i])
			ydata_set_int[i] = y_int

		print 'Interpolating bachground files:'
		for i in tqdm(range(0, dict['nr_bg_files'] - dict['first_bg'])):	
			_, ybg_int = interpol(xmin, xmax, steps, xbg_set[i], ybg_set[i])
			ybg_set_int[i] = ybg_int

		xdata_set = xdata_set_int
		ydata_set = ydata_set_int
		xbg_set = xdata_set
		ybg_set = ybg_set_int 

	else:
		print 'All data have same dimension'
		if np.array_equal(xdata_set, xbg_set) == True:
			print 'No need for interpolation'
		else:
			print 'lort'
	if dict['save_data']:
		print 'Coming in next patch'

if dict['sumstep'] > 1:
	ydata_set = sum_data(dict['nr_files'], dict['sumstep'], ydata_set)
	ybg_set = sum_data(dict['nr_bg_files'], dict['sumstep'], ybg_set)
	dict['nr_files'] = (dict['nr_files']/dict['sumstep'])+1
	dict['nr_bg_files'] = (dict['nr_bg_files']/dict['sumstep'])+1

ybg_set = np.reshape(ybg_set, (dict['nr_bg_files'], steps))

if dict['nr_files'] > dict['nr_bg_files']:																			#If there are less background files the data files exstend bg matrix with last background row til they match
	add_bgy = ybg_set[-1]
	add_bgy = np.reshape(add_bgy, (1, steps))

	print '\n', 'Extending background matrix:'
	for i in tqdm(range(dict['nr_files'] - dict['nr_bg_files'])):
		ybg_set = np.concatenate((ybg_set, add_bgy), axis = 0)

#---------------------------------------------------------------------------------------------------
#	
# Calibrating Background
#
#---------------------------------------------------------------------------------------------------

if dict['calib_bg']:
	scaled_bg = (ybg_set[:] * dict['bg_scaling'])
	y_diff = ydata_set[:] - scaled_bg
	y_diff = np.array(y_diff)
	n = 0

	for i in range(dict['nr_files']):
		if n < 3 and any(t < 0 for t in y_diff[i]): 
			print('WARNING!!! Subtracted data contains negative values!!!')
			n += 1

	lets_plot(xdata_set, ydata_set[0], xbg_set, scaled_bg[0], xdata_set, y_diff[0], xdata_set, ydata_set[-1], xbg_set, scaled_bg[-1], xdata_set, y_diff[-1], dict['save_pics'], owd)
else:
	scale_factor = np.zeros([nr_files])
	y_diff	     = []
	scaled_bg	 = []

	print '\n', 'Scaling background:'
	for i in tqdm(range(dict['nr_files'])):
		a = []
		bound = 0.0
		while not a:
			scale_index = np.where((xdata_set > (dict['qnorm']-bound)) & (xdata_set < (dict['qnorm']+bound)))
			bound += 0.0001
			scale_index = np.array(scale_index)
			if np.isnan(scale_index) == False:
				a.append(scale_index[0])
				y_scaling = ydata_set[i][scale_index]
				auto_scale = y_scaling / ybg_set[i][scale_index]
				scale_factor[i] = auto_scale

		scaled_bg.append(ybg_set[i] * scale_factor[i])
		y_diff.append(ydata_set[i] - scaled_bg[i]) 
	#print scale_factor

	n = 0
	for i in range(dict['nr_files']):
		if n < 3 and any(t < 0 for t in y_diff[i]): 
			print('WARNING!!! Subtracted data contains negative values!!!')
			n += 1

	fig2, ax = plt.subplots(figsize = (12,6))
	ax.plot(scale_factor, 'bo-', label='Scale Factor')

	ax.set_xlabel("Frame", fontsize=16) # the label of the y axis
	ax.set_ylabel('Scale factor', fontsize=16)  # the label of the y axis
	ax.set_title("Scale factor for each frame", fontsize=20) # the title of the plot
	ax.tick_params(axis='x', labelsize = 16)
	ax.tick_params(axis='y', labelsize = 16)
	ax.legend(loc='best', fontsize = 13)
	ax.set_xlim(-1, nr_files)

	lets_plot(xdata_set, ydata_set[0], xbg_set, scaled_bg[0], xdata_set, y_diff[0], xdata_set, ydata_set[-1], xbg_set, scaled_bg[-1], xdata_set, y_diff[-1])

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
		print '\nDirectory has been changed:'
		print os.getcwd()

		cfg = loadPDFConfig(dict['cfg_file'])	

	pg = PDFGetter(config=cfg)
	
	q_matrix = np.ones((len(ydata_set), len(ydata_set[0])))
	q_matrix = q_matrix[:] * xdata_set

	q_iq= []
	iq 	= []
	q_sq= []
	sq 	= []
	q_fq= []
	fq 	= [] 
	r 	= []
	gr 	= []

	for i in range(len(y_diff)):
		data_gr = pg(q_matrix[i], y_diff[i]) 			#pg(q_matrix[i], y_diff[i]) 

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
		print "Generating G(r) files!"
		pic_dir(owd, 'Gr_')
		for i in tqdm(range(dict['nr_files'])):
			np.savetxt(dict['file_name'] + str(i).zfill(3) +'.gr',np.column_stack((r[i],gr[i])))

	if dict['gen_fq_file']:
		print "Generating F(q) files!"
		pic_dir(owd, 'Fq_')
		for i in tqdm(range(dict['nr_files'])):
			np.savetxt(dict['file_name'] + str(i).zfill(3) +'.fq',np.column_stack((q_fq[i],fq[i])))

	if dict['gen_iq_file']:
		print "Generating I(q) files!"
		pic_dir(owd, 'Iq_')
		for i in tqdm(range(dict['nr_files'])):
			np.savetxt(dict['file_name'] + str(i).zfill(3) +'.iq',np.column_stack((q_iq[i],iq[i])))			

	timeresolved = range(len(y_diff))
	timeresolved_q = range(len(fq))

	fig = plt.figure(figsize=(12, 6))
	
	#Plot f(Q)
	plt.subplot(211)
	X, Y = np.meshgrid(q_fq[0], timeresolved_q)
	Z=(fq)
	plt.pcolormesh(X, Y, Z)
	plt.ylabel('Frame', fontsize=18)
	plt.xlabel('Q [$\AA^-$$^1$]', fontsize=18)
	plt.axis([np.amin(q_fq[0]), np.amax(q_fq[0]), dict['first_file'], (dict['nr_files'] - dict['first_file']) - 1])
	plt.title('F(Q)', fontsize=20)
	plt.colorbar()

	# Now the PDFs.
	plt.subplot(212)
	X, Y = np.meshgrid(r[0], timeresolved)
	Z = (gr)
	plt.pcolormesh(X, Y, Z) 
	plt.ylabel('Frame', fontsize=18)
	plt.xlabel('r [$\AA$]', fontsize=18)
	plt.axis([np.amin(r[0]), np.amax(r[0]), dict['first_file'], (dict['nr_files'] - dict['first_file']) - 1])
	plt.title('G(r)', fontsize=20)
	plt.colorbar()

	fig.subplots_adjust(hspace=0.65)

	if dict['save_pics']:
		pic_dir(owd, 'Pictures_')
		fig.savefig(dict['PDF_name'] + '.png')
		plt.clf()
	else:
		plt.draw()
		plt.pause(0.1)
		raw_input("<Hit Enter To Close>")
		plt.close()

	if dict['show_PDF']: 
		fig1, ax1 = plt.subplots(figsize=(12, 6))
		ax1.plot(r[0], gr[dict['pdf_file'] - 1], label='Data')

		ax1.set_xlabel("G(r) [$\AA^-$$^1$]", fontsize=16) # the label of the y axis
		ax1.set_ylabel('Intensity [a.u.]', fontsize=16)  # the label of the y axis
		ax1.set_title("PDF of frame " + str(dict['pdf_file'] + 1), fontsize=20) # the title of the plot
		ax1.tick_params(axis='x', labelsize = 16)
		ax1.tick_params(axis='y', labelsize = 16)
		ax1.legend(loc='best', fontsize = 13)

		if dict['save_pics']:
			pic_dir(owd, 'Pictures_')
			fig1.savefig('Single_PDF' + '.png')
			plt.clf()
		else:
			plt.draw()
			plt.pause(0.1)
			raw_input("<Hit Enter To Close>")
			plt.close()

	if dict['show_all']:
		fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (16,8))

		ax1.plot(q_iq[0], iq[dict['pdf_file'] - 1], label='Frame ' + str(dict['pdf_file'] + 1), color='blue')
		ax1.set_xlabel('Q [$\AA^-$$^1$]', fontsize=16)
		ax1.set_ylabel('Intensity [a.u.]', fontsize=16)
		ax1.legend(loc='best', fontsize=14)
		ax1.tick_params(axis='x', labelsize = 16)
		ax1.tick_params(axis='y', labelsize = 16)
		ax1.set_title('i(Q)', fontsize=20)
		ax1.set_xlim(q_iq[0][0], q_iq[0][-1])

		ax2.plot(q_sq[0], sq[dict['pdf_file'] - 1], label='Frame ' + str(dict['pdf_file'] + 1), color='green')
		ax2.set_xlabel('Q [$\AA^-$$^1$]', fontsize=16)
		#ax2.set_ylabel('Intensity [a.u.]', fontsize=16)
		ax2.legend(loc='best', fontsize=14)
		ax2.tick_params(axis='x', labelsize = 16)
		ax2.tick_params(axis='y', labelsize = 16)
		ax2.set_title('s(Q)', fontsize=20)
		ax2.set_xlim(q_sq[0][0], q_sq[0][-1])

		ax3.plot(q_fq[0], fq[dict['pdf_file'] - 1], label='Frame ' + str(dict['pdf_file'] + 1), color='black')
		ax3.set_xlabel('Q [$\AA^-$$^1$]', fontsize=16)
		ax3.set_ylabel('Intensity [a.u.]', fontsize=16)
		ax3.legend(loc='best', fontsize=14)
		ax3.tick_params(axis='x', labelsize = 16)
		ax3.tick_params(axis='y', labelsize = 16)
		ax3.set_title('f(Q)', fontsize=20)
		ax3.set_xlim(q_fq[0][0], q_fq[0][-1])

		ax4.plot(r[0], gr[dict['pdf_file'] - 1], label='Frame ' + str(dict['pdf_file'] + 1), color='red')
		ax4.set_xlabel('r [$\AA$]', fontsize=16)
		#ax4.set_ylabel('Intensity [a.u.]', fontsize=16)
		ax4.legend(loc='best', fontsize=14)
		ax4.tick_params(axis='x', labelsize = 16)
		ax4.tick_params(axis='y', labelsize = 16)
		ax4.set_title('G(r)', fontsize=20)
		ax4.set_xlim(r[0][0], r[0][-1])

		plt.tight_layout(pad=2, w_pad=1.0, h_pad=1.0)		
		
		if dict['save_pics']:
			pic_dir(owd, 'Pictures_')
			fig2.savefig('All' + '.png')
			plt.clf()
		else:
			plt.draw()
			plt.pause(0.1)
			raw_input("<Hit Enter To Close>")
			plt.close()

if save_dict:
	pic_dir(owd, 'Pictures_')
	pickle.dump(dict, open(dict_name + ".py", "wb" ))