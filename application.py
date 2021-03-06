import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
#from flask_s3 import FlaskS3
import boto3
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import secrets
import tempfile

#where we will store the uploaded files
#UPLOAD_FOLDER = '/static/'

#set of allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'tif'}

application = Flask(__name__)
application.secret_key = 'some secret key'
#application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#application.config['FLASKS3_BUCKET_NAME'] = 'uploadfoldercivmfiles/civmuploads'
global temp_name
temp_name = tempfile.gettempdir()
#s3 = FlaskS3(application)
s3 = boto3.client('s3')
s3_resource = boto3.resource('s3')


#functions that check if an extension is valid and that uploads the file and redirects the user to the URL for the uploaded file
def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def headers_and_dropdown_func(txt_file_name):
	df = pd.read_csv(txt_file_name, sep='\t', encoding='cp850')
	column_headers = list(df.columns.values)
	column_headers.insert(0,'Select a variable to map')
	return df, column_headers

def centroid_bin_avg_func(dataframe, x_dim, y_dim, x_idx, y_idx, var_idx):
	df = dataframe
	column_headers = list(df.columns.values) 
	x_dim_round = round(x_dim/50)*50
	y_dim_round = round(y_dim/50)*50

	centroid_x_df = df.iloc[:,x_idx]
	centroid_y_df = df.iloc[:,y_idx]
	centroid_x = centroid_x_df.astype(float)
	centroid_y = centroid_y_df.astype(float)

	grid_x = int(x_dim_round/50)
	grid_y = int(y_dim_round/50)

	#grid cell coordinates for centroid of nucleus
	cell_x = np.ceil(centroid_x/50)
	cell_y = np.ceil(centroid_y/50)
	cell_x = cell_x.astype(int)
	cell_y = cell_y.astype(int)
	cell_x = cell_x.to_numpy()
	cell_y = cell_y.to_numpy()

	var_df = df.iloc[:,var_idx]
	var_vals = var_df.astype(float)
	
	nan_val_loc = np.where(np.isnan(var_vals) == True)
	no_nan_var_vals = var_vals[~np.isnan(var_vals)]
	no_nan_cell_x = np.delete(cell_x, nan_val_loc)
	no_nan_cell_y = np.delete(cell_y, nan_val_loc)

	#specific variable chosen is being averaged
	var_sum, xi, yi = np.histogram2d(no_nan_cell_x, no_nan_cell_y, bins=(grid_x, grid_y), range=[[1, grid_x], [1, grid_y]], weights = no_nan_var_vals, normed = False)
	#tracks number of values added into given bin
	count, _, _ = np.histogram2d(no_nan_cell_x, no_nan_cell_y, bins=(grid_x, grid_y), range=[[1, grid_x], [1, grid_y]])
	var_sd_object = stats.binned_statistic_2d(no_nan_cell_x, no_nan_cell_y, values = no_nan_var_vals, statistic = 'std', bins=(grid_x, grid_y), range=[[1, grid_x], [1, grid_y]])
	
	var_avg = var_sum/count
	var_avg = np.nan_to_num(var_avg)

	var_sd = var_sd_object.statistic
	var_sd = np.nan_to_num(var_sd)
	var_sd = var_sd**2
	var_sd = var_sd*count
	sample_count = count-1
	var_sd = var_sd/sample_count
	var_sd = var_sd**0.5
	var_sd = np.nan_to_num(var_sd)

	var_cov = var_sd/var_avg
	var_cov = np.nan_to_num(var_cov)

	mean_fname = secrets.token_urlsafe(16) + '.tif'
	mean_fname_data = secrets.token_urlsafe(16) + '.csv'
	im_mean = var_avg*255
	im_mean = np.flip(np.flip(im_mean),1)
	mean_image = Image.fromarray(np.uint8(im_mean), mode='L')
	mean_image.save(os.path.join(temp_name, mean_fname))
	np.savetxt(os.path.join(temp_name, mean_fname_data), var_avg, delimiter=',')

	cov_fname = secrets.token_urlsafe(16) + '.tif'
	cov_fname_data = secrets.token_urlsafe(16) + '.csv'
	min_cov_val = np.amin(var_cov)
	max_cov_val = np.amax(var_cov)
	im_cov = var_cov*255/max_cov_val
	im_cov = np.flip(np.flip(im_cov),1)
	cov_image = Image.fromarray(np.uint8(im_cov), mode='L')
	cov_image.save(os.path.join(temp_name, cov_fname))
	np.savetxt(os.path.join(temp_name, cov_fname_data), var_cov, delimiter=',')

	return mean_fname, cov_fname, mean_fname_data, cov_fname_data, column_headers

def verify_float(dataframe, all_possible_vars):
	verified_float_cols = [all_possible_vars[0]]
	verified_float_idx = [0]
	df = dataframe
	for i in range(1,len(all_possible_vars)):
		s1 = df.iloc[0,i-1]
		if isinstance(s1, np.float64) and not np.isnan(s1):
			verified_float_cols.append(all_possible_vars[i])
			verified_float_idx.append(i-1)
			if 'Centroid X' in all_possible_vars[i]:
				x_idx = i-1
			elif 'Centroid Y' in all_possible_vars[i]:
				y_idx = i-1
	return verified_float_cols, verified_float_idx, x_idx, y_idx

#global entered
#entered = 0
@application.route('/', methods=['GET', 'POST'])
def upload_file():
	global cols_out
	global df_out
	#global entered
	global x_dim
	global y_dim
	global float_cols
	global float_idx
	global centroid_x_idx
	global centroid_y_idx
	if request.method == 'GET':
		if request.args.get('map_vars'):
			var_sel = request.args.get('map_vars')
			df_idx = float_idx[float_cols.index(var_sel)]
			if df_idx == 0:
				flash('Select a valid variable to map')
				return redirect(request.url)
			mean_output, cov_output, mean_csv, cov_csv, map_vars = centroid_bin_avg_func(df_out, x_dim, y_dim, centroid_x_idx, centroid_y_idx, df_idx)	
			return(render_template('uploaded.html', mean_filename=mean_output, cov_filename=cov_output, mean_csv_filename = mean_csv, cov_csv_filename = cov_csv, map_vars=float_cols, selected_var=var_sel))
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename): 
			filename = secure_filename(file.filename)
			#s3.upload_file(filename, 'uploadfoldercivmfiles', 'testmati.txt')
			#needed because otherwise, file does not get added to static folder (may need to make into secure filename)
			#file.save(os.path.join(application.config['FLASKS3_BUCKET_NAME'], filename))
			print(filename)
			file.save(os.path.join(temp_name, filename))
			#s3.upload_file(filename, 'uploadfoldercivmfiles', filename)
			#my_bucket = s3_resource.Bucket('uploadfoldercivmfiles')
			#my_bucket.upload_file(filename, f'tmp/{filename}')
			#s3_resource.
			test_out = request.form
			text = request.form['text'] #17087.52,16753.22
			dims_strings = text.split(',') #must have comma, spaces don't matter
			x_dim = float(dims_strings[0].strip())
			y_dim = float(dims_strings[1].strip())			

			#df_out, cols_out = headers_and_dropdown_func(os.path.join(application.config['FLASKS3_BUCKET_NAME'], filename))

			#df_out, cols_out = headers_and_dropdown_func(s3_resource.Object('uploadfoldercivmfiles',filename).download_file(f'tmp/{filename}')
			df_out, cols_out = headers_and_dropdown_func(os.path.join(temp_name, filename))
			#os.remove(os.path.join(application.config['FLASKS3_BUCKET_NAME'], filename))
			#entered = 1
			float_cols, float_idx, centroid_x_idx, centroid_y_idx = verify_float(df_out, cols_out)

			return(render_template('uploaded.html', map_vars = float_cols))
	return render_template('uploaded.html')

@application.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = text.upper()
    return processed_text

@application.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(temp_name, filename)

if __name__ == "main":
	application.run(debug=True)