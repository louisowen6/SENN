import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler


def build_timeseries(mat, y_col_index,TIME_STEPS):
	# y_col_index is the index of column that would act as output column
	# total number of time-series samples would be len(mat) - TIME_STEPS
	dim_0 = mat.shape[0] - TIME_STEPS
	dim_1 = mat.shape[1]
	x = np.zeros((dim_0, TIME_STEPS, dim_1))
	y = np.zeros((dim_0,))

	for i in range(dim_0):
		x[i] = mat[i:TIME_STEPS+i]
		y[i] = mat[TIME_STEPS+i, y_col_index]
	return x, y


def build_sentiment_data(mat,TIME_STEPS,batch_size):
	dim_0 = mat.shape[0] - TIME_STEPS
	mat=mat[:dim_0]

	mat=trim_dataset(mat,batch_size)
	return mat


def trim_dataset(mat, batch_size):
	"""
	trims dataset to a size that's divisible by BATCH_SIZE
	"""
	no_of_rows_drop = mat.shape[0]%batch_size
	if(no_of_rows_drop > 0):
		return mat[:-no_of_rows_drop]
	else:
		return mat


def mape(y_true, y_pred): 
	'''
	Function to calculate MAPE
	'''
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def adjusted_mape(y_true, y_pred): 
	'''
	Function to calculate Adjusted MAPE
	'''
	y_true_after = y_true[1:]
	y_true_before = y_true[:-1]
	y_pred_after = y_pred[1:]
	y_pred_before = y_pred[:-1]

	y_true_label = (pd.Series(y_true_after)-pd.Series(y_true_before)).apply(lambda x: 1 if x>0 else 0)
	y_pred_label = (pd.Series(y_pred_after)-pd.Series(y_pred_before)).apply(lambda x: 1 if x>0 else 0)

	adjustment_constant = np.abs(y_true_label-y_pred_label).apply(lambda x: 2 if x==1 else 1).tolist()
	adjustment_constant.insert(0,1)

	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return np.mean(np.abs((y_true - y_pred) / (2*y_true)) * adjustment_constant) * 100