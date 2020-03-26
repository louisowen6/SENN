
PATH_ROOT='C:/Users/Louis Owen/Desktop/ICoDSA 2020/SENN/Dataset/'

print('==================== Importing Packages ====================')

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import stockstats

#------------------------------------------------------------------------------------------
print("==================== Importing Data ====================")

df_yfinance_BA = pd.read_csv(PATH_ROOT+'df_yfinance_full_BA.csv')
df_yfinance_BA.index=pd.to_datetime(df_yfinance_BA['Date'].reset_index(drop=True)+' '+df_yfinance_BA['time'].reset_index(drop=True))
df_yfinance_BA=df_yfinance_BA.drop(['Date','time'],1)

#------------------------------------------------------------------------------------------

def main():
	#Create Target Variable
	window=7
	BA_close=df_yfinance_BA['Close'].tolist()
	BA_close_after=BA_close[window:]
	BA_close_before=BA_close[:-window]
	diff=pd.Series(BA_close_after)-pd.Series(BA_close_before)
	diff=diff.apply(lambda x: 1 if x>0 else 0)
	diff=diff.tolist()
	for i in range(window):
		diff.append(np.nan)
	df_yfinance_BA['bullish']=diff

	#Create Volume Price Difference
	volume=df_yfinance_BA['Volume'].tolist()
	volume_before_1=volume.copy()
	volume_before_1.pop()
	volume_before_2=volume_before_1.copy()
	volume_before_2.pop()
	volume_diff_last_hour=(pd.Series(volume_before_1[1:])-pd.Series(volume_before_2)).tolist()
	volume_diff_last_hour.insert(0,np.nan)
	volume_diff_last_hour.insert(0,np.nan)
	volume_before_1.insert(0,np.nan)
	df_yfinance_BA['volume_before_1']=volume_before_1
	df_yfinance_BA['volume_diff_last_hour']=volume_diff_last_hour

	volume_before_3=volume_before_2.copy()
	volume_before_3.pop()
	volume_diff_last_2_hour=(pd.Series(volume_before_2[1:])-pd.Series(volume_before_3)).tolist()
	volume_diff_last_2_hour.insert(0,np.nan)
	volume_diff_last_2_hour.insert(0,np.nan)
	volume_diff_last_2_hour.insert(0,np.nan)
	volume_before_2.insert(0,np.nan)
	volume_before_2.insert(0,np.nan)
	df_yfinance_BA['volume_before_2']=volume_before_2
	df_yfinance_BA['volume_diff_last_2_hour']=volume_diff_last_2_hour

	volume_before_4=volume_before_3.copy()
	volume_before_4.pop()
	volume_diff_last_3_hour=(pd.Series(volume_before_3[1:])-pd.Series(volume_before_4)).tolist()
	volume_diff_last_3_hour.insert(0,np.nan)
	volume_diff_last_3_hour.insert(0,np.nan)
	volume_diff_last_3_hour.insert(0,np.nan)
	volume_diff_last_3_hour.insert(0,np.nan)
	volume_before_3.insert(0,np.nan)
	volume_before_3.insert(0,np.nan)
	volume_before_3.insert(0,np.nan)
	df_yfinance_BA['volume_before_3']=volume_before_3
	df_yfinance_BA['volume_diff_last_3_hour']=volume_diff_last_3_hour

	volume_before_5=volume_before_4.copy()
	volume_before_5.pop()
	volume_diff_last_4_hour=(pd.Series(volume_before_4[1:])-pd.Series(volume_before_5)).tolist()
	volume_diff_last_4_hour.insert(0,np.nan)
	volume_diff_last_4_hour.insert(0,np.nan)
	volume_diff_last_4_hour.insert(0,np.nan)
	volume_diff_last_4_hour.insert(0,np.nan)
	volume_diff_last_4_hour.insert(0,np.nan)
	volume_before_4.insert(0,np.nan)
	volume_before_4.insert(0,np.nan)
	volume_before_4.insert(0,np.nan)
	volume_before_4.insert(0,np.nan)
	df_yfinance_BA['volume_before_4']=volume_before_4
	df_yfinance_BA['volume_diff_last_4_hour']=volume_diff_last_4_hour

	volume_before_6=volume_before_5.copy()
	volume_before_6.pop()
	volume_diff_last_5_hour=(pd.Series(volume_before_5[1:])-pd.Series(volume_before_6)).tolist()
	volume_diff_last_5_hour.insert(0,np.nan)
	volume_diff_last_5_hour.insert(0,np.nan)
	volume_diff_last_5_hour.insert(0,np.nan)
	volume_diff_last_5_hour.insert(0,np.nan)
	volume_diff_last_5_hour.insert(0,np.nan)
	volume_diff_last_5_hour.insert(0,np.nan)
	volume_before_5.insert(0,np.nan)
	volume_before_5.insert(0,np.nan)
	volume_before_5.insert(0,np.nan)
	volume_before_5.insert(0,np.nan)
	volume_before_5.insert(0,np.nan)
	df_yfinance_BA['volume_before_5']=volume_before_5
	df_yfinance_BA['volume_diff_last_5_hour']=volume_diff_last_5_hour

	#Create Close Price Difference
	close=df_yfinance_BA['Close'].tolist()
	close_before_1=close.copy()
	close_before_1.pop()
	close_before_2=close_before_1.copy()
	close_before_2.pop()
	close_diff_last_hour=(pd.Series(close_before_1[1:])-pd.Series(close_before_2)).tolist()
	close_diff_last_hour.insert(0,np.nan)
	close_diff_last_hour.insert(0,np.nan)
	close_before_1.insert(0,np.nan)
	df_yfinance_BA['close_before_1']=close_before_1
	df_yfinance_BA['close_diff_last_hour']=close_diff_last_hour

	close_before_3=close_before_2.copy()
	close_before_3.pop()
	close_diff_last_2_hour=(pd.Series(close_before_2[1:])-pd.Series(close_before_3)).tolist()
	close_diff_last_2_hour.insert(0,np.nan)
	close_diff_last_2_hour.insert(0,np.nan)
	close_diff_last_2_hour.insert(0,np.nan)
	close_before_2.insert(0,np.nan)
	close_before_2.insert(0,np.nan)
	df_yfinance_BA['close_before_2']=close_before_2
	df_yfinance_BA['close_diff_last_2_hour']=close_diff_last_2_hour

	close_before_4=close_before_3.copy()
	close_before_4.pop()
	close_diff_last_3_hour=(pd.Series(close_before_3[1:])-pd.Series(close_before_4)).tolist()
	close_diff_last_3_hour.insert(0,np.nan)
	close_diff_last_3_hour.insert(0,np.nan)
	close_diff_last_3_hour.insert(0,np.nan)
	close_diff_last_3_hour.insert(0,np.nan)
	close_before_3.insert(0,np.nan)
	close_before_3.insert(0,np.nan)
	close_before_3.insert(0,np.nan)
	df_yfinance_BA['close_before_3']=close_before_3
	df_yfinance_BA['close_diff_last_3_hour']=close_diff_last_3_hour

	close_before_5=close_before_4.copy()
	close_before_5.pop()
	close_diff_last_4_hour=(pd.Series(close_before_4[1:])-pd.Series(close_before_5)).tolist()
	close_diff_last_4_hour.insert(0,np.nan)
	close_diff_last_4_hour.insert(0,np.nan)
	close_diff_last_4_hour.insert(0,np.nan)
	close_diff_last_4_hour.insert(0,np.nan)
	close_diff_last_4_hour.insert(0,np.nan)
	close_before_4.insert(0,np.nan)
	close_before_4.insert(0,np.nan)
	close_before_4.insert(0,np.nan)
	close_before_4.insert(0,np.nan)
	df_yfinance_BA['close_before_4']=close_before_4
	df_yfinance_BA['close_diff_last_4_hour']=close_diff_last_4_hour

	close_before_6=close_before_5.copy()
	close_before_6.pop()
	close_diff_last_5_hour=(pd.Series(close_before_5[1:])-pd.Series(close_before_6)).tolist()
	close_diff_last_5_hour.insert(0,np.nan)
	close_diff_last_5_hour.insert(0,np.nan)
	close_diff_last_5_hour.insert(0,np.nan)
	close_diff_last_5_hour.insert(0,np.nan)
	close_diff_last_5_hour.insert(0,np.nan)
	close_diff_last_5_hour.insert(0,np.nan)
	close_before_5.insert(0,np.nan)
	close_before_5.insert(0,np.nan)
	close_before_5.insert(0,np.nan)
	close_before_5.insert(0,np.nan)
	close_before_5.insert(0,np.nan)
	df_yfinance_BA['close_before_5']=close_before_5
	df_yfinance_BA['close_diff_last_5_hour']=close_diff_last_5_hour

	close_before_7=close_before_6.copy()
	close_before_7.pop()
	close_diff_last_6_hour=(pd.Series(close_before_6[1:])-pd.Series(close_before_7)).tolist()
	close_diff_last_6_hour.insert(0,np.nan)
	close_diff_last_6_hour.insert(0,np.nan)
	close_diff_last_6_hour.insert(0,np.nan)
	close_diff_last_6_hour.insert(0,np.nan)
	close_diff_last_6_hour.insert(0,np.nan)
	close_diff_last_6_hour.insert(0,np.nan)
	close_diff_last_6_hour.insert(0,np.nan)
	close_before_6.insert(0,np.nan)
	close_before_6.insert(0,np.nan)
	close_before_6.insert(0,np.nan)
	close_before_6.insert(0,np.nan)
	close_before_6.insert(0,np.nan)
	close_before_6.insert(0,np.nan)
	df_yfinance_BA['close_before_6']=close_before_6
	df_yfinance_BA['close_diff_last_6_hour']=close_diff_last_6_hour

	close_before_8=close_before_7.copy()
	close_before_8.pop()
	close_diff_last_7_hour=(pd.Series(close_before_7[1:])-pd.Series(close_before_8)).tolist()
	close_diff_last_7_hour.insert(0,np.nan)
	close_diff_last_7_hour.insert(0,np.nan)
	close_diff_last_7_hour.insert(0,np.nan)
	close_diff_last_7_hour.insert(0,np.nan)
	close_diff_last_7_hour.insert(0,np.nan)
	close_diff_last_7_hour.insert(0,np.nan)
	close_diff_last_7_hour.insert(0,np.nan)
	close_diff_last_7_hour.insert(0,np.nan)
	close_before_7.insert(0,np.nan)
	close_before_7.insert(0,np.nan)
	close_before_7.insert(0,np.nan)
	close_before_7.insert(0,np.nan)
	close_before_7.insert(0,np.nan)
	close_before_7.insert(0,np.nan)
	close_before_7.insert(0,np.nan)
	df_yfinance_BA['close_before_7']=close_before_7
	df_yfinance_BA['close_diff_last_7_hour']=close_diff_last_7_hour

	close_before_9=close_before_8.copy()
	close_before_9.pop()
	close_diff_last_8_hour=(pd.Series(close_before_8[1:])-pd.Series(close_before_9)).tolist()
	close_diff_last_8_hour.insert(0,np.nan)
	close_diff_last_8_hour.insert(0,np.nan)
	close_diff_last_8_hour.insert(0,np.nan)
	close_diff_last_8_hour.insert(0,np.nan)
	close_diff_last_8_hour.insert(0,np.nan)
	close_diff_last_8_hour.insert(0,np.nan)
	close_diff_last_8_hour.insert(0,np.nan)
	close_diff_last_8_hour.insert(0,np.nan)
	close_diff_last_8_hour.insert(0,np.nan)
	close_before_8.insert(0,np.nan)
	close_before_8.insert(0,np.nan)
	close_before_8.insert(0,np.nan)
	close_before_8.insert(0,np.nan)
	close_before_8.insert(0,np.nan)
	close_before_8.insert(0,np.nan)
	close_before_8.insert(0,np.nan)
	close_before_8.insert(0,np.nan)
	df_yfinance_BA['close_before_8']=close_before_8
	df_yfinance_BA['close_diff_last_8_hour']=close_diff_last_8_hour

	close_before_10=close_before_9.copy()
	close_before_10.pop()
	close_diff_last_9_hour=(pd.Series(close_before_9[1:])-pd.Series(close_before_10)).tolist()
	close_diff_last_9_hour.insert(0,np.nan)
	close_diff_last_9_hour.insert(0,np.nan)
	close_diff_last_9_hour.insert(0,np.nan)
	close_diff_last_9_hour.insert(0,np.nan)
	close_diff_last_9_hour.insert(0,np.nan)
	close_diff_last_9_hour.insert(0,np.nan)
	close_diff_last_9_hour.insert(0,np.nan)
	close_diff_last_9_hour.insert(0,np.nan)
	close_diff_last_9_hour.insert(0,np.nan)
	close_diff_last_9_hour.insert(0,np.nan)
	close_before_9.insert(0,np.nan)
	close_before_9.insert(0,np.nan)
	close_before_9.insert(0,np.nan)
	close_before_9.insert(0,np.nan)
	close_before_9.insert(0,np.nan)
	close_before_9.insert(0,np.nan)
	close_before_9.insert(0,np.nan)
	close_before_9.insert(0,np.nan)
	close_before_9.insert(0,np.nan)
	df_yfinance_BA['close_before_9']=close_before_9
	df_yfinance_BA['close_diff_last_9_hour']=close_diff_last_9_hour

	#Create Moving Average
	stock=stockstats.StockDataFrame.retype(df_yfinance_BA[['Open','Close','High','Low','Volume']])

	df_yfinance_BA['SMA_15']=stock['close_15_sma'] #Untuk data ke 1 sampai 9, SMA dihitung berdasarkan data yang ada
	df_yfinance_BA['SMA_30']=stock['close_30_sma'] #Untuk data ke 1 sampai 29, SMA dihitung berdasarkan data yang ada

	df_yfinance_BA['SMA_indicator']=df_yfinance_BA.apply(lambda x: np.nan if pd.isnull(x.SMA_30) else 1 if x.SMA_15>x.SMA_30 else 0,axis=1)
	SMA_indicator=df_yfinance_BA['SMA_indicator'].tolist()
	SMA_indicator_before_1=SMA_indicator.copy()
	SMA_indicator_before_1.pop()
	SMA_indicator_before_1.insert(0,np.nan)
	df_yfinance_BA['SMA_indicator_before_1']=SMA_indicator_before_1
	SMA_indicator_before_2=SMA_indicator_before_1.copy()
	SMA_indicator_before_2.pop()
	SMA_indicator_before_2.insert(0,np.nan)
	df_yfinance_BA['SMA_indicator_before_2']=SMA_indicator_before_2
	SMA_indicator_before_3=SMA_indicator_before_2.copy()
	SMA_indicator_before_3.pop()
	SMA_indicator_before_3.insert(0,np.nan)
	df_yfinance_BA['SMA_indicator_before_3']=SMA_indicator_before_3
	SMA_indicator_before_4=SMA_indicator_before_3.copy()
	SMA_indicator_before_4.pop()
	SMA_indicator_before_4.insert(0,np.nan)
	df_yfinance_BA['SMA_indicator_before_4']=SMA_indicator_before_4

	#Create Bollinger Bands
	df_yfinance_BA['Upper_Bollinger']=stock['boll_ub']
	df_yfinance_BA['Lower_Bollinger']=stock['boll_lb']
	df_yfinance_BA['Middle_Bollinger']=stock['boll']

	df_yfinance_BA['close_diff_Upper_Bollinger']=df_yfinance_BA['Close']-df_yfinance_BA['Upper_Bollinger']
	df_yfinance_BA['close_diff_Lower_Bollinger']=df_yfinance_BA['Close']-df_yfinance_BA['Lower_Bollinger']
	df_yfinance_BA['Bollinger_indicator']=df_yfinance_BA.apply(lambda x: np.nan if pd.isnull(x.Upper_Bollinger) else 1 if x.Close>x.Upper_Bollinger else 2 if (x.Close>x.Middle_Bollinger) & (x.Close<=x.Upper_Bollinger) else
	                                                           3 if (x.Close>=x.Lower_Bollinger) & (x.Close<=x.Middle_Bollinger) else 4,axis=1)

	Bollinger_indicator=df_yfinance_BA['Bollinger_indicator'].tolist()
	Bollinger_indicator_before_1=Bollinger_indicator.copy()
	Bollinger_indicator_before_1.pop()
	Bollinger_indicator_before_1.insert(0,np.nan)
	df_yfinance_BA['Bollinger_indicator_before_1']=Bollinger_indicator_before_1
	Bollinger_indicator_before_2=Bollinger_indicator_before_1.copy()
	Bollinger_indicator_before_2.pop()
	Bollinger_indicator_before_2.insert(0,np.nan)
	df_yfinance_BA['Bollinger_indicator_before_2']=Bollinger_indicator_before_2
	Bollinger_indicator_before_3=Bollinger_indicator_before_2.copy()
	Bollinger_indicator_before_3.pop()
	Bollinger_indicator_before_3.insert(0,np.nan)
	df_yfinance_BA['Bollinger_indicator_before_3']=Bollinger_indicator_before_3
	Bollinger_indicator_before_4=Bollinger_indicator_before_3.copy()
	Bollinger_indicator_before_4.pop()
	Bollinger_indicator_before_4.insert(0,np.nan)
	df_yfinance_BA['Bollinger_indicator_before_4']=Bollinger_indicator_before_4

	#Create True Range
	df_yfinance_BA['true_range']=stock['tr']

	#Create holiday_day_diff_before
	df_yfinance_BA['holiday_day_diff_before']=(pd.Series(df_yfinance_BA.index).apply(lambda x: holiday_day_diff_before(x))).tolist()

	#Create Time Variable
	df_yfinance_BA['date']=df_yfinance_BA.index.day.tolist()
	df_yfinance_BA['weekday']=df_yfinance_BA.index.weekday.tolist()
	df_yfinance_BA['time']=(pd.Series(df_yfinance_BA.index.time)).tolist()
	df_yfinance_BA['time']=df_yfinance_BA['time'].astype(str)
	df_yfinance_BA['time']=df_yfinance_BA['time'].apply(lambda x: 1 if x=='09:30:00' else 2 if x=='10:30:00' else 3 if x=='11:30:00' else 4 if x=='12:30:00' else 5 if x=='13:30:00' else 6 if x=='14:30:00' else 7)

	df_yfinance_BA.to_csv(PATH_ROOT+'Final/df_yfinance_BA_prepared.csv')


if __name__ == '__main__':
	main()