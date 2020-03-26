import argparse
import pandas as pd
import yfinance as yf

parser = argparse.ArgumentParser()
parser.add_argument('--ticker_name',
	required=True,
	help='Ticker name')
parser.add_argument('--path',
	required=True,
	help="Path to SENN folder")

def yfinance_scrap(start,end,ticker):
	return yf.download(ticker, start=start, end=end,interval = "1h")


def main():
	args = parser.parse_args()
	ticker = args.ticker_name
	PATH = args.path

	df_yfinance=yfinance_scrap(start='2019-01-01',end='2020-01-01',ticker=ticker)
	df_yfinance=df_yfinance[df_yfinance.index!='2018-12-31']
	index_dict_base=df_yfinance.index.astype(str).value_counts().to_dict()
	index_dict=df_yfinance.index.astype(str).value_counts().to_dict()

	def time_count(x):
		if index_dict_base[x]-index_dict[x]==0:
			index_dict[x]-=1
			return('09:30')
		elif index_dict_base[x]-index_dict[x]==1:
			index_dict[x]-=1
			return('10:30')
		elif index_dict_base[x]-index_dict[x]==2:
			index_dict[x]-=1
			return('11:30')
		elif index_dict_base[x]-index_dict[x]==3:
			index_dict[x]-=1
			return('12:30')
		elif index_dict_base[x]-index_dict[x]==4:
			index_dict[x]-=1
			return('13:30')
		elif index_dict_base[x]-index_dict[x]==5:
			index_dict[x]-=1
			return('14:30')
		elif index_dict_base[x]-index_dict[x]==6:
			index_dict[x]-=1
			return('15:30')

	index_iter=pd.Series(df_yfinance.index.astype(str))
	df_yfinance['time']=index_iter.apply(lambda x: time_count(x)).tolist()

	df_yfinance.to_csv(PATH+'/SENN/Dataset/df_yfinance_full_'+ticker+'.csv')


if __name__ == '__main__':
	main()