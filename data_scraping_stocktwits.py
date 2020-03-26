import argparse
import os
import requests
import json
import datetime
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--scrape_iter',
	required=False,
	help='Scrape Iteration')
parser.add_argument('--concat_df',
	required=True,
	choices=['Y', 'N'],
	help='Use this script to concat all of the extracted datasets or to extract StockTwits data part by part')
parser.add_argument('--path',
	required=True,
	help="Path to SENN folder")


def stocktwits_scrap(n,ticker,base=149407261):
	'''
	Function to scrap stocktwits tweet

	Base: Base ID to scrape from
	'''

	id=[]
	created_at=[]
	body=[]
	for i in range(n):
		try:
			url = "https://api.stocktwits.com/api/2/streams/symbol/"+ticker+".json?max="+str(base+5000)+"&since="+str(base)+"&limit=30"

			base=base+5001

			response = requests.request("GET", url, headers={}, data ={})
			response=response.json()
			messages=response['messages']
			iter=range(len(messages))
			for idx in iter:
				id.append(messages[idx]['id'])
				created_at.append(messages[idx]['created_at'])
				body.append(messages[idx]['body'])

				if (len(id) % 100) == 0:
					print('Done Scrape ',len(id),' messages.')
		except:
			print('Warning Messages: \n ID Out of Range or Too Many Requests within 1 Hour')
			break
	try:
		df_stocktwits=pd.DataFrame(id,columns=['id'])
		df_stocktwits['created_at']=created_at
		df_stocktwits['created_at']=pd.to_datetime(df_stocktwits['created_at'])
		df_stocktwits['body']=body
		df_stocktwits=df_stocktwits.sort_values(by='created_at')
		df_stocktwits=df_stocktwits.reset_index(drop=True)
		print('Done Scrape ',len(id),' messages.')
		print('Last ID: ',df_stocktwits.tail(1)['id'].values[0])
		return(df_stocktwits,df_stocktwits.tail(1)['id'].values[0])
	except:
		print('Error')
		return None,last_id


def main():
	args = parser.parse_args()
	is_concat = args.concat_df
	PATH = args.path

	if is_concat:
		source='C:/Users/Louis Owen/Desktop/ICoDSA 2020/SENN/Dataset/stocktwits BA scraping/'
		_, _, filenames = next(os.walk(source))
		filenames=pd.Series(filenames).sort_values().to_list()
		df_stocktwits=pd.read_csv(source+filenames[0])
		filenames.pop(0)
		for filename in filenames:
			toy=pd.read_csv(source+filename)
			df_stocktwits=pd.concat([df_stocktwits,toy])

		df_stocktwits.to_csv(PATH+'/Dataset/df_stocktwits_full_BA.csv',index=False)
	else:
		scrape_iter = args.scrape_iter

		if scrape_iter==1:
			base=149407261
		else:
			base=last_id

		df_stocktwits,last_id=stocktwits_scrap(n=1000,ticker='BA',base=base)

		df_stocktwits.to_csv(PATH+'/Dataset/stocktwits BA scraping/df_stocktwits_'+str(scrape_iter)+'.csv',index=False)


if __name__ == '__main__':
	main()