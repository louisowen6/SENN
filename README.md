```bash
                       Code implementation of SENN: Stock Ensemble-based Neural Network
```

# Abstract
`
Stock market prediction is one of the most appealing and challenging problems in the realm of data science. In this paper, authors investigate the potential of exploiting sentiment score extracted from microblog text data along with historical stock data to improve the stock market prediction performance. The sentiment score is extracted by using an ensemble-based model which utilize the power of Long Short-Term Memory (LSTM) and Multi-Layer Perceptron (MLP) along with Convolutional Neural Network (CNN). We propose a robust Stock Ensemble-based Neural Network (SENN) model which is trained on the Boeing historical stock data and sentiment score extracted from StockTwits microblog text data in 2019. Furthermore, we also propose a novel way to measure the stock market prediction model performance which extent the classic Mean Absolute Percentage Error (MAPE) metric, namely Adjusted MAPE (AMAPE). It has been observed from the experiments that utilizing SENN to integrate sentiment score as additional features could improve the stock market prediction performance up to 25% and also decreasing the margin of error up to 48%. With the training data limitation, the proposed model achieves a superior performance of 0.89% AMAPE. Our codes are available at https://www.github.com/louisowen6/SENN.
`

# Requirements

yfinance, tensorflow, standfordnlp, bs4, contractions, inflect, nltk, textblob, string, dtaidistance, stockstats, pandas, numpy, gensim, sklearn, datetime, matplotlib, seaborn


# Data Scraping

## Historical Stock Data

Simply use [data_scraping_yfinance.py](https://github.com/louisowen6/SENN/blob/master/data_scraping_yfinance.py) to gather the historical stock data.

```bash
usage: data_scraping_yfinance.py [-h] --ticker_name {AAPL,AXP,BA,...,XOM} --path PATH

optional arguments:
  -h, --help            show this help message and exit
  --ticker_name         {AAPL,AXP,BA,...,XOM}
                        Ticker name 
  --path                Path to SENN folder
```

## StockTwits Microblog Text Data

In order to gather StockTwits data, one needs to have the [StockTwits API account](https://api.stocktwits.com/developers/docs). Then, use [data_scraping_stocktwits.py](https://github.com/louisowen6/SENN/blob/master/data_scraping_stocktwits.py) to scrape the data.

```bash
usage: data_scraping_stocktwits.py [-h] --concat_df {Y,N} --scrape_iter INTEGER --path PATH

optional arguments:
  -h, --help            show this help message and exit
  --concat_df           {Y,N}
                        Use this script to concat all of the extracted datasets or to extract StockTwits data part by part
  --scrape_iter         integer; to specified the index of extracted data since there is the API limitation regarding the amount of data
  --path                Path to SENN folder
```


# Historical Stock Data Preparation

Please refer to [README](https://github.com/louisowen6/SENN/tree/master/Historical_Stock_Data) inside /Historical_Stock_Data folder


# Sentiment Score Extraction

Please refer to [README](https://github.com/louisowen6/SENN/tree/master/Microblog_Text_Data) inside /Microblog_Text_Data folder
