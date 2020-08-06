# SENN

This repository provides code implementation of the accepted paper on The International Conference on Data Science and Its Applications (ICoDSA) 2020, entitled "SENN: Stock Ensemble-based Neural Network for Stock Market Prediction using Historical Stock Data and Sentiment Analysis", written by [Louis Owen](http://louisowen6.github.io/) and [Finny Oktariani](https://www.itb.ac.id/staff/view/finny-oktariani-twd).

![SENN Architecture](https://github.com/louisowen6/SENN/blob/master/SENN_Architecture.png)


You can see the recorded presentation [here](https://drive.google.com/file/d/1gYJ519EEwMjU0ukpLYuTrGsgJIR31pN_/view?usp=sharing) and download the PPT [here](https://github.com/louisowen6/SENN/blob/master/SENN_PPT.pptx) 


## Abstract
`
Stock market prediction is one of the most appealing and challenging problems in the realm of data science. In this paper, authors investigate the potential of exploiting sentiment score extracted from microblog text data along with historical stock data to improve the stock market prediction performance. The sentiment score is extracted by using an ensemble-based model which utilize the power of Long Short-Term Memory (LSTM) and Multi-Layer Perceptron (MLP) along with Convolutional Neural Network (CNN). We propose Stock Ensemble-based Neural Network (SENN) model which is trained on the Boeing historical stock data and sentiment score extracted from StockTwits microblog text data in 2019. Furthermore, we also propose a novel way to measure the stock market prediction model performance which is the modification of the classic Mean Absolute Percentage Error (MAPE) metric, namely Adjusted MAPE (AMAPE). It has been observed from the experiments that utilizing SENN to integrate sentiment score as additional features could improve the stock market prediction performance up to 25% and also decreasing the margin of error up to 48%. With the training data limitation, the proposed model achieves a superior performance of 0.89% AMAPE. Our codes are available at https://www.github.com/louisowen6/SENN.
`

## Requirements

yfinance==0.1.54, tensorflow==1.15.2, standfordnlp, bs4, contractions, inflect, nltk, textblob, string, dtaidistance, stockstats, pandas, numpy, gensim, sklearn, datetime, matplotlib, seaborn


## Data Scraping

### Historical Stock Data

Simply use [data_scraping_yfinance.py](https://github.com/louisowen6/SENN/blob/master/data_scraping_yfinance.py) to gather the historical stock data.

```bash
usage: data_scraping_yfinance.py [-h] --ticker_name {AAPL,AXP,BA,...,XOM} --path PATH

optional arguments:
  -h, --help            show this help message and exit
  --ticker_name         {AAPL,AXP,BA,...,XOM}
                        Ticker name 
  --path                Path to SENN folder
```

### StockTwits Microblog Text Data

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


## Historical Stock Data Preparation

Please refer to [README](https://github.com/louisowen6/SENN/tree/master/Historical_Stock_Data) inside the /Historical_Stock_Data folder


## Sentiment Score Extraction

Please refer to [README](https://github.com/louisowen6/SENN/tree/master/Microblog_Text_Data) inside the /Microblog_Text_Data folder


## Stock Price Prediction

Please refer to [README](https://github.com/louisowen6/SENN/tree/master/Prediction) inside the /Prediction folder

## Citation

If you find the paper and the code helpful, please cite us.

```
@INPROCEEDINGS{Owen2008:SENN,
AUTHOR="Louis Owen and Finny Oktariani",
TITLE="{SENN:} Stock Ensemble-based Neural Network for Stock Market Prediction
using Historical Stock Data and Sentiment Analysis",
BOOKTITLE="2020 International Conference on Data Science and Its Applications (ICoDSA)
(ICoDSA 2020)",
DAYS=4,
MONTH=aug,
YEAR=2020
}
```

## License

The underlying code of this project is licensed under the [MIT license](https://github.com/louisowen6/SENN/blob/master/LICENSE).

