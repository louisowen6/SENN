NRC Hashtag Sentiment Lexicon
Version 1.0
9 April 2013
Copyright (C) 2013 National Research Council Canada (NRC)
************************************************************


************************************************************
Contact: 
************************************************************


Technical enquiries

Saif M. Mohammad (Senior Research Officer at NRC and creator of these lexicons)Saif.Mohammad@nrc-cnrc.gc.ca 

Business enquiries

Pierre Charron (Client Relationship Leader at NRC)
Pierre.Charron@nrc-cnrc.gc.ca



Information on various lexicons is available here:
http://saifmohammad.com/WebPages/lexicons.html

You may also be interested in some of the other resources and work we have done on the analysis of emotions in text:
http://saifmohammad.com/WebPages/ResearchAreas.html
http://saifmohammad.com/WebPages/ResearchInterests.html#EmotionAnalysis



************************************************************
Terms of Use: 
************************************************************

1. The lexicons mentioned in this page can be used freely for non-commercial research and educational purposes.

2. Cite the papers associated with the lexicons in your research papers and articles that make use of them. (The papers associated with each lexicon are listed below, and also in the READMEs for individual lexicons.) 

3. In news articles and online posts on work using these lexicons, cite the appropriate lexicons. For example:
"This application/product/tool makes use of the <resource name>, created by <author(s)> at the National Research Council Canada." (The creators of each lexicon are listed below. Also, if you send us an email, we will be thrilled to know about how you have used the lexicon.) If possible hyperlink to this page: http://saifmohammad.com/WebPages/lexicons.html

4. If you use a lexicon in a product or application, then acknowledge this in the 'About' page and other relevant documentation of the application by stating the name of the resource, the authors, and NRC. For example:
"This application/product/tool makes use of the <resource name>, created by <author(s)> at the National Research Council Canada." (The creators of each lexicon are listed below. Also, if you send us an email, we will be thrilled to know about how you have used the lexicon.) If possible hyperlink to this page: http://saifmohammad.com/WebPages/lexicons.html

5. Do not redistribute the data. Direct interested parties to this page: http://saifmohammad.com/WebPages/AccessResource.htm

6. If interested in commercial use of any of these lexicons, see information here: https://shop-magasin.nrc-cnrc.gc.ca/nrcb2c/app/displayApp/(cpgnum=1&layout=7.01-7_1_71_63_73_6_9_3&uiarea=3&carea=0000000104&cpgsize=0)/.do?rf=y.

7. National Research Council Canada (NRC) disclaims any responsibility for the use of the lexicons listed here and does not provide technical support. However, the contact listed above will be happy to respond to queries and clarifications.



We will be happy to hear from you, especially if:
- you give us feedback regarding these lexicons;
- you tell us how you have (or plan to) use the lexicons;
- you are interested in having us analyze your data for sentiment, emotion, and other affectual information;
- you are interested in a collaborative research project. We also regularly hire graduate students for research internships.




Creators: Svetlana Kiritchenko and Saif M. Mohammad


Papers associated with this lexicon:

Kiritchenko, S., Zhu, X., Mohammad, S. (2014). Sentiment Analysis of Short Informal Texts. Journal of Artificial Intelligence Research, 50:723-762, 2014.

Mohammad, S., Kiritchenko, S., Zhu, X. (2013). NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets. In Proceedings of the seventh International Workshop on Semantic Evaluation Exercises (SemEval-2013), June 2013, Atlanta, USA. 

Zhu, X., Kiritchenko, S., Mohammad, S. (2014). NRC-Canada-2014: Recent Improvements in Sentiment Analysis of Tweets. In Proceedings of the eigth international workshop on Semantic Evaluation Exercises (SemEval-2014), August 2014, Dublin, Ireland.    







************************************************************
DATA SOURCE
************************************************************

The NRC Hashtag Sentiment Lexicon is automatically generated from the following data source: 
775,000 tweets with sentiment-word hashtags collected by NRC.


************************************************************
FILE FORMAT
************************************************************

Each line in the lexicons has the following format:
<term><tab><score><tab><Npos><tab><Nneg>

<term> can be a unigram, a bigram, or a pair of unigrams and bigrams (unigram--unigram, unigram--bigram, bigram--unigram, or bigram--bigram);
<score> is a real-valued sentiment score: score = PMI(w, pos) - PMI(w, neg), where PMI stands for Point-wise Mutual Information between a term w and the positive/negative class;
<Npos> is the number of times the term appears in the positive class, i.e., in tweets with positive hashtags;
<Nneg> is the number of times the term appears in the negative class, i.e., in tweets with negative hashtags.


************************************************************
Sentiment-Word Hashtags
************************************************************

This distribution includes the list of positive and negative hashtags that were used to collect the tweets. The list can be found in file sentiment-hashtags.txt.


************************************************************
More Information
************************************************************

Details on the process of creating the lexicon can be found in the associated papers (see above).

