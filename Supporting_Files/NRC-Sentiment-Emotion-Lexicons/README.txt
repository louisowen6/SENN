NRC Sentiment and Emotion Lexicons 
April 2016
Copyright (C) 2016 National Research Council Canada (NRC)
----------------------------------------------------------------


Contact: 
-------------------------------------------------


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



Terms of Use: 
-------------------------------------------------

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



NRC Sentiment and Emotion Lexicons 
-----------------------------

The Sentiment and Emotion Lexicons is a collection of lexicons that was entirely created by the experts of the National Research Council of Canada. Developed with a wide range of applications, this lexicon collection can be used in a multitude of contexts such as sentiment analysis, product marketing, consumer behaviour and even political campaign analysis. 

The technology uses a list of words that help identify emotions, sentiment, as well as analyzing hashtags, emoticons and word-colour associations. The lexicons contain entries for English words, and can be used to analyze English texts. Also provided are translations of the entries in the NRC Emotion Lexicon in 105 other languages, including French, Arabic, Chinese, and Spanish.

Further details about each lexicon are available in the READMEs (provided within the folders associated with the lexicons) and in the associated papers listed in the READMEs. 


The Sentiment and Emotion Lexicons included in this distribution:
----------------------------------------------------------------

1. NRC Emotion Lexicon: association of words with eight emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive) manually annotated on Amazon's Mechanical Turk. Available in 105 different languages.
	Version: 0.92
	Number of terms: 14,182 unigrams (words), ~25,000 word senses
	Association scores: binary (associated or not)
	Creators: Saif M. Mohammad and Peter D. Turney

	Papers for (1):

	Crowdsourcing a Word-Emotion Association Lexicon, Saif Mohammad and Peter Turney, Computational Intelligence, 29 (3), 436-465, 2013.

	Emotions Evoked by Common Words and Phrases: Using Mechanical Turk to Create an Emotion Lexicon, Saif Mohammad and Peter Turney, In Proceedings of the NAACL-HLT 2010 Workshop on Computational Approaches to Analysis and Generation of Emotion in Text, June 2010, LA, California.


2. NRC Valence, Arousal, and Dominance (VAD) Lexicon: a list of English words and their valence, arousal, and dominance scores. The lexicon with its fine-grained real-valued scores was created by manual annotation using Best--Worst Scaling. Available in 104 different languages.
	Version: 1.0
	Number of terms: 20,007 unigrams (words)
	Association scores: real-valued
	Creator: Saif M. Mohammad

	Papers for (2):

	Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words. Saif M. Mohammad. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, Melbourne, Australia, July 2018.


3. NRC Affect Intensity Lexicon: association of words with four basic emotions (anger, fear, sadness, joy). The lexicon with its fine-grained real-valued scores was created by manual annotation using Best--Worst Scaling. 
	Version: 0.5
	Number of terms: 6,000 unigrams (words)
	Association scores: real-valued
	Creator: Saif M. Mohammad

	Papers for (3):

	Word Affect Intensities. Saif M. Mohammad. In Proceedings of the 11th edition of the Language Resources and Evaluation Conference, May 2018, Miyazaki, Japan.


4. NRC Hashtag Emotion Lexicon: association of words with eight emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) generated automatically from tweets with emotion-word hashtags such as #happy and #anger. 
	Version: 0.2
	Number of terms: 16,862 unigrams (words)
	Association scores: real-valued
	Creator: Saif M. Mohammad

	Papers for (4):

	Using Hashtags to Capture Fine Emotion Categories from Tweets. Saif M. Mohammad, Svetlana Kiritchenko, Computational Intelligence, 31(2): 301-326, 2015.

	#Emotional Tweets, Saif Mohammad, In Proceedings of the First Joint Conference on Lexical and Computational Semantics (*Sem), June 2012, Montreal, Canada.


5. NRC Hashtag Sentiment Lexicon: association of words with positive (negative) sentiment generated automatically from tweets with sentiment-word hashtags such as #amazing and #terrible. 
	Version: 1.0
	Number of terms: 54,129 unigrams, 316,531 bigrams, 308,808 pairs
	Association scores: real-valued
	Creators: Saif M. Mohammad and Svetlana Kiritchenko


6. NRC Hashtag Affirmative Context Sentiment Lexicon and NRC Hashtag Negated Context Sentiment Lexicon: association of words with positive (negative) sentiment in affirmative or negated contexts generated automatically from tweets with sentiment-word hashtags such as #amazing and #terrible. 
	Version: 1.0
	Number of terms: Affirmative contexts: 36,357 unigrams, 159,479 bigrams; Negated contexts: 7,592 unigrams, 23,875 bigrams
	Association scores: real-valued
	Creators: Svetlana Kiritchenko and Saif M. Mohammad


7. NRC Emoticon Lexicon (a.k.a. Sentiment140 Lexicon): association of words with positive (negative) sentiment generated automatically from tweets with emoticons such as :) and :(. 
	Version: 1.0
	Number of terms: 62,468 unigrams, 677,698 bigrams, 480,010 pairs
	Association scores: real-valued
	Creators: Saif M. Mohammad and Svetlana Kiritchenko


8. NRC Emoticon Affirmative Context Lexicon and NRC Emoticon  Negated Context Lexicon: association of words with positive (negative) sentiment in affirmative or negated contexts generated automatically from tweets with emoticons such as :) and :(.
	Version: 1.0
	Number of terms: Affirmative contexts: 45,255 unigrams, 240,076 bigrams; Negated contexts: 9,891 unigrams, 34,093 bigrams
	Association scores: real-valued
	Creators: Svetlana Kiritchenko and Saif M. Mohammad

	Papers for (5), (6), (7), and (8):

	Sentiment Analysis of Short Informal Texts. Svetlana Kiritchenko, Xiaodan Zhu and Saif Mohammad. Journal of Artificial Intelligence Research, volume 50, pages 723-762, August 2014.    

	NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets, Saif M. Mohammad, Svetlana Kiritchenko, and Xiaodan Zhu, In Proceedings of the seventh International Workshop on Semantic Evaluation Exercises (SemEval-2013), June 2013, Atlanta, USA. 

	NRC-Canada-2014: Recent Improvements in Sentiment Analysis of Tweets, Xiaodan Zhu, Svetlana Kiritchenko, and Saif M. Mohammad. In Proceedings of the eigth International Workshop on Semantic Evaluation Exercises (SemEval-2014), August 2014, Dublin, Ireland.    


9. NRC Word-Colour Association Lexicon: association of words with colours manually annotated on Amazon's Mechanical Turk.
	Version: 0.92
	Number of terms: 14,182 unigrams (words), ~25,000 word senses
	Association scores: binary (associated or not)
	Creator: Saif M. Mohammad

	Papers for (9):

	Colourful Language: Measuring Word-Colour Associations, Saif Mohammad, In Proceedings of the ACL 2011 Workshop on Cognitive Modeling and Computational Linguistics (CMCL), June 2011, Portland, OR.

	Even the Abstract have Colour: Consensus in Word-Colour Associations, Saif Mohammad, In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, June 2011, Portland, OR.

 


CONTACT INFORMATION
-------------------
Saif M. Mohammad
Senior Research Officer, National Research Council Canada
email: saif.mohammad@nrc-cnrc.gc.ca
phone: +1-613-993-0620

