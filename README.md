## Topic-Model-and-Multinomial-NLP-Model-Trained-from-Job-Description-Data

This project establishes a topic language model and a multinomial model from job description texts to categorize text

### Topic Model

Topic modeling is a method used in machine learning and natural language processing to discover abstract topics within text. The most common algorithm for topic modeling is Latent Dirichlet Allocation (LDA).

Unsupervised Learning: Topic modeling is typically an unsupervised machine learning method, which means it does not require pre-labeled training data. Instead, it identifies patterns based on the distribution of words within the documents.

Probabilistic Models: Most topic models are probabilistic, meaning they calculate the probability of topics given documents and the probability of words given topics.

Latent Dirichlet Allocation (LDA): One of the most popular topic modeling techniques is Latent Dirichlet Allocation. LDA assumes that each document is a mixture of a small number of topics and that each word's presence is attributable to one of the document's topics.

A "twenty-topic model" is a type of topic model that has been trained to identify twenty distinct topics within a collection of documents or text data.

#### Model Building & Topic Visualization
![000010](https://github.com/ANewGitHuber/Topic-Model-and-Multinomial-NLP-Model-Trained-from-Job-Description-Data/assets/88078123/906c9f95-99c7-451b-b13d-1a7968a00015)

#### Topic Digging
"The most representative documents for topic 1:"

 Topic 1: 
 	 I am filing this complaint because  has ignored my request to provide me with the documents that their company has on file that was used to verify the accounts I disputed. Being that they have gone past the 30 day mark and can not verify these accounts, under Section 611 ( 5 ) ( A ) of the FCRA - they are required to "" promptly delete all information which can not be verified '' that I have disputed. Please resolve this manner as soon as possible. Thank you.

"More words for the 20 topics"
Topic 1 Top Words:
 	 Highest Prob: verifi, file, can, disput, account, document, pleas 
 	 FREX: verifi, requir, can, section, pleas, document, resolv 
 	 Lift: manner, prompt, gone, soon, ignor, mark, possibl 
 	 Score: manner, verifi, section, prompt, account, ignor, fcra 
Topic 2 Top Words:
 	 Highest Prob: transunion, violat, result, notic, right, request, system 
 	 FREX: transunion, result, violat, notic, right, given, suffer 
 	 Lift: suffer, transunion, result, violat, code, notic, given 
 	 Score: suffer, transunion, violat, result, notic, right, system 
Topic 3 Top Words:
 	 Highest Prob: account, report, credit, open, close, remov, show 
 	 FREX: account, open, close, thei, remark, old, equifa 
 	 Lift: thei, remark, open, account, close, duplic, asap 
 	 Score: thei, account, open, close, report, remov, credit 
Topic 4 Top Words:
 	 Highest Prob: state, lien, tax, remov, report, credit, releas 
 	 FREX: lien, tax, state, releas, offic, satisfi, irs 
 	 Lift: lien, tax, irs, releas, satisfi, counti, offic 
 	 Score: lien, tax, releas, irs, state, satisfi, counti 
Topic 5 Top Words:
 	 Highest Prob: $, paid, balanc, amount, full, charg, date 
 	 FREX: balanc, $, amount, full, --, status, charg 
 	 Lift: --, settlement, balanc, settl, amount, $, limit 
 	 Score: --, $, balanc, paid, amount, charg, full 
......
(See full result in R markdown and report)

#### Word Cloud of the Words (for Topic 14)
![0000110](https://github.com/ANewGitHuber/Topic-Model-and-Multinomial-NLP-Model-Trained-from-Job-Description-Data/assets/88078123/a72260a9-bfcb-4c59-9286-a81045fdd3e1)

#### Highest Proportion Documents (for Topic 14)
(Document 1) I have I submitted  letters to the credit reporting agency asking for verification of account and how the verification as obtained. All letters were sent certified. I am questioning FCRA 611 nd FCRA 609 process. 
To date I have not received any reply to by letters sent certified on   2015 and   2015 certified. The Credit reporting agency has refused to reply and provided proper documentation for the records I listed in the correspondence and remove the items since not reply was provided. All copies of the letters and certified mail receipts are attached.

(Document 2) I have I submitted  letters to the credit reporting agency asking for verification of account and how the verification as obtained. All letters were sent certified. I am questioning FCRA 611 nd FCRA 609 process. 
To date I have not received any reply to by letters sent certified on   2015 and   2015 certified. The Credit reporting agency has refused to reply and provided proper documentation for the records I listed in the correspondence and remove the items since not reply was provided. All copies of the letters and certified mail receipts are attached.

#### Topic Connection Analysis
![0001010](https://github.com/ANewGitHuber/Topic-Model-and-Multinomial-NLP-Model-Trained-from-Job-Description-Data/assets/88078123/7c3fd34f-e501-4490-8c9d-7c2504976bbd)
![15151](https://github.com/ANewGitHuber/Topic-Model-and-Multinomial-NLP-Model-Trained-from-Job-Description-Data/assets/88078123/daa07cca-4239-459f-92bc-885894e4e71f)

To find which topics correlate with each other, we would typically look for topics with similar correlation values or overlapping confidence intervals on the x-axis.

#### LASSO Classifier Model based on Topic Proportions Feature
![011100010](https://github.com/ANewGitHuber/Topic-Model-and-Multinomial-NLP-Model-Trained-from-Job-Description-Data/assets/88078123/88c920e3-e47c-40ee-a150-9cee6977a3f5)
Note that we didn't give enough features so there is no U shape.
But we can also see that topics 15-18 are the best predictors of the outcome.

#### Model Accuracy
![00150014](https://github.com/ANewGitHuber/Topic-Model-and-Multinomial-NLP-Model-Trained-from-Job-Description-Data/assets/88078123/cc5e5c0e-df7e-4036-bdea-0a4c4e029836)
Model Accuracy (52.56%). Sentiment Benchmark (47.93%), Word-Count Benchmark (53.87%)

#### Comparison with N-gram Model
N-gram LASSO
![151515000010](https://github.com/ANewGitHuber/Topic-Model-and-Multinomial-NLP-Model-Trained-from-Job-Description-Data/assets/88078123/6868a594-38db-4e8d-93d4-53388a350f41)
![image](https://github.com/ANewGitHuber/Topic-Model-and-Multinomial-NLP-Model-Trained-from-Job-Description-Data/assets/88078123/919accf0-a9fb-4874-ba2e-589fd0c50bb4)
N-gram Model Accuracy (59.29%)
Note: There is drop in performance of the topic model compared to the ngrams

### Multinomial Classifier

A multinomial classifier is a type of model used in machine learning for classification tasks that predicts the probability of each category based on a multinomial probability distribution. This kind of classifier is particularly suited for features that can occur multiple times, such as words in text data. Each document is represented as a feature vector, where features correspond to words in the vocabulary, and the values indicate the frequency of that word in the document.

Each product category has several different "Issues" in the dataset. In the training data, create a multinomial classifier to predict the five different issues from the narrative text.

#### Model Establishment and LASSO for Multinomial
![Lasso](https://github.com/ANewGitHuber/Topic-Model-and-Multinomial-NLP-Model-Trained-from-Job-Description-Data/assets/88078123/5f67a011-9a15-40ca-8fb8-8ccdf31187da)

#### Model Accuracy
74.66%

#### Confusion Matrix
![Screenshot](https://github.com/ANewGitHuber/Topic-Model-and-Multinomial-NLP-Model-Trained-from-Job-Description-Data/assets/88078123/c2c2a97a-5082-485e-aa9a-1c66bb54a35c)
"Credit monitoring or identity protection": 44 instances were correctly predicted as "Credit monitoring or identity protection". However, 1 was incorrectly predicted as "Credit reporting company's investigation", 1 as "Improper use of my credit report", 3 as "Incorrect information on credit report", and 3 as "Unable to get credit report/credit score". Overall 44/52 are correct (84.6%).

"Credit reporting company's investigation": 275 instances were correctly predicted as "Credit reporting company's investigation". However, 2 were incorrectly predicted as "Credit monitoring or identity protection", 45 as "Improper use of my credit report", 8 as "Incorrect information on credit report", and 7 as "Unable to get credit report/credit score". Overall 275/375 are correct (73.3%)

"Improper use of my credit report": 45 instances were correctly predicted as "Improper use of my credit report". However, 2 were incorrectly predicted as "Credit monitoring or identity protection", 3 as "Credit reporting company's investigation", 5 as "Incorrect information on credit report". Overall 45/55 are correct (81.8%).

"Incorrect information on credit report": 1722 instances were correctly predicted as "Incorrect information on credit report". However, 37 were incorrectly predicted as "Credit monitoring or identity protection", 407 as "Credit reporting company's investigation", 83 as "Improper use of my credit report", 70 as "Unable to get credit report/credit score". Overall 1722/2319 are correct (74.3%).

"Unable to get credit report/credit score": 154 instances were correctly predicted as "Unable to get credit report/credit score". However, 17 were incorrectly predicted as "Credit monitoring or identity protection", 10 as "Credit reporting company's investigation", 3 as "Improper use of my credit report", 15 as "Incorrect information on credit report". Overall 154/199 are correct (77.4%)

@John Chen, Yixin, Qianru, Zhongshi, Functions @Michael Yeomans, 2024, Imperial College London. All rights to source codes are reserved.
