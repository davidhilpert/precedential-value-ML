# Machine Learning from Precedent

Precedent refers to a legal principle or rule established in a previous case that is either binding or persuasive for judges deciding similar issues in future cases. While international courts do not adhere to precedent strictly speaking, they do cite previous judgments in an effort to build consistent case-law over time (Lupu and Voeten 2012). Precedential value can be seen as the influence or authority of a judgment for future case law building. 

In this project, I draw on the powers of machine learning and natural language processing (NLP) in order to process large quantities of legal documents (both judgment texts and EU laws) and thereby look into the factors behind precedential value. A variety of machine learning frameworks (Linear Regression, Support Vector Machine, Random Forest, XGBoost) are applied to the problem. By way of concluding, potentials and limitations of the machine learning framework are discussed. 

Keywords: **Machine Learning**, **scikit-learn**, **support vector machine**, **random forest**, **XGBoost**, **NLP**

## Descriptives

The problem definition is to predict the number of citations for a given judgment over the next 3 years as a function of a feature matrix. That matrix collects (1) features that relate to legal substance at stake, (2) features that relate to how the judgment is crafted, (3) features of the court, and (4) features of the (political) context. 

The first cluster of features relates to the legal substance at stake in a given judgment. First, dummies that indicate whether the judgments affects an EU regulation (affectsR), directive (affectsL), decision (affectsD), earlier court judgment (affectsCJ) or treaty article (affectsT), respectively. Second, a count variable measures how often a given legal act has been touched upon before the judgment in question (prior_touches_min_vec). The reasoning is that when the court gets to interpret a legal act such as a treaty article or a directive for the first time, the judgment could be more controversial than if it is the tenth court case to touch upon an article. In case a judgment affects multiple legal acts, I take the minimum count in order to capture the least-interpreted, presumably most controversial legal act affected. Finally, I include a measure of novelty for legal acts developed in my dissertation. Based on structural topic models Roberts, Stewart and Nielsen (2020), I seek to quantify the extent to which legal acts break new substantive ground, which could lead to more controversial disputes before court, and higher precedential value.

The second cluster of features relates to the way the judgment is crafted. First, the year when the case was lodged at the CJEU is recorded (year_lodgment). Second, the number of days that passed between the day the case was lodged and the day the final judgment is made (days_it_took). Third, the length of the judgment text and fourth, the number of distinct pieces cited in the judgment (num_citations_vec), both being a rough approximation to the care that went into crafting the judgment. Fifth, there are two measures of substantive similarity between the judgment in question and earlier cases. This measure is created drawing on the judgment texts, which are preprocessed using standard steps, including stemming, the removal of digits and stopwords, as well as several terms frequent to CJEU judgments, such as the institutions involved. The pruned judgment texts are then converted into a TF-IDF matrix. Finally, substantive similarity is captured using cosine similarity on the TF-IDF matrix. As an alternative measure, I apply the same procedure using the subject matter classification scheme available in EUR-Lex, the official EU database on legislative data.

The third cluster of features relates to characteristics of the court. Here I measure the chamber size which proxies the degree of controversy around a dispute (Larsson & Naurin 2016). Furthermore, I explicitly control for whether a judgment is delivered by the grand chamber, as opposed to smaller chambers which consist of a subset of three or five judges and handle more routine cases. 

The fourth and final clusters relates to the wider political context. This includes, first, the country of origin for the case in question. Countries are weighted by their population size (country_of_origin_weights). Court judgments may raise attention particularly in countries where the dispute originates, and larger member states may wield greater political influence over the court. A second measure taps into the salience of the case among member state member state governments by measuring how many of them weigh in on the case by filing amicus curiae briefs num_obs_vec. Another version of this variable weights governmental briefs with population size as a proxy for political pressure (Larsson & Naurin 2016). 

Finally, a lagged version of the outcome variable, the cite count of substantively similar judgments weighted by the overall number of judgments in the past is included (lagged_hit_ratio). This autoregressive component takes into account that over the time observed, there are temporal trends to affect the production of precedent. 

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>affectsR</th>
      <td>5310</td>
      <td>0.29</td>
      <td>0.46</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>affectsL</th>
      <td>5310</td>
      <td>0.51</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>affectsD</th>
      <td>5310</td>
      <td>0.02</td>
      <td>0.14</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>affectsCJ</th>
      <td>5310</td>
      <td>0.01</td>
      <td>0.10</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>affectsT</th>
      <td>5310</td>
      <td>0.20</td>
      <td>0.40</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>prior_touches_min_vec</th>
      <td>5310</td>
      <td>33.81</td>
      <td>71.43</td>
      <td>0.00</td>
      <td>408.00</td>
    </tr>
    <tr>
      <th>act_scores</th>
      <td>5310</td>
      <td>-0.06</td>
      <td>0.71</td>
      <td>-1.00</td>
      <td>1.59</td>
    </tr>
    <tr>
      <th>year_lodgment</th>
      <td>5310</td>
      <td>2009.01</td>
      <td>7.05</td>
      <td>1995.00</td>
      <td>2019.00</td>
    </tr>
    <tr>
      <th>textlength</th>
      <td>5286</td>
      <td>6888.59</td>
      <td>3545.02</td>
      <td>1.00</td>
      <td>53818.00</td>
    </tr>
    <tr>
      <th>days_it_took</th>
      <td>5310</td>
      <td>581.72</td>
      <td>184.14</td>
      <td>40.00</td>
      <td>2056.00</td>
    </tr>
    <tr>
      <th>num_citations_vec</th>
      <td>5310</td>
      <td>4.04</td>
      <td>3.27</td>
      <td>0.00</td>
      <td>47.00</td>
    </tr>
    <tr>
      <th>cosine_sims_vec</th>
      <td>5310</td>
      <td>0.78</td>
      <td>0.13</td>
      <td>0.40</td>
      <td>1.99</td>
    </tr>
    <tr>
      <th>subj_cosine_sims_vec</th>
      <td>5310</td>
      <td>0.98</td>
      <td>0.07</td>
      <td>0.08</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>grand_chamber</th>
      <td>5310</td>
      <td>0.09</td>
      <td>0.29</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>chamber_size</th>
      <td>5310</td>
      <td>3.82</td>
      <td>2.62</td>
      <td>0.00</td>
      <td>22.00</td>
    </tr>
    <tr>
      <th>country_of_origin_weights</th>
      <td>5310</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>num_obs_vec</th>
      <td>5310</td>
      <td>2.04</td>
      <td>1.76</td>
      <td>0.00</td>
      <td>15.00</td>
    </tr>
    <tr>
      <th>amicus_curiae_weights</th>
      <td>5310</td>
      <td>0.13</td>
      <td>0.12</td>
      <td>0.00</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>lagged_hit_ratio</th>
      <td>5310</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
</div>

The following plot visualizes the bivariate correlations among these features.

![Corrplot](figures/corrplot.jpg)

## Analysis

The objective of this project is to predict the number of times a given judgment will be cited in the future, which is a regression problem. I use three different machine learning algorithms suited to this type of problem, a Random Forest regressor, extreme gradient boosting (XGBoost), and a Support Vectors Machine (SVM). For each one, hyperparameters are set through grid search. As a benchmark algorithm, I add Ordinary Least Squares regression. Prior to estimation, the data are divided into train and test sets using 80 and 20 percent of the observations, respectively. Data are normalized using a standard scaler.  

The following table shows the performance of each model. Overall, XGBoost performs best, leading to improvements of 32.9% on the RMSE and 37.9% on the MAE, respectively. 

|Algorithm           | RMSE  | Error Reduction RMSE  | MAE     | Error Reduction MAE |
|--------------------|-------|-----------------------|---------|---------------------|
| E[y]               | 3.12  | (benchmark)           | 1.96    | (benchmark)         |
| SVM                | 2.22  | 28.9%                 | 1.24    | 36.8%  |
| Linear Regression  | 2.16  | 30.8%                 | 1.23    | 37.1%  |
| Random Forest      | 2.13  | 31.8%                 | 1.22    | 37.4%  |
| XGBoost            | 2.09  | 32.9%                 | 1.21    | 37.9%  |


Feature importance is visualized in the following plot:

![Importance](figures/feature_importance.jpg)


## Reference

Lupu, Yonatan, and Erik Voeten. "Precedent in international courts: a network analysis of case citations by the European Court of Human Rights." British Journal of Political Science 42.2 (2012): 413-439.
