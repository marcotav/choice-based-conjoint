<p align="center"><img width=42.5% src="images/ChMod-logo-black.png"></p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![image title](https://img.shields.io/badge/python-v3.6-blue.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-blue.svg) ![Image title](https://img.shields.io/badge/pandas-0.22.0-blue.svg) ![Image title](https://img.shields.io/badge/matplotlib-v2.1.2-blue.svg)






# Conjoint Analysis

From [Wiki](https://en.wikipedia.org/wiki/Conjoint_analysis):

> 'Conjoint analysis' is a survey-based statistical technique used in market research that helps determine how people value different attributes (feature, function, benefits) that make up an individual product or service.
The objective of conjoint analysis is to determine what combination of a limited number of attributes is most influential on respondent choice or decision making. A controlled set of potential products or services is shown to survey respondents and by analyzing how they make preferences between these products, the implicit valuation of the individual elements making up the product or service can be determined. These implicit valuations (utilities or part-worths) can be used to create market models that estimate market share, revenue and even profitability of new designs.

## Types of Conjoint Analysis

Here are a very brief description of a few conjoint analysis methods:

### 1) Full-Profile Conjoint Analysis 

In this type of CA, a large number of full product descriptions is displayed to the respondent yielding large amounts of data for each one of them. Different product descriptions are presented for acceptability or preference evaluations.
 
### 2) Adaptive Conjoint Analysis 
This type of CA varies the choice sets presented based on the respondents’ preference. As a consequence, the features and levels shown are increasingly more competitive optimizing the data.

### 3) Choice-Based Conjoint 
The CBC (or discrete-choice conjoint analysis) is the most common type. It requires respondents to select their preferred full-profile concept repeatedly from sets of around 3 to 5 full profile concepts. This idea is to try to simulate an actual buying scenario, and mimick shopping behavior as close as possible. From the trade-offs that are made when the respondent chooses one, or none, of the available choices, the importance of the attribute features and levels can be statistically derived. Based on the results one can estimate the value of each of the levels and also the optimal combinations that make-up products. 


### 4) Hierarchical Bayes Conjoint Analysis (HB) 
This type of CA is used to estimate utilities from respondents’ choice data. It is particularly useful when the respondent cannot provide preference evaluations for all attribute levels due to the size of the data collection. 



### 5) Max-Diff Conjoint Analysis 
This type of CA presents to the respondents an assortment of packages that must be selected under best/most preferred and worst/least preferred 

We will focus on Choice-Based Conjoint Analysis in the following:

# Choice-Based Conjoint Analysis

## Basic Assumptions

The basic assumptions of Conjoint Analysis are:
- Product are a bundle of attributes
- The utility of a product is a function of the utilities of each of its attributes
- Behavior, such as purchases, can be predicted from utilities

## Steps

- One must first choose the attributes to be included 
- The number of levels for each attribute must also be chosen
- Definition of hypothetical products (all combinations of attribute levels would generate too many products)
- One should make sure that:
 - All combinations of levels for pairs of attributes occur in some product 
 - The subset of products should have orthogonal design i.e. the chances of finding a given level of some attribute B in a product should be the same irregardless of the level of another attribute A. 
- Estimation of utilities (usually using ordinary linear regression with dummy variables)

The linear regression model with conjoint preference data has the form:

$$R_i = u_0 + u_{j}^k X_{ij}^k$$

where, $R_i$ is the ranking/rating assigned to product $i$, 


$$X_{ij}^k = \left\{ {\begin{array}{*{20}{l}}
1&{{\text{if product }}i{\text{ has level j on attribute }}k}\\
0&{{\text{otherwise}}}
\end{array}} \right.$$


and $u_j$ is the utility coefficient for level $j$ on attribute $k$. 

## Data

This dataset is based on [1].
### Importing data

The model input data has the form below. Each row corresponds to one product **profile**, a combination of **attributes**.

import pandas as pd
filename = 'data/mobile_services_ranking.csv'
pd.read_csv(filename)

## Dummy variables

We now will calculate $X_{ij}^k$ from the definition above, where we recall

$$X_{ij}^k = \left\{ {\begin{array}{*{20}{l}}
1&{{\text{if product }}i{\text{ has level j on attribute }}k}\\
0&{{\text{otherwise}}}
\end{array}} \right.$$

For example for product: 

- $X_{{\text{US Cellular}},{\text{brand}}}^{\text{0}}=1$ since the product profile on row with index $k=1$ has level $i=1={\text{Verizon}}$ on attribute $j=1={\text{brand}}$.
- $X_{{\text{4G YES}},{\text{service}}}^{\text{8}}=1$ since the product profile on row with index $k=8$ has level $i=0={\text{4G YES}}$ on attribute $j=1={\text{service}}$.

The cell below performs the following steps:
- The `for` loops over the attributes
- The first loop fixes the attribute to `brand`
- The line below counts the number of levels in the attribute `brand`

      nlevels = len(list(np.unique(conjoint_data_frame[brand])))
      
- The `aux` variables is the the list of names corresponding to that attribute (brand):

      array(['"AT&T"', '"T-Mobile"', '"US Cellular"', '"Verizon"'], dtype=object)
      
- The following line appends this array into an empty list of level names which becomes:

      level_name = [['"AT&T"', '"T-Mobile"', '"US Cellular"', '"Verizon"']]
      
- Next the variables `begin` and `end` are calculated and a list `new_part_worth` is created, which contains the part worth associated with the level 'T-Mobile'. Notice that the list has three elements since the last we be obtained by imposing zero sum:

      begin = 1 
      end = 1 + 4 - 1 = 4
      
      new_part_worth = list(main_effects_model_fit.params[1:4]) = [0.0, -0.25, 0.0]
      
- The command below grabs the parameters from `main_effects_model_fit` skipping the intercept

        main_effects_model_fit.params[1:4]) 
        
- The next line calculates the next value since the utilities are zero-centered.

- The range of levels of the attribute `brand` is appended to the `part_worth_range` list

- After the for loop is finished, we have a list of list, each sub-list containing the part-worths of an attribute

import numpy as np
import pandas as pd
import statsmodels.api as sm

def lr_params(filename):
    
    df = pd.read_csv(filename)
    
    cols = df.columns.tolist()
    
    dummies = pd.concat([pd.get_dummies(df[col], drop_first = True, prefix= col) 
                     for col in cols[0:-1]], axis=1)
    dummies.columns = [c.replace('"','').replace(" ", "_").lower() for c in dummies.columns.tolist()]

    X,y = dummies, df[cols[-1]]
    X = sm.add_constant(X)
    lr = sm.OLS(y, X).fit()
    betas = lr.params.round(3)
    v = dummies.columns.tolist()
    res = list(zip(v,betas))
    res = pd.DataFrame(res, columns=['attribute', 'beta'])
    
    attributes = ['brand', 'startup', 'monthly', 'service','retail', 'apple', 'samsung', 'google']
    levels, pw, pw_range = [],[],[]
    b = 1 
    for att in attributes:
        num_levels = len(list(np.unique(df[att])))
        levels.append(list(np.unique(df[att]))) 
        a = b 
        b = a + num_levels - 1
        pw_new = [round(i,3) for i in list(lr.params[a:b])]
        pw_new.append((-1) * sum(pw_new)) 
        pw.append(pw_new)
        pw_range.append(max(pw_new) - min(pw_new)) 
       
    
    importance = []
    for item in pw_range:
        importance.append(round(100 * (item / sum(pw_range)),2))

    name_dict = {'brand' : 'Provider', \
                 'startup' : 'Start-up Cost', 'monthly' : 'Monthly Cost', \
                 'service' : '4G Service', 'retail' : 'Nearby retail store', \
                 'apple' : 'Apple products sold', 'samsung' : 'Samsung products sold', \
                 'google' : 'Google/Nexus products sold'}  

    lst = []
    
    idx = 0 
    for att in attributes:
        print('\nAttribute and Importance:', name_dict[att],'and',importance[idx])
        print('    Level Part-Worths')
        for level in range(len(levels[idx])):
            print('       ',levels[idx][level],'-->', pw[idx][level])  
            lst.append([levels[idx][level],pw[idx][level]])
        idx = idx + 1
    
    dfnew = pd.DataFrame(list(zip(name_dict.values(),importance)), 
                         columns=['attribute', 'importance']).sort_values('importance',ascending=False)
    
    lst_new = [[lst[i][0].replace('"',''),lst[i][1]] for i in range(len(lst))]
    print(lst_new)
    tup = (lr.summary(),res,dfnew,lst_new)
    
    return tup

tup = lr_params(filename)

## Results:

print('Summary of statistics:')
tup[0]
print('Utilities:')
tup[1]
print('Importances:')
tup[2]

## References

1. [Marketing Data Science](https://www.amazon.com/Marketing-Data-Science-Techniques-Predictive/dp/0133886557)
