#!/usr/bin/env python
# coding: utf-8

# ## Energy saved from recycling
# <p>Did you know that recycling saves energy by reducing or eliminating the need to make materials from scratch? For example, aluminum can manufacturers can skip the energy-costly process of producing aluminum from ore by cleaning and melting recycled cans. Aluminum is classified as a non-ferrous metal.</p>
# <p>Singapore has an ambitious goal of becoming a zero-waste nation. The amount of waste disposed of in Singapore has increased seven-fold over the last 40 years. At this rate, Semakau Landfill, Singapore’s only landfill, will run out of space by 2035. Making matters worse, Singapore has limited land for building new incineration plants or landfills.</p>
# <p>The government would like to motivate citizens by sharing the total energy that the combined recycling efforts have saved every year. They have asked you to help them.</p>
# <p>You have been provided with three datasets. The data come from different teams, so the names of waste types may differ.</p>
# <div style="background-color: #efebe4; color: #05192d; text-align:left; vertical-align: middle; padding: 15px 25px 15px 25px; line-height: 1.6;">
#     <div style="font-size:16px"><b>datasets/wastestats.csv - Recycling statistics per waste type for the period 2003 to 2017</b>
#     </div>
#     <div>Source: <a href="https://www.nea.gov.sg/our-services/waste-management/waste-statistics-and-overall-recycling">Singapore National Environment Agency</a></div>
# <ul>
#     <li><b>waste_type: </b>The type of waste recycled.</li>
#     <li><b>waste_disposed_of_tonne: </b>The amount of waste that could not be recycled (in metric tonnes).</li>
#     <li><b>total_waste_recycle_tonne: </b>The amount of waste that could be recycled (in metric tonnes).</li>
#     <li><b>total_waste_generated: </b>The total amount of waste collected before recycling (in metric tonnes).</li>
#     <li><b>recycling_rate: </b>The amount of waste recycled per tonne of waste generated.</li>
#     <li><b>year: </b>The recycling year.</li>
# </ul>
#     </div>
# <div style="background-color: #efebe4; color: #05192d; text-align:left; vertical-align: middle; padding: 15px 25px 15px 25px; line-height: 1.6; margin-top: 17px;">
#     <div style="font-size:16px"><b>datasets/2018_2019_waste.csv - Recycling statistics per waste type for the period 2018 to 2019</b>
#     </div>
#     <div> Source: <a href="https://www.nea.gov.sg/our-services/waste-management/waste-statistics-and-overall-recycling">Singapore National Environment Agency</a></div>
# <ul>
#     <li><b>Waste Type: </b>The type of waste recycled.</li>
#     <li><b>Total Generated: </b>The total amount of waste collected before recycling (in thousands of metric tonnes).</li> 
#     <li><b>Total Recycled: </b>The amount of waste that could be recycled. (in thousands of metric tonnes).</li>
#     <li><b>Year: </b>The recycling year.</li>
# </ul>
#     </div>
# <div style="background-color: #efebe4; color: #05192d; text-align:left; vertical-align: middle; padding: 15px 25px 15px 25px; line-height: 1.6; margin-top: 17px;">
#     <div style="font-size:16px"><b>datasets/energy_saved.csv -  Estimations of the amount of energy saved per waste type in kWh</b>
#     </div>
# <ul>
#     <li><b>material: </b>The type of waste recycled.</li>
#     <li><b>energy_saved: </b>An estimate of the energy saved (in kiloWatt hour) by recycling a metric tonne of waste.</li> 
#     <li><b>crude_oil_saved: </b>An estimate of the number of barrels of oil saved by recycling a metric tonne of waste.</li>
# </ul>
# 
# </div>
# <pre><code>
# </code></pre>

# In[299]:


import pandas as pd
import numpy as np


# # Helper functions for string matching

# In[300]:


def levenshtein_ratio_and_distance(s, t, ratio_calc = True):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])
def best_levenshtein_ratio(s:str, possible_matches:list):
    '''
    Returns best levenshtein ratio from a list of possible matches
    
    Parameters
    * s - input string
    * possible_matches - list of possible matches
    
    Returns
    * best_ratio (float) - best possible ratio
    '''
    ratios = [levenshtein_ratio_and_distance(s, t, ratio_calc = True) for t in possible_matches]
    return max(ratios)
def best_levenshtein_match(s:str, possible_matches:list):
    '''
    Returns best levenshtein ratio from a list of possible matches
    
    Parameters
    * s - input string
    * possible_matches - list of possible matches
    
    Returns
    * best_ratio (str) - best possible match
    '''  
    ratios = [levenshtein_ratio_and_distance(s, t, ratio_calc = True) for t in possible_matches]
    return possible_matches[ratios.index(max(ratios))]


# # df0 - dimension table for the energy savings per material

# In[301]:


# df0 - dimension table for the energy savings per material
df0 = pd.read_csv('datasets/energy_saved.csv', skiprows=3)
display(df0.head())
# swapaxes and remove index
df0 = df0.swapaxes("index", "columns").iloc[1:]
df0 = df0.reset_index()
# rename columns
df0.columns = ["material", "kWh_energy_saved_per_tonne", "barrels_crude_oil_saved_per_tonne"]
# convert material to lower case
df0['material'] = df0['material'].apply(str.lower)
# remove units and convert to float
df0["kWh_energy_saved_per_tonne"] = df0["kWh_energy_saved_per_tonne"].str.replace(r'\D', '').astype(float) #reg ex not a digit
df0["barrels_crude_oil_saved_per_tonne"] = df0["barrels_crude_oil_saved_per_tonne"].str.replace(r'\D', '').astype(float)
# create list of materials to be used in string matching
MATERIALS = list(df0['material'])
# have a look
display(df0.head())
df0.info()


# In[302]:


df0.reset_index()


# # df1 - Recycling statistics per waste type for the period 2003 to 2017

# In[303]:


# df1 - Recycling statistics per waste type for the period 2003 to 2017
df1 = pd.read_csv('datasets/wastestats.csv', usecols=['waste_type','total_waste_recycled_tonne','year'])
# rename columns
df1.columns = ['material','total_waste_recycled_tonne','year']
# convert materials to lower case
df1['material'] = df1['material'].apply(str.lower)
# have a look
display(df1.head())
df1.info()


# In[304]:


# look at materials
df1['material'].value_counts()


# In[305]:


# compute the best possible ratio and match for each material in df1
df1['best_lev_ratio'] = df1['material'].apply(best_levenshtein_ratio, possible_matches=MATERIALS)
df1['best_lev_match'] = df1['material'].apply(best_levenshtein_match, possible_matches=MATERIALS)


# In[306]:


# get only material that has lev ratio > some threshold
DF1_LEV_RATIO_THRESHOLD = 0.5
df1 = df1[df1.best_lev_ratio >= DF1_LEV_RATIO_THRESHOLD]
df1['material'].value_counts()


# In[307]:


# matching looks good --> use best_lev_match as material name
df1 = df1.loc[:,['year','best_lev_match','total_waste_recycled_tonne']]
df1.columns = ['year','material','total_waste_recycled_tonne']
# have a look
display(df1.head())
display(df1['year'].value_counts())
display(df1['material'].value_counts())
df1.info()


# # df2 - Recycling statistics per waste type for the period 2018 to 2019

# In[308]:


# df2 - Recycling statistics per waste type for the period 2018 to 2019
df2 = pd.read_csv('datasets/2018_2019_waste.csv')
# rename and reorder columns
df2.columns = ['material','total_waste_generated','total_waste_recycled_tonne','year']
df2 = df2.loc[:,['year','material','total_waste_recycled_tonne']]
# convert units
df2['total_waste_recycled_tonne'] = df2['total_waste_recycled_tonne'].mul(1000)
# convert materials to lower case
df2['material'] = df2['material'].apply(str.lower)

display(df2.info())
df2.head()


# In[309]:


df2.material.value_counts()


# In[310]:


df2.year.value_counts()


# In[311]:


# compute the best possible ratio and match for each material in df2
df2['best_lev_ratio'] = df2['material'].apply(best_levenshtein_ratio, possible_matches=MATERIALS)
df2['best_lev_match'] = df2['material'].apply(best_levenshtein_match, possible_matches=MATERIALS)

# get only material that has lev ratio > some threshold
DF2_LEV_RATIO_THRESHOLD = 0.5
df2 = df2[df2.best_lev_ratio >= DF2_LEV_RATIO_THRESHOLD]
df2['material'].value_counts()


# In[312]:


# matching looks good --> use best_lev_match as material name
df2 = df2.loc[:,['year','best_lev_match','total_waste_recycled_tonne']]
df2.columns = ['year','material','total_waste_recycled_tonne']
# have a look
display(df2.head())
display(df2['year'].value_counts())
display(df2['material'].value_counts())
df2.info()


# # Putting it all together

# In[313]:


# concat and sort by year and material
df = pd.concat([df1,df2]).sort_values(by=['year','material'])
# remove paper from the calculations -- they don't want to see it
df = df[df.material != 'paper']
display(df['year'].value_counts())
display(df['material'].value_counts())
# merge with df0
df = df.merge(df0, on='material', how='inner')
df.head()


# In[314]:


# calculate total_energy_saved by merging
df['total_energy_saved'] = df['total_waste_recycled_tonne'].mul(df['kWh_energy_saved_per_tonne'])
df['total_energy_saved'].head()


# In[315]:


# aggregate data to find total_energy_saved of each year
annual_energy_savings = df.groupby('year')['total_energy_saved'].sum()


# In[316]:


# slice year of interest and save answer as dataframe
annual_energy_savings = annual_energy_savings.loc[2015:].to_frame()


# In[317]:


# look at answer
annual_energy_savings


# In[318]:


get_ipython().run_cell_magic('nose', '', '\nimport pandas as pd\nimport re\nimport numpy as np\n\nconvert_index = lambda x: [re.match(\'(\\d{4})\', date).group(0) for date in x.index.values.astype(str)]\n\ntest_solution = pd.DataFrame({\'year\': [2015, 2016, 2017, 2018, 2019],\\\n                             \'total_energy_saved\': [3.435929e+09, 2554433400, 2.470596e+09, 2.698130e+09,\n       2.765440e+09]}).set_index(\'year\')\n\ndef test_project():\n    \n    # Check whether the answer has been saved and is a DataFrame\n    assert \'annual_energy_savings\' in globals() and type(annual_energy_savings) == pd.core.frame.DataFrame, \\\n    "Have you assigned your answer to a DataFrame named `annual_energy_savings`?"\n    \n    # Check whether they have the right column in their DataFrame\n    assert annual_energy_savings.columns.isin([\'total_energy_saved\']).any(), \\\n    "Your DataFrame is missing the required column!"\n    \n    # Check whether they have included the correct index\n    assert annual_energy_savings.index.name == \'year\', \\\n    "Your DataFrame is missing the required index!"\n    \n    # Check whether the values (converted to an integer) contain in the only column are correct\n    # and check whether the index is identical\n    assert (test_solution.total_energy_saved.astype(\'int64\').values == \\\n    annual_energy_savings.total_energy_saved.astype(\'int64\').values).all()\\\n    and convert_index(test_solution) == convert_index(annual_energy_savings), \\\n    "Your submitted DataFrame does not contain the correct values!"')

