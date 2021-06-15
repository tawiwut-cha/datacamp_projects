# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Wastewater injection from fracking became widespread in Oklahoma since 2010, and it may cause more earthquakes in the region.<br>
# ## 1. EDA of earthquakes overtime
# ## 2. Estimate the mean interearthquake times
# ## 3. Did earthquake frequency changed after fracking became widespread

# %%
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dc_stat_think as dcst


# %%
df = pd.read_csv('oklahoma_earthquakes_1950-2017.csv', skiprows=2, parse_dates=['time'])
display(df.head())
df.info()


# %%
# EDA
# 1. Number of earthquakes by year
annual_eq = df.groupby(df['time'].dt.year)['time'].count()

_ = plt.plot(annual_eq, linestyle='--', marker='.')
_ = plt.axvline(x=2010, color='r', linestyle='--')
_ = plt.title('Annual earthquakes in Oklahoma before and after fracking')
_ = plt.xlabel('Year')
_ = plt.ylabel('Earthquakes')

plt.show()


# %%
# 2. Estimating mean inter-earthquake times
# sort df by time
df = df.sort_values(by='time')
# calculate inter-earthquake times
intereq_time = df['time'].diff().dropna().dt.days
print('Before filtering out aftershocks')
display(intereq_time.describe())
# filter out aftershocks
aftershock_threshold = 5 #days
intereq_time = intereq_time[intereq_time >= 10]
print('After filtering out aftershocks')
display(intereq_time.describe())

# %% [markdown]
# Aftershocks happen within 5 days based on our filter

# %%
# visualize
_ = sns.ecdfplot(data=intereq_time)
_ = plt.xlabel('Inter-earthquake time (days)')


# %%
# calculate observed statistics
mean_intereq_time = np.mean(intereq_time)
print('Observed mean =', mean_intereq_time, 'days')
# drawbootstrap replicates of the mean to estimate the population parameter
bs_reps = dcst.draw_bs_reps(intereq_time, np.mean, 10000)
bs_mean = np.mean(bs_reps)
bs_median = np.median(bs_reps)
bs_ci = tuple(np.percentile(bs_reps, [2.5,97.5]))
# print('Bootstrap mean = {} with 95% ci = {}'.format(bs_mean, bs_ci))
# plot histogram
_ = sns.histplot(bs_reps, bins=20)
_ = plt.axvline(x=bs_median, color='red')
_ = plt.title('Bootstrap mean = {} with 95% ci = {}'.format(bs_mean, bs_ci))
plt.show()


# %%
# 3. Did earthquake frequency changed after fracking became widespread
'''
Null hypothesis: Before and after 2010 have the same mean inter-earthquake time
Test stat: Mean inter-earthquake time difference
At least as extreme as: Observed difference in mean inter-earthquake time
'''
# split df to before and after
before_fracking = df[df.time.dt.year < 2010]['time'].diff().dropna().dt.days
after_fracking = df[df.time.dt.year >= 2010]['time'].diff().dropna().dt.days
# filter out aftershocks
before_fracking = before_fracking[before_fracking >= aftershock_threshold]
after_fracking = after_fracking[after_fracking >= aftershock_threshold]
# calculate observed statistics
mean_diff_obs = np.mean(before_fracking) - np.mean(after_fracking)
print('Observed mean difference =', mean_diff_obs, 'days')


# %%
# shift the means of before and after to a common mean
common_mean = np.mean(df['time'].diff().dropna().dt.days)
before_fracking_shifted = before_fracking - np.mean(before_fracking) + common_mean
after_fracking_shifted = after_fracking - np.mean(after_fracking) + common_mean
# draw bs reps from both sample, subtract to get our test statistic
before_bs_reps = dcst.draw_bs_reps(before_fracking_shifted, np.mean, 10000)
after_bs_reps = dcst.draw_bs_reps(after_fracking_shifted, np.mean, 10000)
test_stat_bs_reps = before_bs_reps - after_bs_reps


# %%
# visualize
_ = sns.histplot(data=test_stat_bs_reps, bins=20)
_ = plt.axvline(x=mean_diff_obs, color='red')
_ = plt.xlabel('Mean inter-earthquake time difference (days)')
plt.show()
# calculate p-value
p_value = sum(test_stat_bs_reps >= mean_diff_obs) / len(test_stat_bs_reps)
print('p value:', p_value)

# %% [markdown]
# Go look at Prof. Bois' take

