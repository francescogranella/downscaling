import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices, dmatrix, Treatment

import context
context.pdsettings()

# %%
def synth_data(res, df, target_vals, year=2010, iso3='USA'):
    synth_df = df.iloc[:len(target_vals)].copy()
    y, X = dmatrices(res.model.formula, data=df, return_type='dataframe')
    synth_df = synth_df[['temp', 'temp2', 'avg_y_country', 'year', 'iso3']]
    synth_df['temp'] = target_vals
    synth_df['temp2'] = target_vals**2
    for var in ['avg_y_country', 'precip']:
        synth_df[var] = X.loc[X[f'C(iso3)[T.{iso3}]']==1, var].mean()
    synth_df['precip2'] = synth_df.precip**2
    synth_df['year'] = year
    synth_df['iso3'] = iso3
    return res.get_prediction(synth_df).summary_frame()

# Setup data
path = context.projectpath() + "/data/out/globaldata.parquet"
df = pd.read_parquet(path)
df = df[['iso3', 'year', 'lgdppc', 'gdppc_growth', 'temperature_mean', 'precip', 'avg_y_country'] + [f'D{i}' for i in range(1,11)]]
df['temp'] = df.temperature_mean
df['temp2'] = df.temperature_mean**2
df['precip2'] = df.precip**2

for n in range(1,11):
    # decile income = decile share * exp(log gdp pc)
    df[f'D{n}_gdppc'] = df[f'D{n}'] * (np.exp(df['lgdppc']))
    # decile growth = (decile income - lag(decile income))/lag(decile income)
    df = df.sort_values(by=['iso3', 'year'])
    df[f'D{n}_growth'] = df.sort_values(by=['iso3', 'year'], ascending=True).groupby('iso3', sort=False)[f'D{n}_gdppc'].pct_change()

target_countries = ['USA', 'KEN', 'MEX']
target_deciles = ['D1', 'D2', 'D5', 'D10', 'gdppc']
target_vals = np.arange(-5, 36, 1)

l = []
for target_decile in target_deciles:
    formula = f"{target_decile}_growth ~ C(year) + C(iso3) + C(iso3):year + (temp + temp2 + precip + precip2) * avg_y_country"
    mod = smf.ols(formula, data=df)
    res = mod.fit()
    for target_country in target_countries:
        _pred_df = synth_data(res, df, target_vals, year=2010, iso3=target_country)
        _pred_df['country'] = target_country
        _pred_df['decile'] = target_decile
        l.append(_pred_df)

pred_df = pd.concat(l)

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(5,15), sharex=False, sharey=False, constrained_layout=True)
for (ax, target_country) in zip(axs, target_countries):
    for target_decile in target_deciles:
        temp = pred_df[(pred_df.country == target_country) & (pred_df.decile == target_decile)]
        ax.plot(target_vals, temp['mean'], label=target_decile)
        ax.fill_between(target_vals, temp['mean_ci_lower'], temp['mean_ci_upper'], alpha=0.25)
    ax.axhline(0, color='silver', ls='dashed', zorder=0)
    ax.set_title(target_country)
ax.set_ylabel('GDP growth')
ax.set_xlabel('Temperature')
ax.legend(frameon=False)
plt.show()

# %%
from matplotlib.pyplot import cm
for target_country in target_countries:
    dat = pred_df[pred_df.country==target_country]
    colors = iter(cm.tab10.colors)
    fig, axs = plt.subplots(nrows=1, ncols=len(target_deciles), figsize=(7,5), sharex=True, sharey=True, dpi=150, constrained_layout=True)
    for i, ax in enumerate(axs):
        for target_decile in target_deciles:
            temp = dat[(dat.decile == target_decile)]
            if target_decile == target_deciles[i]:
                title = target_decile if target_decile != 'gdppc' else 'GDPpc'
                color = next(colors)
                zorder = 2
            else:
                color = 'silver'
                zorder = 1
            ax.plot(target_vals, temp['mean'], label=target_decile, color=color, zorder=zorder)
            ax.fill_between(target_vals, temp['mean_ci_lower'], temp['mean_ci_upper'], alpha=0.25, color=color, zorder=zorder)
        ax.set_title(title)
        ax.axhline(0, color='silver', ls='dashed', zorder=0)
    axs[0].set_ylabel('GDP growth')
    axs[1].set_xlabel('Temperature')

    plt.suptitle(target_country)
    plt.savefig(context.projectpath() + f'/img/decile_effect_{target_country}.png')
    plt.show()
# %%
