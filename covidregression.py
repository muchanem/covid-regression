import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from math import ceil
# 1976-2020 dataset from the MIT elections lab, i couldn't figure out how to download only 1 year
electdata = pd.read_csv("1976-2020-president.csv")
electdata = electdata.loc[(electdata["year"] == 2020) & (electdata["candidate"] == "BIDEN, JOSEPH R. JR")]


# latest vaccination data downloaded from the CDC
coviddata = pd.read_csv("covid19_vaccinations_in_the_united_states.csv", header=2)
coviddata.iat[9,0] = "District Of Columbia"
coviddata.iat[42,0] = "New York"

# 2019 ACS demographic estimates
censusdata = pd.read_csv("ACSDP1Y2019.DP05_data_with_overlays_2021-06-15T125436.csv", skiprows=[1,53])
black_por = censusdata["DP05_0038PE"] 
nonwhite_por = (100 - censusdata["DP05_0037PE"])

data = pd.DataFrame({"state": electdata["state"], "abv": electdata["state_po"], "biden_share": (electdata["candidatevotes"]/electdata["totalvotes"])})
data = data.reset_index(drop=True)
data["black_por"] = black_por
data["nonwhite_por"] = nonwhite_por
for s in data.iterrows():
    data.at[s[0], "adultvacc"] = ((coviddata.loc[coviddata["State/Territory/Federal Entity"] == (s[1]["state"]).title()])["Percent of 18+ Pop with 2 Doses by State of Residence"]).iloc[0]

#data = data.loc[data["abv"] != "HI" ]
X = data["biden_share"].values.reshape(-1,1)
y = data["adultvacc"].values
reg = LinearRegression().fit(X,y)
pred = reg.predict(X)
print(reg.score(X,y))
corr, _ = pearsonr(data["biden_share"], data["adultvacc"])
corrtext = "Correlation: " + str(round(corr,3))
reg.score(X, y)

plt.scatter(X, y,  color='black')
plt.plot(X, pred, color='blue', linewidth=3)
plt.xlabel("Biden vote share")
plt.ylabel("Percent of adults with at least 1 Covid vaccination 6/15")
plt.title("Biden Vote Share vs Vaccination \n " + corrtext)
for i, label in enumerate(data["abv"].values):
    plt.annotate(label, (X[i], y[i]))

"""
resids = (y - pred).reshape(-1,1)
y_1 = data["nonwhite_por"].values
reg1 = LinearRegression().fit(resids,y_1)
pred1 = reg1.predict(resids)
print(reg.score(resids,y_1))
corr1, _ = pearsonr((y - pred), data["nonwhite_por"])
corr1text = "Correlation: " + str(round(corr1,3))
print(corr1text)
reg1.score(resids, y_1)
plt.scatter(resids, y_1,  color='black')
plt.plot(resids, pred1, color='blue', linewidth=3)
plt.xlabel("Residuals on Biden vote share vs Vaccination")
plt.ylabel("Nonwhite Population Share (ACS 2019)")
plt.title("Residuals vs Nonwhite Population (w/o HI)\n " + corr1text)
axes = plt.gca()
axes.set_ylim([0,round(ceil(max(y_1))/10) * 10])
for i, label in enumerate(data["abv"].values):
    plt.annotate(label, (resids[i], y_1[i]))
"""

#plt.savefig("nonwhitevsresidsnohi.png", bbox_inches="tight")
plt.show()
