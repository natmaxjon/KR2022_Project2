from BayesNet import BayesNet
from BNReasoner import BNReasoner
import pandas as pd

bn = BayesNet()
bn.load_from_bifxml("./testing/stroke_network.BIFXML")
bayes = BNReasoner(bn)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# Calculate the prior marginal of survival
Q = {"Survival"}
e = pd.Series()

prior_marg = bayes.marginal_distribution(Q, e)
print(prior_marg)

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Calculate the posterior marginal of survival, given stroke symptoms
Q = {"Survival"}
e = pd.Series({"Diabetes Type II": True, "Smoking": True, "High Blood Pressure":True, "Obesity": True})

Q = {"Survival"}
e = pd.Series({"High Blood Pressure": True})

Q = {"Survival"}
e = pd.Series({"Smoking": True})

Q = {"Survival"}
e = pd.Series({"Obesity": True})

Q = {"Survival"}
e = pd.Series({"Diabetes Type II": True})

posterior_marg = bayes.marginal_distribution(Q, e)
print(posterior_marg)

#------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate the MPE for Survival
Q = {"Smoking", "Obesity", "Diabetes Type II", "High Blood Pressure", "Stroke Symptoms", "Intracerebral Hemorrhage", "Acute Ischemic Stroke", "No Stroke", "Treatment"}
e = pd.Series({"Survival": True})

mpe = bayes.MPE(Q, e)
for index, row in mpe.iterrows():
    print(index, row)

#------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate the MAP for Smoking, Obesity, High Blood Pressure and Diabetes Type II
Q = {"Survival"}
e = pd.Series({"Diabetes Type II": True})
map_diabetes = bayes.MAP(Q, e)
print(f'The MAP of Survival, given Diabetes Type II = True: {map_diabetes}')

Q = {"Survival"}
e = pd.Series({"Smoking": True})
map_smoking = bayes.MAP(Q, e)
print(f'The MAP of Survival, given Smoking = True: {map_smoking}')

Q = {"Survival"}
e = pd.Series({"Obesity": True})
map_obesity = bayes.MAP(Q, e)
print(f'The MAP of Survival, given Obesity = True: {map_obesity}')

Q = {"Survival"}
e = pd.Series({"High Blood Pressure": True})
map_highBP = bayes.MAP(Q, e)
print(f'The MAP of Survival, given High Blood Pressure = True: {map_highBP}')

#---------------------------------------------------------------------------------------------------------------------------------------------------
Q = {"Obesity"}
e = pd.Series({"Survival": False})
map_obesity = bayes.MAP(Q, e)
print(f'The MAP of Survival, given Obesity = True: {map_obesity}')

Q = {"Smoking"}
e = pd.Series({"Survival": False})
map_obesity = bayes.MAP(Q, e)
print(f'The MAP of Survival, given Obesity = True: {map_obesity}')

Q = {"Diabetes Type II"}
e = pd.Series({"Survival": False})
map_obesity = bayes.MAP(Q, e)
print(f'The MAP of Survival, given Obesity = True: {map_obesity}')

Q = {"High Blood Pressure"}
e = pd.Series({"Survival": False})
map_obesity = bayes.MAP(Q, e)
print(f'The MAP of Survival, given Obesity = True: {map_obesity}')

Q = {"Smoking", "Obesity", "Diabetes Type II", "High Blood Pressure"}
e = pd.Series({"Survival": False})
map_obesity = bayes.MAP(Q, e)
print(f'The MAP of Survival, given Obesity = True: {map_obesity}')