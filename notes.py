from BayesNet import BayesNet
from BNReasoner import BNReasoner
import pandas as pd


# def print(self):

#         all_ctp = self.bn.get_all_cpts()

#         for ctp in all_ctp:
            
#             test_ctp = all_ctp[ctp]
        
#         test_cpt2 = all_ctp["Rain?"]
#         print(test_ctp)
#         print(test_cpt2)
#         combined_cpt = self.factor_multiplication(test_cpt2, test_ctp) 

#         print(combined_cpt)


# def print(self):

#         all_ctp = self.bn.get_all_cpts()
#         test_ctp = all_ctp["Wet Grass?"]
#         cpt = self.marginalization("Rain?", test_ctp) 

#         print(cpt)

# def print(self):

#         all_ctp = self.bn.get_all_cpts()
#         test_ctp = all_ctp["Wet Grass?"]
#         cpt, extended_factor = self.maxing_out("Rain?", test_ctp) 

#         print(cpt)

#         print(extended_factor)


# # Get instantiation of X where variable X is maxed-out
        # combined_cpt = pd.concat([cpt, new_cpt], axis=1)
        # print(combined_cpt)
        # reduced_cpt = combined_cpt.dropna(axis=0, how='any')
        # reduced_cpt[X] = reduced_cpt[X].map({True: f'{X} = True', False: f'{X} = False'}) 
        # reduced_cpt = reduced_cpt.iloc[:, :- (len(variables_left) + 1)]
        # print(reduced_cpt)
        # reduced_cpt["p"] = reduced_cpt["p"].astype(str)
        # reduced_cpt['factor'] = reduced_cpt[["p", X]].agg(': '.join, axis=1)
        # extended_factor = reduced_cpt.drop([X, "p"], axis=1)
        
        # return new_cpt, extended_factor

bn = BayesNet()
bn.load_from_bifxml("./testing/stroke_network.BIFXML")
bayes = BNReasoner(bn)

#Q = set(["Smoking", "Obesity", "Diabetes Type II", "High Blood Pressure", "Stroke Symptoms", "Intracerebral Hemorrhage", "Acute Ischemic Stroke", "No Stroke" ])
Q = set(["Smoking", "Obesity", "Diabetes Type II", "High Blood Pressure"])
e = pd.Series({"Survival": False})
X = "I"

# cpts = bn.get_all_cpts()
# test_cpt = cpts["I"]
# print(test_cpt)

#outcome = bayes.maxing_out(X, test_cpt)
outcome = bayes.marginal_distribution(Q, e)
map = bayes.MAP(Q, e)
print(map)