import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from BayesNet import BayesNet
from BNReasoner import BNReasoner

# ---------------------------- Helper Functions ----------------------------- #

def frames_are_equal(df1, df2):
    try:
        assert_frame_equal(df1, df2)
        return True
    except:
        return False

# --------------------------------- Test BNs -------------------------------- #
# Dog problem
@pytest.fixture
def bn1():
    return BNReasoner("./testing/dog_problem.BIFXML")

# Lecture example 1 (Winter?, Sprinkler? ... Slippery Road?)
@pytest.fixture
def bn2():
    bn = BayesNet()
    bn.load_from_bifxml("./testing/lecture_example.BIFXML")

    return BNReasoner(bn)

# Lecture example 2 (I, J, X, Y, O)
@pytest.fixture
def bn3():
    return BNReasoner("./testing/lecture_example2.BIFXML")

# Example for d-sep from lecture 2
@pytest.fixture
def bn4():
    variables = ['A', 'S', 'T', 'C', 'P', 'B', 'X', 'D']
    edges = [('A', 'T'), ('S', 'C'), ('S', 'B'), ('T', 'P'), ('C', 'P'), ('P', 'X'), ('P', 'D'), ('B', 'D')]
    cpts = {}
    for v in variables:
        cpts[v] = None

    bn = BayesNet()
    bn.create_bn(variables, edges, cpts)

    return BNReasoner(bn)

# ----------------------------------- Tests -----------------------------------

class TestPruning:
    def test_case1(self, bn2):
        Q = ['Wet Grass?']
        e = pd.Series({'Winter?': True, 'Rain?': False})

        bn2.prune(Q, e)

        expected_nodes = ['Winter?', 'Sprinkler?', 'Rain?', 'Wet Grass?']
        expected_edges = [('Sprinkler?', 'Wet Grass?')]
        cpt_winter = pd.DataFrame({'Winter?': [False, True], 'p': [0.4, 0.6]})
        cpt_sprinkler = pd.DataFrame({'Sprinkler?': [False, True], 'p': [0.8, 0.2]})
        cpt_rain = pd.DataFrame({'Rain?': [False, True], 'p': [0.2, 0.8]})
        cpt_wetgrass = pd.DataFrame({'Sprinkler?': [False, False, True, True], 'Wet Grass?': [False, True, False, True], 'p': [1, 0, 0.1, 0.9]})

        assert expected_nodes == bn2.bn.get_all_nodes()
        assert expected_edges == bn2.bn.get_all_edges()
        assert frames_are_equal(cpt_winter, bn2.bn.get_cpt('Winter?'))
        assert frames_are_equal(cpt_sprinkler, bn2.bn.get_cpt('Sprinkler?'))
        assert frames_are_equal(cpt_rain, bn2.bn.get_cpt('Rain?'))
        assert frames_are_equal(cpt_wetgrass, bn2.bn.get_cpt('Wet Grass?'))

class TestDSeparation:
    def test_case1(self, bn4):
        X = ['B']
        Y = ['C']
        Z = ['S']
        assert bn4.is_dsep(X, Y, Z)

    def test_case2(self, bn4):
        X = ['X']
        Y = ['S']
        Z = ['C', 'D']
        assert not bn4.is_dsep(X, Y, Z)

    def test_case3(self, bn2):

        X = ["Winter?"]
        Y = ["Slippery Road?"]
        Z = ["Rain?"]

        assert bn2.is_dsep(X, Y, Z)

class TestMarginalisation:

    def test_case1(self, bn2):
        bayes = BayesNet()
        bayes.load_from_bifxml("./testing/lecture_example.BIFXML")

        all_ctp = bayes.get_all_cpts()
        test_cpt = all_ctp["Sprinkler?"]
        X = 'Winter?'
    
        outcome = bn2.marginalization(X, test_cpt)
        expected_outcome = pd.DataFrame({"Sprinkler?": [False, True], "p": [1.05, 0.95]})
   
        assert outcome.equals(expected_outcome)
        
class TestMaxingOut:
    
    def testcase1(self, bn2):
        bayes = BayesNet()
        bayes.load_from_bifxml("./testing/lecture_example.BIFXML")
        all_cpt = bayes.get_all_cpts()
        test_cpt = all_cpt["Wet Grass?"]
        X = 'Rain?'
       
        cpt = bn2.maxing_out(X, test_cpt)
        expected_cpt = pd.DataFrame({"Sprinkler?": [False, False, True, True], "Wet Grass?": [False, True, False, True], "p": [1.00, 0.80, 0.10, 0.95],
        f'extended factor {X}': [False, True, False, True]})

        assert cpt.equals(expected_cpt)

class TestFactorMultiplication:

    def testcase1(self, bn2):
        bayes = BayesNet()
        bayes.load_from_bifxml("./testing/lecture_example.BIFXML")
        all_cpt = bayes.get_all_cpts()
        test_cpt1 = all_cpt["Winter?"]
        test_cpt2 = all_cpt["Rain?"]

        multiplication = bn2.factor_multiplication(test_cpt1, test_cpt2)
        expected = pd.DataFrame({'Winter?': [False, False, True, True], 'Rain?': [False, True, False, True], 'p': [0.36, 0.04, 0.12, 0.48]})

        assert frames_are_equal(expected, multiplication)

class TestMAP:

    def testcase1(self, bn2):

        Q = {"Slippery Road?"}
        e = pd.Series({"Winter?": True})

        map = bn2.MAP(Q, e)
        map_real = pd.DataFrame({"Winter?": [True], "p": [0.336], "extended factor Slippery Road?": [True]})
        
        assert map.equals(map_real)

class TestMPE:

    def testcase1(self, bn2):

        Q = {"Slippery Road?", "Sprinkler?", "Wet Grass?", "Rain?"}
        e = pd.Series({"Winter?": True})

        mpe = bn2.MPE(Q, e)
      
        assert not  mpe["p"].squeeze() == 0.21504


        

