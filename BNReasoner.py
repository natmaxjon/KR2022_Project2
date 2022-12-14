from itertools import combinations, product
from typing import Union, List, Set
import pandas as pd

from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net
        
        self.variables = self.bn.get_all_variables()
        self.extended_factor = {}

    def prune(self, Q: List[str], e: pd.Series) -> None:
        """
        Given a set of query variables Q and evidence e, performs node and edge pruning on
        the BN such that queries of the form P(Q|e) can still be correctly calculated.

        :param Q: List of query variables. E.g.: ["X", "Y", "Z"]
        :param e: Evidence as a series of assignments. E.g.: pd.Series({"A": True, "B": False})
        """
        # Edge pruning - remove outgoing edges of every node in evidence e
        for e_node, e_value in e.items():
            for edge in list(self.bn.out_edges(e_node)):
                # Delete the edge
                self.bn.del_edge(edge)

                # Update the cpts of the nodes on the receiving side of the edge
                recv_node = edge[1]
                new_cpt = self.bn.reduce_factor(pd.Series({e_node: e_value}), self.bn.get_cpt(recv_node))
                new_cpt = self.marginalization(e_node, new_cpt)
                self.bn.update_cpt(recv_node, new_cpt)
        
        # Node pruning - iteratively delete any leaf nodes that do not appear in Q or e
        node_deleted = True
        while node_deleted:
            node_deleted = False
            for node in self.bn.get_all_variables():
                if self.bn.is_leaf_node(node) and (node not in [*Q, *list(e.keys())]):
                    self.bn.del_var(node)
                    node_deleted = True
                    break
    
    def is_dsep(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is d-separated of Y given Z.
        """
        # Delete all outgoing edges from nodes in Z
        for node in Z:
            for edge in list(self.bn.out_edges(node)):
                self.bn.del_edge(edge)
            
        # Iteratively delete any leaf nodes that are not in X, Y or Z
        node_deleted = True
        while node_deleted:
            node_deleted = False
            for node in self.bn.get_all_variables():
                if self.bn.is_leaf_node(node) and (node not in [*X, *Y, *Z]):
                    self.bn.del_var(node)
                    node_deleted = True
                    break
        
        # If X and Y are disconnected, then they are d-separated by Z
        for x in X:
            reachable_nodes = self.bn.all_reachable(x)
            if any(node in Y for node in reachable_nodes):
                return False
        
        return True
    
    def is_independent(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is independent of Y given Z
        """

        return self.is_dsep(X, Y, Z)

    def marginalization(self, X, cpt):
        """
        This function computes the CPT in which the variable X is summed-out 
        """

        variables_left = [variable for variable in cpt.columns if variable != X and variable != 'p']

        if len(variables_left) == 0:
            p = cpt["p"].sum()
            return pd.DataFrame({"T": ["T"], "p": p})
    
        # Take the sum of the factors
        new_cpt  = pd.DataFrame(cpt.groupby(variables_left, as_index=False).agg({'p': 'sum'}))
        
        return new_cpt
      
    def maxing_out(self, X, cpt):
        """
        This function computes the CPT in which the variable X is maxed-out
        """

        # Compute the CPT with the maximum probabilty when X is maxed-out 
        variables_left = [variable for variable in cpt.columns if variable != X and variable != 'p']

        if len(variables_left) == 0:
            p = cpt["p"].max()
            new_cpt = pd.DataFrame({"T": ["T"], "p": [p]})

        else:
            new_cpt = pd.DataFrame(cpt.groupby(variables_left).agg({"p": "max"}))
            new_cpt.reset_index(inplace=True)
        
        # Check if there are any previous factors in the table 
        previous_factors = [column for column in cpt.columns.tolist() if "extended factor" in column]

        # Compute the new CPT with the extended factor added
        extended_factor = pd.merge(cpt, new_cpt, on=["p"], how="inner").rename(columns= {X: "extended factor " + X})[f'extended factor {X}']
        
        if previous_factors:
            return new_cpt.assign(**dict(cpt[previous_factors]), **{f'extended factor {X}': extended_factor}) 
        else:
            return new_cpt.assign(**{f"extended factor {X}": extended_factor})

    def factor_multiplication(self, cpt1, cpt2):
        """
        This function computes the multiplied factor of two factors for two cpt's
        """
        cpt1_variables = list(cpt1.columns)
        cpt2_variables = list(cpt2.columns)
        common_variables = [variable for variable in cpt1_variables if variable in cpt2_variables and variable != 'p']

        if len(common_variables) == 0:
            cpt_combined = cpt1.merge(cpt2, suffixes=('_1', '_2'), how='cross')
        else:
            cpt_combined = cpt1.merge(cpt2, left_on=common_variables ,right_on=common_variables, suffixes=('_1', '_2'))

        cpt_combined['p'] = cpt_combined['p_1'] * cpt_combined['p_2']
        cpt_combined = cpt_combined.drop(['p_1','p_2'], axis=1)

        return pd.DataFrame(cpt_combined)

    
    def min_degree_ordering(self, X):
        """
        Given a set of variables X in the Bayesian network, 
        compute a good ordering for the elimination of X based on the min-degree heuristics.
        """

        graph = self.bn.get_interaction_graph()

        degrees = dict(graph.degree)
        for node, _ in graph.degree:
            if node not in X:
                del degrees[node]
        degrees = sorted(degrees.items(), key=lambda item: item[1])

        order = []
        while len(degrees):
            node = degrees[0][0]
            # print(node)

            # connect neighbours with each other
            neighbours = list(graph.neighbors(node)) # get all the neighbours of the node, participating in order
            for ind, _ in enumerate(neighbours):
                neighbors_i = list(graph.neighbors(neighbours[ind])) # get all the neighbours from the current neibour
                if ind < len(neighbours):
                    for jnd in range(ind + 1, len(neighbours)): # check residual neighbours
                        if neighbours[jnd] not in neighbors_i:
                            graph.add_edge(neighbours[ind], neighbours[jnd])

            # remove node from interaction graph
            graph.remove_node(node)
            order.append(node)

            # recalculate degrees
            degrees = dict(graph.degree)
            for node, _ in graph.degree:
                if node not in X:
                    del degrees[node]
            degrees = sorted(degrees.items(), key=lambda item: item[1])

        return order

    @staticmethod
    def calculate_additional_edges(graph, node):
        #TODO: rewrite with combinations
        amount = 0
        neighbours = list(graph.neighbors(node)) # get all the neighbours of the node
        for ind, _ in enumerate(neighbours):
            neighbors_i = list(graph.neighbors(neighbours[ind])) # get all the neighbours from the current neibour
            if ind < len(neighbours):
                for jnd in range(ind + 1, len(neighbours)): # check residual neighbours
                    if neighbours[jnd] not in neighbors_i:
                        amount += 1
        
        return amount

    def min_fill_ordering(self, X: Set[str]):
        """Given a set of variables X in the Bayesian network, 
        compute a good ordering for the elimination of X based on the min-fill heuristics.
        """
        graph = self.bn.get_interaction_graph()

        amounts = dict()
        for node in X:
            amounts[node] = self.calculate_additional_edges(graph, node)
        amounts = sorted(amounts.items(), key=lambda item: item[1])

        order = []
        while len(amounts):
            node = amounts[0][0]
            # print(node)

            # connect neighbours with each other
            neighbours = list(graph.neighbors(node)) # get all the neighbours of the node, participating in order
            for ind, _ in enumerate(neighbours):
                neighbors_i = list(graph.neighbors(neighbours[ind])) # get all the neighbours from the current neibour
                if ind < len(neighbours):
                    for jnd in range(ind + 1, len(neighbours)): # check residual neighbours
                        if neighbours[jnd] not in neighbors_i:
                            graph.add_edge(neighbours[ind], neighbours[jnd])

            # remove node from interaction graph
            graph.remove_node(node)
            order.append(node)

            # recalculate degrees
            amounts = dict()
            for node in (X - set(order)):
                amounts[node] = self.calculate_additional_edges(graph, node)
            amounts = sorted(amounts.items(), key=lambda item: item[1])
        
        return order

    def elimination_order(self, X: Set[str], heuristic=None):
        if heuristic is None:
            order = list(X)
        elif heuristic == 'min_deg':
            order = self.min_degree_ordering(X)
        elif heuristic == 'min_fill':
            order = self.min_fill_ordering(X)
        else:
            raise ValueError('Unknown ordering heuristic')

        return order

    def variable_elimination(self, cpt: pd.DataFrame, X: Set[str], heuristic=None):
        """
        Sum out a set of variables by using variable elimination. 
        """
        new_cpt = cpt.copy()
        order = self.elimination_order(X, heuristic)

        for node in order:
            if node in new_cpt and len(new_cpt.columns) != 1:
                new_cpt = self.marginalization(node, cpt)
        
        return new_cpt


    def marginal_distribution(self, Q: Set[str], e: pd.Series, heuristic='min_deg') -> pd.DataFrame:
        """
        Given query variables Q and possibly empty evidence e, compute the marginal distribution P(Q|e). 
        Note that Q is a subset of the variables 
        in the Bayesian network X with Q ⊂ X but can also be Q = X. 
        """

        cpts = self.bn.get_all_cpts()

        # reduce all factors w.r.t. e
        upd_cpts = {}
        for var, cpt in cpts.items():
            upd_cpt = self.bn.get_compatible_instantiations_table(e, cpt)
            upd_cpts[var] = upd_cpt

        # get all the variables that are not in Q or e
        X = set(self.bn.get_all_variables()) - set(Q) - set(e.keys())

        # get order of variables summation
        order = self.elimination_order(X, heuristic=heuristic)

        # get joint probability Q and e - P(Q, e)
        p_Q_e = pd.DataFrame()
        visited = []
        #print(order)
        for var in order:
            #print(var)
            for child in self.bn.get_children(var):
                #print(child)
                if child not in visited:
                    #print("no child")
                    if p_Q_e.size == 0:
                        #print("size 0")
                        p_Q_e = self.factor_multiplication(upd_cpts[var], upd_cpts[child])
                        visited.extend([var, child])
                    else:
                        #print("size not 0")
                        p_Q_e = self.factor_multiplication(p_Q_e, upd_cpts[child])
                        #print(p_Q_e)
                        visited.append(child)

            p_Q_e = self.marginalization(var, p_Q_e)

        # compute probability of e
        p_e = p_Q_e.copy()
        for var in Q:
            #print(f'before {p_e}')
            p_e = self.marginalization(var, p_e)
            #print(p_e)
        p_e = p_e['p'][0]

        # divide joint probability on probability of evidence
        p_Q_e['p'] = p_Q_e['p'].apply(lambda x: x/p_e)

        return p_Q_e

    def marginal_distribution_brutto(self, Q: Set[str], e: pd.Series) -> pd.DataFrame:
        """
        Given query variables Q and possibly empty evidence e, compute the marginal distribution P(Q|e). 
        Note that Q is a subset of the variables 
        in the Bayesian network X with Q ⊂ X but can also be Q = X. 
        """

    def marginal_distribution_brutto(self, Q: Set[str], e: pd.Series) -> pd.DataFrame:
        """
        Given query variables Q and possibly empty evidence e, compute the marginal distribution P(Q|e). 
        Note that Q is a subset of the variables 
        in the Bayesian network X with Q ⊂ X but can also be Q = X. 
        """

        cpts = self.bn.get_all_cpts()

        # reduce all factors w.r.t. e
        upd_cpts = {}
        for var, cpt in cpts.items():
            upd_cpt = self.bn.get_compatible_instantiations_table(e, cpt)
            upd_cpts[var] = upd_cpt

        # calculate probability of Q and e
        p_Q_and_e = None
        for var in upd_cpts:
            if p_Q_and_e is None:
                p_Q_and_e = upd_cpts[var]
            else:
                p_Q_and_e = self.factor_multiplication(p_Q_and_e, upd_cpts[var])

        # variables not in Q and e
        X = set(self.bn.get_all_variables()) - Q - set(e.keys())
        order = self.elimination_order(X, heuristic='min_deg')
        for node in order:
            p_Q_and_e = self.marginalization(node, p_Q_and_e)

        # compute probability of e
        p_e = p_Q_and_e.copy()
        for var in Q:
            p_e = self.marginalization(var, p_e)
        p_e = p_e['p'][0]

        # divide joint probability on probability of evidence
        p_Q_e = p_Q_and_e.drop(axis=1, labels=list(e.index))
        
        # normalise probabilities
        p_Q_e['p'] = p_Q_e['p'].apply(lambda x: x/p_e)

        return p_Q_e
    
    def MAP(self, Q:Set[str], e:pd.Series):
        """
        This function calculates the maximum a-posteriori instantiation and query variables
        given some (possible empty) evidence
        """
        cpt = self.marginal_distribution(Q, e)
        print(cpt)
        print("max out:", Q[0])
        cpt = self.maxing_out(Q[0], cpt)
        # for var in Q:
        #     cpt = self.maxing_out(var, cpt)

        # max = cpt["p"].max()
        # map = cpt.loc[cpt['p'] == max]

        return cpt

    def MPE(self, Q, e):
        """
        This function calculates the most probable explanation given an evidence
        """
        
        return self.MAP(Q, e)
    
if __name__ == "__main__":
    bayes = BNReasoner('testing/stroke_network.BIFXML')
    