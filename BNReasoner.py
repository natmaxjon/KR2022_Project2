from itertools import combinations, product
import pandas as pd
from typing import Union, List
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

    def prune(self, Q, e):
        """
        Given a set of query variables Q and evidence e, performs node and edge pruning on
        the BN such that queries of the form P(Q|e) can still be correctly calculated.

        :param Q: List of query variables
        :param e: List of query variables
        """
        # Edge pruning - remove outgoing edges of every node in evidence e
        for node in e:
            for edge in list(self.bn.out_edges(node)):
                self.bn.del_edge(edge)
        
        # Node pruning - iteratively delete any leaf nodes that do not appear in Q or e
        node_deleted = True
        while node_deleted:
            node_deleted = False
            for node in self.bn.get_all_variables():
                if self.bn.is_leaf_node(node) and (node not in [*Q, *e]):
                    self.bn.del_var(node)
                    node_deleted = True
                    break
    
    def is_dsep(self, X, Y, Z):
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
    
    def is_independent(self, X, Y, Z):
        """
        Given three sets of variables X, Y, and Z, determine whether X is independent of Y given Z
        """
        return self.is_dsep(X, Y, Z)

    def compute_factor(self):
        pass
    
    def marginalization(self, X, cpt):
        """
        This function computes the CPT in which the variable X is summed-out 
        """

        # Delete node X
        new_cpt = cpt.drop([X], axis=1)
        variables_left = [variable for variable in new_cpt.columns if variable != X and variable != 'p']

        # Take the sum of the factors
        new_cpt = new_cpt.groupby(variables_left).agg({'p': 'sum'})
        cpt.reset_index(inplace=True)

        return new_cpt

    def maxing_out(self, X, cpt):
        """
        This function computes the CPT in which the variable X is maxed-out
        """
        
        # Delete node X
        new_cpt = cpt.drop([X], axis=1)
        variables_left = [variable for variable in new_cpt.columns if variable != X and variable != 'p']

        # Take the max of the factors
        new_cpt = new_cpt.groupby(variables_left).agg({'p': 'max'})
        cpt.reset_index(inplace=True)

        return new_cpt

    def factor_multiplication(self, cpt1, cpt2):
        """
        This function computes the multiplied factor of two factors for two cpt's
        """

        # Add an edge between every neighbour of 𝑋 that is not already connected by an edge
        
        pass

    
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
            print(node)

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

    def min_fill_ordering(self, X):
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

    def elimination_order(self, X, heuristic=None):
        if heuristic is None:
            order = list(X)
        elif heuristic == 'min_deg':
            order = self.min_degree_ordering(X)
        elif heuristic == 'min_fill':
            order = self.min_fill_ordering(X)
        else:
            raise ValueError('Unknown ordering heuristic')

        return order

    def variable_elimination(self, cpt, X, heuristic=None):
        """
        Sum out a set of variables by using variable elimination. 
        """
        new_cpt = cpt.copy()
        order = self.elimination_order(X, heuristic)

        for node in order:
            if node in new_cpt and len(new_cpt.columns) != 1:
                new_cpt = self.marginalization(node, cpt)
        
        return new_cpt


    def marginal_distributions(self, Q, e=None):
        """
        Given query variables Q and possibly empty evidence e, compute the marginal distribution P(Q|e). 
        Note that Q is a subset of the variables 
        in the Bayesian network X with Q ⊂ X but can also be Q = X. 
        """

        
        
        # reduce e
        # compute probability Q and e - P(Q, e)
        # compute probability of e
        # compute P(Q, e)/P(e)

        pass
        

if __name__ == "__main__":
    bayes = BNReasoner('testing/lecture_example.BIFXML')
    bayes.min_degree_ordering({'I', 'J'})
    