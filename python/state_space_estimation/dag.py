import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


class dag():
    def __init__(self, m):
        '''
        Inputs:
            m: pd.DataFrame
                Contains an adjacency matrix
        '''
        self.m = m
        self.nodes = self.m.columns.values
        self.lags = np.array([n for n in self.nodes if '_1' in n])
        if not all(self.m.columns.values == self.m.index.values):
            raise ValueError('Invalid adjacency matrix (columns and rows are not the same)')
        self.isdag = self.directed()

            
    def parents(self, n):
        '''
        Inputs:
            n: String
                A node in the graph
        Returns:
            np.array(String)
                Contains the parents of node n
        '''
        if n not in self.nodes:
            raise ValueError('n is not in graph')
        return self.m.loc[self.m.loc[:,n] != 0, n].index.values
    
    
    def children(self, n):
        '''
        Inputs:
            n: String
                A node in the graph
        Returns:
            np.Array(String)
                Contains the children of node n
        '''
        if n not in self.nodes:
            raise ValueError('n is not in graph')
        return self.m.loc[:,self.m.loc[n,:] != 0].columns.values
    
    
    def directed(self):
        '''
        Returns:
            Bool
                True if the DAG is fully directed, False otherwise
        '''
        def path(n, visited):
            next_layer = self.children(n)
            if len(next_layer) == 0:
                return True
            elif len([x for x in next_layer if x in visited]) > 0:
                return False
            else:
                visited = np.append(visited, next_layer)
                return all([path(m, visited) for m in next_layer])

        return all([path(n, np.array([])) for n in self.nodes])
        
    
    def depth(self, n):
        '''
        Inputs:
            n: String
                A node in the DAG
        Returns:
            int
                The length of the shortest path to a root node from n
        '''
        if not self.isdag:
            raise ValueError('Cannot compute depth, graph is undirected')
        if len(self.parents(n)) == 0:
            return 0
        else:
            return 1 + max([self.depth(p) for p in self.parents(n)])
    
    
    def root_nodes(self):
        '''
        Returns:
            np.array(String)
                The root nodes of the DAG
        '''
        return np.array([n for n in self.nodes if len(self.parents(n)) == 0])
        
        
    def isolated_nodes(self):
        '''
        Returns:
            np.array(String)
                The nodes in the DAG with no children and no parents
        '''
        return np.array([n for n in self.nodes if (len(self.parents(n)) == 0) & (len(self.children(n)) == 0)])
    
    
    def connected_roots(self):
        '''
        Returns:
            np.array(String)
                The root nodes of the graph that have children
        '''
        return np.array([n for n in self.nodes if (len(self.parents(n)) == 0) & (len(self.children(n)) > 0)])
    
    
    def structure(self):
        '''
        Returns:
            np.ndarray
                The structure of the DAG where 1 indicates an edge and 0 the lack thereof
        '''
        M = self.m.copy()
        M[M != 0] = 1
        return M
    
    
    def shd(self, d):
        '''
        Inputs:
            d: np.ndarray
                An adjacency matrix
        Returns:
            int
                The structural hamming distancee between this DAG and d.
                This is the number of edges in this DAG that need to be 
                changed to arrive a d.
        '''
        return len(np.nonzero(self.m.values[np.nonzero(d)])) + len(np.nonzero(d[np.nonzero(self.m.values)]))
    

    def impute(self, values):
        '''
        Inputs:
            values: np.array() (len(self.nodes),)
        Performs:
            Impute the missing elements of values implied by this DAG.
        Returns:
            values: np.array() (len(self.nodes),)
        '''
        na_nodes = values[values.isna()]
        depth = 0
        try:
            max_depth = max([self.depth(n) for n in self.nodes])
        except ValueError as e:
            print('Cannot impute values as graph is not directed')
            raise e
        while depth <= max_depth:
            for node in na_nodes.index:
                if self.depth(node) == depth:
                    values[node] = np.dot(self.m.loc[:,node], values.fillna(0))
            depth += 1
        return values


    def calculate_irf(self, x_0, T=250, verbose=False):
        '''
        Inputs:
            x_0: np.array() (len(self.nodes),)
                Starting values for the IRF
            T: int
                Number of periods to compute IRF for
            verbose: Bool
                If True print progress
        Performs:
            Given starting state x_0 containing 1 or more shocks
            use this DAG to compute the implied future path of 
            the variables in the DAG for T periods
        Returns:
            pd.DataFrame (T, len(self.nodes))
        '''
        if verbose:
            print('Simulating irf...')
        for lag in self.lags:
            x_0[lag] = 0
        irf = pd.DataFrame([self.impute(x_0)], columns=self.nodes)
        for t in range(T-1):
            if verbose:
                print('Simulating t={} of {} ({}%)'.format(t+1, T, np.round(100 * ((t+1)/T), decimals=2)))
            nr = pd.Series(np.full(len(self.nodes), np.nan), index=self.nodes)
            for lag in self.lags:
                nr[lag] = irf.iloc[-1,:][lag.rstrip('_1')]
            nr = self.impute(nr)    
            irf = irf.append(nr, ignore_index=True)
        irf.loc[:,'t'] = range(T)
        irf.set_index('t', inplace=True)
        irf.drop(self.lags, axis=1, inplace=True)
        return irf


    def plot_irf(self, irf, layout=None, figsize=(20,10)):
        '''
        Inputs:
            irf: pd.DataFrame (T, len(self.nodes))
                An irf to plot (from self.calculate_irf)
            layout: tuple (2)
                The arrangement of subplots
            figsize: tuple (2)
                The dimensions of the output pot
        Returns:
            matplotlib.plt
        '''
        if layout is None:
            side = math.ceil(math.sqrt(len(irf.columns)))
            layout = (side, side)
        axes = irf.plot(subplots=True, layout=layout, 
                        color="black", legend=False,
                        figsize=figsize)
        for ax, name in zip(axes.flatten(), irf.columns.values):
            ax.axhline(y=0, color="red")
            ax.set_title(name)
        return plt
    
    
    def plot_structure(self):
        '''
        Performs:
            Return a plot of the structure of this DAG
        Returns: 
            igraph.Graph.Adjacency
        '''
        import igraph as ig
        M = self.m.values
        g = ig.Graph.Adjacency((M != 0.0).tolist())
        g.es['weight'] = M[M.nonzero()]
        g.vs['label'] = self.nodes
        g.vs['color'] = 'white'
        g.vs['size'] = 45
        return g