#!/usr/bin/env python3


from collections import defaultdict as ddict
import numpy as np

class GraphSearch(object):
    """A class that performs path traversals of a graph

    Attributes
    ----------
    nodes : list of str
        A name for each unique node in the graph

    edges : dict of list
        The graph edges. The keys are the edge name; e.g., "a~b",
        and the values are the list of nodes in the edge; e.g., ["a","b"]

    Methods
    -------
    """
    
    def __init__(self,edges):
        from collections import defaultdict as ddict
        self.nodes = list(set([ x for e in edges for x in e.split("~") ]))
        self.nodes.sort()
        self.edges = ddict(list)
        for e in edges:
            a,b=e.split("~")
            self.edges[a].append(b)
            self.edges[b].append(a)

            
    def PathToStr(self,path):
        """Return a path name, given a path

        Parameters
        ----------
        path : list of str
            The list of node names defining a path; e.g., ["a","b"]

        Returns
        name : str
            The path name; e.g., "a~b"
        """
        return "~".join(path)

    
    def StrToPath(self,name):
        """Return a path, given its name

        Parameters
        ----------
        name : str
            The path name; e.g., "a~b"

        Returns
        -------
        path : list of str
            The list of node names defining a path; e.g., ["a","b"]
        """
        return name.split("~")

    
    def FindAllPaths(self,snode,tnode,minsize=2):
        """Return a list of all paths that connect snode to tnode

        Parameters
        ----------
        snode : str
            The name of the starting node

        tnode : str
            The name of the ending node

        minsize : int, default=2
            The paths that reference fewer than minsize nodes are
            excluded from the returned list of paths

        Returns
        -------
        paths : list of list of str
            The found paths. Each path is a list of nodes
        """
        from collections import defaultdict as ddict
        if snode not in self.nodes:
            raise Exception("Node %s is not in graph"%(snode))
        if tnode not in self.nodes:
            raise Exception("Node %s is not in graph"%(tnode))
        allpaths = self._find(
            snode,
            tnode,
            path=None,
            visited=None,
            allpaths=None,
        )
        if len(allpaths) > 0:
            if minsize > 2:
                sizes = np.array([ len(path) for path in allpaths ],dtype=int)
                allpaths = [allpaths[x] for x in np.where( sizes >= minsize )[0]]
            allpaths.sort()
        return allpaths

    
    # def FindMinPaths_orig(self,snode,tnode,minsize=2):
    #     """Return a list of all minimum-length paths that 
    #     connect snode to tnode. Although there are many long and short
    #     pathways that may connect the nodes, this function only returns
    #     the list of paths that reference the fewest possible number of
    #     nodes to connect the endpoints.

    #     Parameters
    #     ----------
    #     snode : str
    #         The name of the starting node

    #     tnode : str
    #         The name of the ending node

    #     minsize : int, default=2
    #         The returned minimum-length paths will not be smaller than
    #         minsize

    #     Returns
    #     -------
    #     paths : list of list of str
    #         The found paths. Each path is a list of nodes
    #     """
    #     allpaths = [ x for x in self.FindAllPaths(snode,tnode)
    #                  if len(x) >= minsize ]
    #     if len(allpaths) > 0:
    #         sizes = np.array([ len(path) for path in allpaths ],dtype=int)
    #         return [allpaths[x]
    #                 for x in np.where( sizes == sizes.min() )[0]]
    #     else:
    #         return []

        
    def FindMinPaths(self,snode,tnode,minsize=2):
        """Return a list of all minimum-length paths that 
        connect snode to tnode. Although there are many long and short
        pathways that may connect the nodes, this function only returns
        the list of paths that reference the fewest possible number of
        nodes to connect the endpoints.

        Parameters
        ----------
        snode : str
            The name of the starting node

        tnode : str
            The name of the ending node

        minsize : int, default=2
            The returned minimum-length paths will not be smaller than
            minsize

        Returns
        -------
        paths : list of list of str
            The found paths. Each path is a list of nodes
        """
        import numpy as np
        if snode not in self.nodes:
            raise Exception("Node %s is not in graph"%(snode))
        if tnode not in self.nodes:
            raise Exception("Node %s is not in graph"%(tnode))
        allpaths = self._findmin(
            snode,
            tnode,
            path=None,
            visited=None,
            allpaths=None,
            minsize=minsize,
        )
        if len(allpaths) > 0:
            sizes = np.array([ len(path) for path in allpaths ],dtype=int)
            return [allpaths[x]
                    for x in np.where( sizes == sizes.min() )[0]]
        else:
            return []

        
    def FindAllCycles(self,minsize=3):
        """Returns all unique closed cycles within a graph

        Parameters
        ----------
        minsize : int, default=3
            All cycles that contain fewer than minsize nodes will be
            excluded

        Returns
        -------
        paths : list of list of str
            The found paths. Each path is a list of nodes
        """
        import numpy as np
        cycs=[]
        for snode in self.nodes:
            for tnode in self.edges[snode]:
                paths = self.FindAllPaths(snode,tnode,minsize=3)
                for path in paths:
                    minnode = min(path)
                    minidx = [i for i,j in enumerate(path) if j == minnode][0]
                    path = path[minidx:] + path[:minidx]
                    if path[-1] < path[1]:
                        f = path.pop(0)
                        path.append(f)
                        path.reverse()
                    path.append(path[0])
                    cycs.append( self.PathToStr(path) )
        allpaths = [ self.StrToPath(x) for x in list(set(cycs)) ]
        if len(allpaths) > 0:
            if minsize > 2:
                sizes = np.array([ len(path) for path in allpaths ],dtype=int)
                allpaths = [allpaths[x] for x in np.where( sizes >= minsize )[0]]
            allpaths.sort()            
        return allpaths

    
    # def FindMinCycles_orig(self):
    #     """Returns all minimum-length unique closed cycles within a graph.
    #     That is, the returned cycles cannot be expressed as a sum of two
    #     smaller cycles.

    #     Parameters
    #     ----------
    #     minsize : int, default=3
    #         All cycles that contain fewer than minsize nodes will be
    #         excluded

    #     Returns
    #     -------
    #     paths : list of list of str
    #         The found paths. Each path is a list of nodes
    #     """
    #     cycs=[]
    #     for snode in self.nodes:
    #         for tnode in self.edges[snode]:
    #             paths = self.FindMinPaths(snode,tnode,minsize=3)
    #             for path in paths:
    #                 minnode = min(path)
    #                 minidx = [i for i,j in enumerate(path) if j == minnode][0]
    #                 path = path[minidx:] + path[:minidx]
    #                 if path[-1] < path[1]:
    #                     f = path.pop(0)
    #                     path.append(f)
    #                     path.reverse()
    #                 path.append(path[0])
    #                 cycs.append( self.PathToStr(path) )
    #     allpaths = [ self.StrToPath(x) for x in list(set(cycs)) ]
    #     allpaths.sort()
    #     return allpaths

    def FindMinCycles(self):
        """Returns all minimum-length unique closed cycles within a graph.
        That is, the returned cycles cannot be expressed as a sum of two
        smaller cycles.

        Parameters
        ----------
        minsize : int, default=3
            All cycles that contain fewer than minsize nodes will be
            excluded

        Returns
        -------
        paths : list of list of str
            The found paths. Each path is a list of nodes
        """
        cycs=[]
        seen=[]
        for snode in self.nodes:
            for tnode in self.edges[snode]:
                st=(snode,tnode)
                ts=(tnode,snode)
                if ts in seen:
                    continue
                seen.append(st)
                
                paths = self.FindMinPaths(snode,tnode,minsize=3)
                for path in paths:
                    minnode = min(path)
                    minidx = [i for i,j in enumerate(path) if j == minnode][0]
                    path = path[minidx:] + path[:minidx]
                    if path[-1] < path[1]:
                        f = path.pop(0)
                        path.append(f)
                        path.reverse()
                    path.append(path[0])
                    cycs.append( self.PathToStr(path) )
        allpaths = [ self.StrToPath(x) for x in list(set(cycs)) ]
        allpaths.sort()
        return allpaths


    def ComponentBeyondBond(self, atom_in, atom_out):
        """Return nodes on the ``atom_out`` side of the ``atom_in``–``atom_out`` bond.

        Mirrors scission's ``component_beyond_bond``: BFS from ``atom_out``
        without crossing ``atom_in``. Includes ``atom_out``. Runs in ``O(N+E)``.

        Parameters
        ----------
        atom_in : str
            Node on the blocked side of the cut bond.
        atom_out : str
            Node on the far side; search starts here.

        Returns
        -------
        set of str
            Nodes reachable from ``atom_out`` without passing through ``atom_in``.
        """
        if atom_in not in self.nodes:
            raise Exception("Node %s is not in graph" % (atom_in,))
        if atom_out not in self.nodes:
            raise Exception("Node %s is not in graph" % (atom_out,))
        visited = {atom_in}
        stack = [atom_out]
        component = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for nbr in self.edges[node]:
                if nbr not in visited:
                    stack.append(nbr)
        return component

                
    def _find(self,start,stop,path=None,
              visited=None,
              allpaths=None):
        """Utility function that recursively traverses the graph to
        find paths.  One should instead use the FindAllPaths method,
        which uses this recursive algorithm, but which is harder to
        screw up, because of python's weird behavior of saved-state
        in recursive functions

        Parameters
        ----------
        start : str
            The starting node

        stop : str
            The stopping node

        path : list of str or None
            The current list of nodes in the path 
            (``None`` starts a fresh path)

        visited : dict of bool or None
            Indicates if a node has already been used within the path

        allpaths : list of list of str or None
            The list of all paths found in the graph
        
        Returns
        -------
        allpaths : list of list of str
            The list of all paths found in the graph
        """

        #
        # Adapted from
        # https://www.geeksforgeeks.org/find-paths-given-source-destination
        # Recursive Depth First Traversal with a boolean array used to avoid
        # revisiting a node twice
        #
        if path is None:
            path = []
        if visited is None:
            visited = ddict(lambda: False)
        if allpaths is None:
            allpaths = []

        visited[start]=True
        path.append(start)
        if start == stop:
            q = [ p for p in path ]
            n = len(q)
            skip=False
            for h in allpaths:
                if len(h) == n:
                    same=True
                    for i in range(n):
                        if q[i] != h[i]:
                            same=False
                            break
                    if same:
                        skip=True
            if not skip:
                allpaths.append( q )
        else:
            for i in self.edges[start]:
                if not visited[i]:
                    allpaths=self._find(i,stop,path,visited,allpaths=allpaths)
        path.pop()
        visited[start]=False
        return allpaths

 
    def _findmin(self,start,stop,path=None,
                 visited=None,
                 allpaths=None,minsize=2):
        """Utility function that recursively traverses the graph to
        find paths.  One should instead use the FindAllPaths method,
        which uses this recursive algorithm, but which is harder to
        screw up, because of python's weird behavior of saved-state
        in recursive functions

        Parameters
        ----------
        start : str
            The starting node

        stop : str
            The stopping node

        path : list of str or None
            The current list of nodes in the path 
            (``None`` starts a fresh path)

        visited : dict of bool or None
            Indicates if a node has already been used within the path

        allpaths : list of list of str or None
            The list of all paths found in the graph
        
        Returns
        -------
        allpaths : list of list of str
            The list of all paths found in the graph
        """
        import numpy as np
        #
        # Adapted from
        # https://www.geeksforgeeks.org/find-paths-given-source-destination
        # Recursive Depth First Traversal with a boolean array used to avoid
        # revisiting a node twice
        #
        if path is None:
            path = []
        if visited is None:
            visited = ddict(lambda: False)
        if allpaths is None:
            allpaths = []

        visited[start]=True
        path.append(start)
        pathlen = len(path)
        
        if start == stop:
            q = [ p for p in path ]
            n = len(q)
            if pathlen >= minsize:
                skip=False
                for h in allpaths:
                    if len(h) == n:
                        same=True
                        for i in range(n):
                            if q[i] != h[i]:
                                same=False
                                break
                        if same:
                            skip=True
                if not skip:
                    allpaths.append( q )
        else:
            curmin = 0
            if len(allpaths) > 0:
                curmin = np.amin( [len(p) for p in allpaths] )
            if pathlen < curmin or curmin < minsize:
                for i in self.edges[start]:
                    if not visited[i]:
                        allpaths=self._findmin(i,stop,path,visited,
                                               allpaths=allpaths,
                                               minsize=minsize)
        path.pop()
        visited[start]=False
        return allpaths

