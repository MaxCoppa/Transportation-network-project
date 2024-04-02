class Union_set:
    """
    A class representing trees as adjacency lists and a list which represents the depth of each node, implementing various algorithms on the trees.
    Trees in the class are not directed.
    Attributes:
    -----------
    n: int
       The number of nodes.
    p: list
        The adjacency list which represents the trees.
    r: list
        A list which represents the depth of each node.
    """
    def __init__(self, n=1):
        """
        Initializes the union_set with a number of nodes, and no edges.
        Parameters:
        -----------
        n: int, optional
            The number of nodes.
        """
        self.n = n
        self.p = [i for i in range(n)]
        self.r = [1 for i in range(n)]
        self.pi = [0 for i in range(n)]
       
    def rep(self, x):
        """
        Returns the representative of x in the Union_set.
        Parameters:
        -----------
        x: int
            A node.
        Returns:
        --------
        tuple
            A tuple containing the representative of x and the power associated with the path to the representative.
        """
        u = self.p
        pi = self.pi
        i = x
        po = pi[x]
        while u[i] != i:
            i = u[i]
            if pi[i] != 0:
                po = max(pi[i], po)
        return (i, po)

    def rang(self, x):
        """
        Returns the rank of a node in the Union_set.
        Parameters:
        -----------
        x: int
            A node.
        Returns:
        --------
        int
            The rank of the node.
        """
        r0, _ = Union_set.rep(self, x)
        return self.r[r0]
   
    def fusion(self, x, y, p):
        """
        Merges the two sets which contain x and y.
        The complexity is very smooth, almost O(1) for reasonable n.
        Parameters:
        -----------
        x, y: int
            Nodes to merge.
        p: int
            The power associated with the merge.
        """
        rx, _ = Union_set.rep(self, x)
        ry, _ = Union_set.rep(self, y)
        Rx, Ry = Union_set.rang(self, x), Union_set.rang(self, y)
        if Rx > Ry:
            self.p[ry] = rx
            self.pi[ry] = p
        elif Rx < Ry:
            self.p[rx] = ry
            self.pi[rx] = p
        else:
            self.p[rx] = ry
            self.pi[rx] = p
            self.r[ry] += 1
     
    def path_root(self, x):
        """
        Returns the path to the root node of a given node x.
        Parameters:
        -----------
        x: int
            A node.
        Returns:
        --------
        list
            A list of tuples containing the nodes on the path and their associated power.
        """
        u = self.p
        pi = self.pi
        i = x
        po = pi[x]
        l = [(x, po)]
        while u[i] != i:
            i = u[i]
            if pi[i] != 0:
                po = max(pi[i], po)
            l.append((i, po))
        return l
   
    def path(self, x, y):
        """
        Returns the path between two nodes x and y.
        Parameters:
        -----------
        x, y: int
            Nodes.
        Returns:
        --------
        tuple
            A tuple containing the power of the path and a list of nodes on the path.
        """
        l1 = Union_set.path_root(self, x)
        l1.reverse()
        l2 = Union_set.path_root(self, y)
        l2.reverse()
        n = min(len(l1), len(l2))
        i = 0
        while i < n and l1[i][0] == l2[i][0]:
            i += 1
        i -= 1
        p = max(l1[i][1], l2[i][1])
        l0 = max(l1[i], l2[i])
        l = l1[i+1:]
        l.reverse()
        l.append(l0)
        return p, l + l2[i+1:]
