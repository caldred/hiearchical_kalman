import numpy as np
from collections import deque


class DAG:
    """
    Handles DAG structure, topological ordering, and computation of the B matrix.
    
    The DAG is provided as {child: [parent1, parent2, ...]} format.
    """
    
    def __init__(self, edges, skill_names=None):
        """
        edges: dict mapping child -> list of parents
               e.g. {'exit_velo': ['bat_speed'], 'xwobacon': ['exit_velo', 'launch_angle']}
        skill_names: optional list of all skill names (inferred from edges if not provided)
        """
        self.edges = edges
        
        if skill_names is None:
            skill_names = self._infer_skill_names(edges)
        self.skill_names = list(skill_names)
        self.n_skills = len(self.skill_names)
        self.skill_to_idx = {name: i for i, name in enumerate(self.skill_names)}
        
        self._validate_dag()
        self.topo_order = self._topological_sort()
    
    def _infer_skill_names(self, edges):
        names = set()
        for child, parents in edges.items():
            names.add(child)
            for p in parents:
                names.add(p)
        return sorted(names)
    
    def _validate_dag(self):
        """Check for cycles using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for parent in self.edges.get(node, []):
                if parent not in visited:
                    if dfs(parent):
                        return True
                elif parent in rec_stack:
                    raise ValueError(f"Cycle detected involving {parent}")
            rec_stack.remove(node)
            return False
        
        for node in self.skill_names:
            if node not in visited:
                if dfs(node):
                    raise ValueError("DAG contains a cycle")
    
    def _topological_sort(self):
        """Return nodes in topological order (parents before children)."""
        in_degree = {name: 0 for name in self.skill_names}
        children = {name: [] for name in self.skill_names}
        
        for child, parents in self.edges.items():
            in_degree[child] = len(parents)
            for p in parents:
                children[p].append(child)
        
        queue = deque([n for n in self.skill_names if in_degree[n] == 0])
        order = []
        
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return order
    
    def get_edge_mask(self):
        """
        Return a boolean mask of shape (n_skills, n_skills) where
        mask[i, j] = True if there is an edge from skill j to skill i.
        """
        mask = np.zeros((self.n_skills, self.n_skills), dtype=bool)
        for child, parents in self.edges.items():
            i = self.skill_to_idx[child]
            for p in parents:
                j = self.skill_to_idx[p]
                mask[i, j] = True
        return mask
    
    def get_parents(self, skill_name):
        """Return list of parent skill names for a given skill."""
        return self.edges.get(skill_name, [])
    
    def get_parent_indices(self, skill_idx):
        """Return list of parent indices for a given skill index."""
        skill_name = self.skill_names[skill_idx]
        parents = self.edges.get(skill_name, [])
        return [self.skill_to_idx[p] for p in parents]
    
    def compute_B_matrix(self, W):
        """
        Compute B = (I - W)^(-1), the total effect matrix.
        
        W[i, j] is the direct effect of skill j on skill i.
        B[i, j] is the total effect (including indirect paths) of intrinsic skill j on actual skill i.
        
        Uses solve instead of inv for numerical stability.
        """
        I = np.eye(self.n_skills)
        I_minus_W = I - W
        
        eigvals = np.linalg.eigvals(W)
        max_eigval = np.max(np.abs(eigvals))
        if max_eigval >= 1.0:
            raise ValueError(
                f"W has spectral radius {max_eigval:.4f} >= 1; (I-W) is singular or ill-conditioned"
            )
        
        B = np.linalg.solve(I_minus_W, I)
        return B
    
    def init_weight_matrix(self, init_scale=0.1):
        """
        Initialize edge weight matrix W with small random values.
        Only positions allowed by the DAG are nonzero.
        """
        W = np.zeros((self.n_skills, self.n_skills))
        mask = self.get_edge_mask()
        n_edges = mask.sum()
        W[mask] = np.random.randn(n_edges) * init_scale
        return W