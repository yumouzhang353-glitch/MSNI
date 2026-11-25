class ScopedCore:
    def __init__(self, module, tie_closed):
        self.module = module
        self.tie_closed = bool(tie_closed)

    def __getattr__(self, name):
        return getattr(self.module, name)

    def natural_neighbor_search(self, X):
        function = (
            self.module.natural_neighbor_search
            if self.tie_closed
            else self.module._original_natural_neighbor_search
        )
        return function(X)

    def compute_original_graph_info(self, X):
        (r, NN, RNN, nb, tree) = self.natural_neighbor_search(X)
        return dict(r=r, NN=NN, RNN=RNN, nb=nb, tree=tree)
