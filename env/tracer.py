import numpy as np
import networkx as nx
import operator as op

from warnings import warn
from itertools import chain
from math import fsum, isclose, isnan
from heapq import heappop, heappush

# use monotonic for visitation order
from time import monotonic_ns

from typing import Union
from enum import Enum
from numpy import ndarray
from collections import namedtuple

from ecole.core.scip import Model as ecole_Model

from pyscipopt import Model
from pyscipopt.scip import Node, Solution
from pyscipopt.scip import PY_SCIP_NODETYPE as SCIP_NODETYPE
from pyscipopt.scip import PY_SCIP_LPSOLSTAT as SCIP_LPSOLSTAT


# SCIP_NODETYPE: form `type_tree.h`
# FOCUSNODE   =  0  # the focus node
# PROBINGNODE =  1  # temp. child node of the focus or refocused node used for probing
# SIBLING     =  2  # unsolved sibling of the focus node
# CHILD       =  3  # unsolved child of the focus node
# LEAF        =  4  # unsolved leaf of the tree, stored in the tree's queue
# DEADEND     =  5  # temporary type of focus node, if it was solved completely
# JUNCTION    =  6  # fork without LP solution
# PSEUDOFORK  =  7  # fork without LP solution and added rows and columns
# FORK        =  8  # fork with solved LP and added rows and columns
# SUBROOT     =  9  # fork with solved LP and arbitrarily changed rows and columns
# REFOCUSNODE = 10  # refocused for domain propagation (junction, fork, or subroot)


SCIP_LPSOLSTAT_TO_NUMERIC = {
    SCIP_LPSOLSTAT.NOTSOLVED: 4,
    SCIP_LPSOLSTAT.OPTIMAL: 0,
    SCIP_LPSOLSTAT.ITERLIMIT: 1,
    SCIP_LPSOLSTAT.TIMELIMIT: 1,
    SCIP_LPSOLSTAT.INFEASIBLE: 2,
    SCIP_LPSOLSTAT.UNBOUNDEDRAY: 3,
    SCIP_LPSOLSTAT.OBJLIMIT: 4,
    SCIP_LPSOLSTAT.ERROR: 4,
}


# NT replacement for scipy's `OptimizeResult` is a dict with dot-attr access
# XXX from scipy.optimize import OptimizeResult
OptimizeResult = namedtuple("OptimizeResult", "x,success,status,message,fun,nit")


def build_optresult(
    x: ndarray = None,
    status: int = 1,
    message: str = "",
    fun: float = np.inf,
    nit: int = 0,
    **ignore,
) -> OptimizeResult:
    """Build an OptimizeResult with only the essential fields."""
    return OptimizeResult(
        x=x,
        # whether or not the optimizer exited successfully
        success=status == 0,
        # termination status code of the optimizer
        # 0 success, 1 iteration/time limit, 2 infeasible, 3 unbounded, 4 other
        status=status,
        message=message,
        fun=fun,
        nit=nit,
    )


# SCIP_NODETYPE: form `type_tree.h`
# FOCUSNODE   =  0  # the focus node
# PROBINGNODE =  1  # temp. child node of the focus or refocused node used for probing
# SIBLING     =  2  # unsolved sibling of the focus node
# CHILD       =  3  # unsolved child of the focus node
# LEAF        =  4  # unsolved leaf of the tree, stored in the tree's queue
# DEADEND     =  5  # temporary type of focus node, if it was solved completely
# JUNCTION    =  6  # fork without LP solution
# PSEUDOFORK  =  7  # fork without LP solution and added rows and columns
# FORK        =  8  # fork with solved LP and added rows and columns
# SUBROOT     =  9  # fork with solved LP and arbitrarily changed rows and columns
# REFOCUSNODE = 10  # refocused for domain propagation (junction, fork, or subroot)


def evaluate_sol(
    m: Model,
    x: Union[Solution, dict[str, float]],
    sign: float = 1.0,
    *,
    transformed: bool = True,
) -> OptimizeResult:
    """Parse SCIP's solution and evaluate it"""
    if isinstance(x, Solution):
        # pyscipopt does not implement meaningful way to extract a solution, but
        #  thankfully, its repr is that of a dict, so it is kind of a proto-dict
        x = eval(repr(x), {}, {})  # XXX dirty hack!

    if not isinstance(x, dict):
        raise NotImplementedError(type(x))

    # get the coefficients in linear objective and the offset
    # XXX it appears that getVars and sol's dict have variables is the same order
    obj = {v.name: v.getObj() for v in m.getVars(transformed)}
    if obj.keys() != x.keys():
        raise RuntimeError(f"Invalid solution `{x}` for `{obj}`")

    val = fsum(obj[k] * v for k, v in x.items())
    # XXX the LP seems to be always `\min`
    # assert isclose(val, m.getCurrentNode().getLowerbound())  # XXX lb is w.r.t. `\min`

    # we hope, the offset has the correct sign for the LPs inside nodes
    c0 = m.getObjoffset(not transformed)  # XXX to be used for comparing with incumbent
    return build_optresult(x=x, fun=sign * (c0 + val), status=-1, nit=0)


DualBound = namedtuple("DualBound", "val,node")


class TracerNodeType(Enum):
    OPEN = 0
    FOCUS = 1  # the OPEN node is being focused on
    CLOSED = 2
    FATHOMED = 3
    PRUNED = 4
    SPECIAL = -1


class TracerLogicWarning(RuntimeWarning):
    pass


class Tracer:
    """Search-tree tracer for SCIP

    Details
    -------
    Tracking SCIP through pyscipopt is hard. This logic in this class attempts
    its best to recover and assign the lp solutions, lowerbounds to the tree
    nodes. The situation is exacerbated by the very rich and complex inner logic
    of SCIP: the nodes may change their type, may have their lower bound updated
    after a visit, and may get repurposed without any prior notification.

    The entry point to the instances is `.update` method. It begins by adding the
    current focus node into the tree, making sure that every ancestor is up-to-date,
    then tracks changes to the frontier -- the so called open nodes, which represent
    yet unvisited solution regions.
    """

    sign: float
    is_worse: callable
    T: nx.DiGraph
    focus_: int
    nit_: int
    trace_: list
    duals_: list
    frontier_: set
    fathomed_: set

    def __init__(self, m: Model, *, ensure: str = None) -> None:
        """Initialize the tracer tree."""
        assert ensure is None or ensure in ("min", "max")

        # internally SCIP represents LPs as minimization problems
        sense = m.getObjectiveSense()[:3].lower()
        self.sign = -1.0 if sense == "max" else +1.0
        self.is_worse = op.lt if sense == "max" else op.gt

        # figure out the objective and how to compare the solutions
        # XXX actually we can use the `evaluate_sol(m, m.getBestSol(), self.sign)`
        inc = self.get_default_lp(float("-inf" if sense == "max" else "+inf"))

        self.duals_, self.trace_, self.focus_, self.nit_ = [], [], None, 0
        self.shadow_, self.frontier_, self.fathomed_ = None, set(), set()
        self.T = nx.DiGraph(root=None, incumbent=inc)

    @staticmethod
    def get_default_lp(fun: float = float("nan")) -> OptimizeResult:
        # fun = self.sign * (m.getObjoffset(not transformed) + n.getLowerbound())
        # default solution status is `NOTSOLVED`
        return build_optresult(x={}, fun=fun, status=4, nit=-1)

    def get_focus_lp(self, m: Model, *, transformed: bool = True) -> OptimizeResult:
        """Recover the local LP of the focus node"""
        # the only lp solution accessible through the model is the one at the focus
        # XXX the lp solution at the FOCUS node cannot be integer feasible,
        #  since otherwise we would not have been called in the first place
        x = {v.name: v.getLPSol() for v in m.getVars(transformed)}
        partial = evaluate_sol(m, x, self.sign, transformed=transformed)
        return build_optresult(
            x=partial.x,
            fun=partial.fun,
            status=SCIP_LPSOLSTAT_TO_NUMERIC[m.getLPSolstat()],
            nit=m.getNLPIterations() - self.nit_,
        )

    def create_node(self, n: Node, overwrite: bool = True) -> int:
        """Create a node in the tree with default attributes"""
        n_visits = 0

        # only open nodes and the root can be overwritten
        j = int(n.getNumber())
        if j in self.T:
            if (
                self.T.nodes[j]["type"] != TracerNodeType.OPEN
                and n.getParent() is not None
            ):
                if overwrite:
                    return j
                raise RuntimeError(j)

            n_visits = self.T.nodes[j]["n_visits"]

        self.T.add_node(
            j,
            scip_type=n.getType(),  # SIBLING, LEAF, CHILD
            type=TracerNodeType.OPEN,
            best=None,
            lp=self.get_default_lp(),
            # the lowerbound is untrusted until the node focused
            lb=float("nan"),
            n_visits=n_visits,
            n_order=-1,  # monotonic visitation order (-1 -- never visited)
        )

        return j

    def update_node(self, m: Model, n: Node) -> int:
        """Update an existing node"""
        j = int(n.getNumber())
        if j not in self.T:
            raise KeyError(j)

        # if the node has been added earlier, then update its SCIP's type
        # SIBLING -> LEAF, SIBLING -> FOCUSNODE, LEAF -> FOCUSNODE
        self.T.add_node(j, scip_type=n.getType())

        return j

    def update_lineage(self, m: Model, n: Node) -> None:
        """Add node's representation to the tree and ensure its lineage exists."""
        assert isinstance(n, Node)
        if n.getType() not in (
            SCIP_NODETYPE.SIBLING,
            SCIP_NODETYPE.CHILD,
            SCIP_NODETYPE.LEAF,
            SCIP_NODETYPE.FOCUSNODE,
        ):
            raise NotImplementedError

        v = self.update_node(m, n)
        vlp = self.T.nodes[v]["lp"]

        # ascend unless we have reached the root
        p = n.getParent()
        while p is not None:
            # guard against poorly understood node types
            if p.getType() not in (
                SCIP_NODETYPE.FOCUSNODE,
                SCIP_NODETYPE.FORK,
                SCIP_NODETYPE.SUBROOT,
                SCIP_NODETYPE.JUNCTION,
            ):
                raise NotImplementedError

            # add an un-visited node
            # XXX `add_edge` silently adds the endpoints, so we add them first
            u = self.update_node(m, p)
            ulp = self.T.nodes[u]["lp"]
            assert ulp.x  # XXX the parent must have non-default lp by design

            # get the gain using the recovered lp solutions
            # XXX `getLowerbound` is w.r.t. `\min`, so no need for the `sign`, however
            #  it's value on prior focused nodes is unreliable, e.g. infinite bound,
            #  while the node reports a valid LP solution (with `success=True`).
            # `gain = max(n.getLowerbound() - p.getLowerbound(), 0.0)`
            gain = max(self.sign * (vlp.fun - ulp.fun), 0.0)
            # np.isclose(sum(c * (vx[k] - ux[k]) for k, c in obj.items()), gain)

            # XXX unless NaN, gain IS the difference between SCIP's reported lb
            ref = self.T.nodes[v]["lb"] - self.T.nodes[u]["lb"]
            if not m.isInfinity(ref):
                if not (isnan(gain) or isclose(gain, ref, rel_tol=1e-5, abs_tol=1e-6)):
                    warn(
                        f"Recovered gain `{gain}`, lb-based `{ref}`.",
                        TracerLogicWarning,
                    )

            ref = self.T.edges.get((u, v), dict(g=float("nan")))["g"]
            if not (isnan(ref) or isclose(gain, ref, rel_tol=1e-5, abs_tol=1e-6)):
                warn(
                    f"Gain {(u, v)} changed from `{ref}` to `{gain}`.",
                    TracerLogicWarning,
                )

            # establish or update the parent (u) child (v) link
            # XXX see [SCIP_BOUNDTYPE](/src/scip/type_lp.h#L44-50) 0-lo, 1-up
            # XXX the parent branching may not exist, when SCIP is shutting down
            dir, by, frac, cost = None, None, float("nan"), float("nan")
            if n.getParentBranchings() is not None:
                (var,), (bound,), (uplo,) = n.getParentBranchings()
                # if the bound is `up` then dir should be `lo`, and vice versa
                dir = -1 if uplo > 0 else +1  # XXX same dir signs as in `toybnb.tree`
                by, cost = var.getIndex(), var.getObj()

                # XXX use the (unique) name of the splitting variable
                frac = abs(ulp.x[repr(var)] - bound)

            self.T.add_edge(u, v, key=dir, j=by, g=gain, f=frac, c=cost)

            v, vlp, n, p = u, ulp, p, p.getParent()

    def enter(self, m: Model) -> int:
        """Begin processing the focus node at the current branching point."""

        # get the focus node
        # XXX We neither borrow nor own any `scip.Node` objects, but we are
        #  guaranteed that `n` references a valid focus node at least for the
        #  duration of the current branching call
        n: Node = m.getCurrentNode()
        assert isinstance(n, Node)

        # the focus node is not integer-feasible, since we got called,
        #  and is formally an open node, since it has just been the
        #  frontier node, and bnb does not revisit nodes
        assert n.getType() == SCIP_NODETYPE.FOCUSNODE

        # add the node to the tree and recover its LP solution
        # XXX the current node might have been added earlier during the frontier,
        #  scan. SCIP guarantees that a node's number uniquely identifies a search
        #  node, even those whose memory SCIP reclaimed.
        # XXX if we're visiting a former child/sibling/leaf make sure it is OPEN
        j = self.create_node(n, overwrite=True)

        # only the root may get visited twice
        if n.getParent() is not None and self.T.nodes[j]["n_visits"] > 0:
            warn(
                f"SCIP should not revisit nodes, other than the root. Got `{j}`.",
                TracerLogicWarning,
            )

        self.T.add_node(
            j,
            scip_type=n.getType(),  # FOCUSNODE
            type=TracerNodeType.FOCUS,
            lp=self.get_focus_lp(m, transformed=True),
            lb=n.getLowerbound(),  # XXX for sanity check
            best=None,
            # use monotonic clock for recording the focus/visitation order
            n_order=monotonic_ns(),
            # n_visits=...,  # XXX updated when leaving
        )
        self.update_lineage(m, n)

        # XXX technically we don't need to store the root node, since there can
        # only be one
        if self.T.graph["root"] is None:
            self.T.graph["root"] = j

        # maintain our own pruning pq (max-heap)
        # XXX the lp value of a node is not available until it is in focus, so
        #  we do not do this, when enumerating the open frontier
        # XXX the values in `duals` are -ve (for max heap) of the lp bounds in MIN sense
        heappush(self.duals_, DualBound(-self.sign * self.T.nodes[j]["lp"].fun, j))

        # then current focus node may not have been designated by us as an open
        #  node, since it is the immediate child of the last focus node, and we
        #  can only see the open node set as it is after focusing, but before
        #  branching
        if j in self.frontier_:
            self.frontier_.remove(j)

        else:
            # j's parent is the focus node that we visited immediately before
            pass

        self.nit_ = m.getNLPIterations()
        self.focus_ = j

        return j

    def leave(self, m: Model) -> None:
        """Conclude the data collection for the last focus node from SCIP's state
        that was revealed upon calling branching at a new focus node.
        """
        assert self.focus_ is not None
        assert self.T.nodes[self.focus_]["type"] == TracerNodeType.FOCUS

        # close the focus node from our previous visit, since SCIPs bnb never
        #  revisits
        self.T.nodes[self.focus_]["type"] = TracerNodeType.CLOSED
        self.T.nodes[self.focus_]["n_visits"] += 1

    def prune(self) -> None:
        """SCIP shadow-fathoms the nodes for us. We attempt to recover, which
        nodes were fathomed by pruning.
        """

        nodes = self.T.nodes
        # convert the incumbent cutoff to MIN sense, since duals is max heap of `MIN`
        cutoff = self.sign * self.T.graph["incumbent"].fun
        while self.duals_ and (cutoff < -self.duals_[0].val):
            node = heappop(self.duals_).node
            # do not fathom nodes, re-fathomed by SCIP
            if nodes[node]["type"] == TracerNodeType.FATHOMED:
                continue

            assert nodes[node]["type"] in (TracerNodeType.OPEN, TracerNodeType.CLOSED)
            nodes[node]["type"] = TracerNodeType.PRUNED

    def add_frontier(self, m: Model) -> set:
        """Update the set of tracked open nodes and figure out shadow-visited ones."""
        leaves, children, siblings = m.getOpenNodes()
        if children:
            warn(
                "Children created prior to branching on the parent!",
                TracerLogicWarning,
            )

        # ensure all currently open nodes from SCIP are reflected in the tree
        # XXX [xternal.c](scip-8.0.1/doc/xternal.c#L3668-3686) implies that the
        #  other getBest* methods pick nodes from the open (unprocessed) frontier
        new_frontier = set()
        for n in chain(leaves, children, siblings):
            # sanity check: SIBLING, LEAF, CHILD
            assert n.getType() != SCIP_NODETYPE.FOCUSNODE
            new_frontier.add(self.create_node(n, overwrite=False))
            # XXX We do not add to the dual pq here, becasue the leaf, child
            #  and sibling nodes appear to have uninitialized default lp values
            self.update_lineage(m, n)

        # if the current set of open nodes is not a subset of the open nodes
        #  upon processing the previous focus node, then SCIP in its solver
        #  loop visited more than one node focus node, before asking for a
        #  branchrule's decision.
        shadow = self.frontier_ - new_frontier
        for j in shadow:
            # the shadow nodes are all nodes processed by SCIP
            #  in between consecutive calls to var branching. Each
            #  could've been PRUNED, or marked as FEASIBLE/INFEASIBLE.
            #  One way or the other they're FATHOMED.
            self.T.nodes[j]["type"] = TracerNodeType.FATHOMED

        self.frontier_ = new_frontier
        return shadow

    def update(self, m: Model) -> None:
        """Update the tracer tree."""

        # finish processing the last focus
        if self.focus_ is not None:
            self.leave(m)

        # start processing the current focus node, unless the search has finished
        if m.getCurrentNode() is not None:
            j = self.enter(m)

            # record the path through the tree
            self.trace_.append(
                (
                    j,
                    m.getPrimalbound(),  # self.T.graph["incumbent"].fun,
                    self.T.nodes[j]["lp"].fun,  # XXX not m.getDualbound()
                )
            )

        else:
            # clear the focus node when SCIP terminates the bnb search
            self.focus_ = None

        # track the best sol maintained by SCIP
        # XXX While processing branching takes place at [SCIPbranchExecLP](solve.c#4420)
        # Therefore, the current best, if it has been updated, should be attributed
        # to a previous focus node or to a node kindly fathomed for us by SCIP,
        #  however. not all solutions get into SCIP's storage
        # XXX [addCurrentSolution](solve.c#5039) the integer-feasible solution
        #  is added after the focus node is processed [solveNode](solve.c#4982).
        #  [primalAddSol](primal.c#1064)
        sols = m.getSols()
        if sols:
            lp = evaluate_sol(m, sols[0], self.sign, transformed=True)
            if self.is_worse(self.T.graph["incumbent"].fun, lp.fun):
                self.T.graph["incumbent"] = lp

        self.prune()

        # attributing the best bound chance to the last focus is not ideal, since
        #  branchrule is not called when SCIP's LP solver detected integer feasibility
        #  or overall infeasibility. A good thing is that such nodes were open in
        #  the past, so we must recompute their fate.
        self.shadow_ = shadow = self.add_frontier(m)
        self.fathomed_.update(shadow)


def subtree_size(T: nx.DiGraph, n: int) -> int:
    """Recursively compute the sizes of all sub-trees."""
    size = 1
    for c in T[n]:
        assert n != c
        size += subtree_size(T, c)

    T.nodes[n]["n_size"] = size
    return size


class NegLogTreeSize:
    """Reward function with bnb tree tracing for Ecole's branching env."""

    tracer: Tracer

    def __init__(self) -> None:
        self.tracer = None

    def before_reset(self, model: ecole_Model) -> None:
        self.tracer = Tracer(model.as_pyscipopt())

    def extract(self, model: ecole_Model, fin: bool) -> ndarray:
        m = model.as_pyscipopt()
        self.tracer.update(m)

        # ecole will indicate `fin=True` while `m.getCurrentNode() is not None`
        #  in the case when a limit is reached (gap, nodes, iter, etc)
        if not fin:
            return None

        T = self.tracer.T

        # the instance was pre-solved if the tracer could not find the root
        if T.graph["root"] is None:
            return np.zeros(0, dtype=np.float32)

        # ensure correct tree size (including shadow-visited nodes)
        subtree_size(T, T.graph["root"])
        n_size = nx.get_node_attributes(T, "n_size")

        # the list of visited nodes ordered according to visitation sequence
        n_visits = nx.get_node_attributes(T, "n_visits")
        visited = [n for n in T if n_visits[n] > 0]

        n_order = nx.get_node_attributes(T, "n_order")
        visited = sorted(visited, key=n_order.get)
        assert all(n_order[n] >= 0 for n in visited)

        # fetch all visited nodes, which bore no children
        # XXX could've checked for `n_size[n] == 1` just as well
        # visited_leaves = [n for n in visited if not T[n]]
        # XXX these are the nodes makred as fathomed by Parsonson

        return -np.log(np.array([n_size[n] for n in visited], dtype=np.float32))


class LPGains:
    """Reward for Ecole's branching env based on dual bound gains."""

    tracer: Tracer

    def __init__(self, gamma: float = 0.25, pseudocost: bool = False) -> None:
        self.gamma, self.pseudocost = gamma, pseudocost
        self.tracer = None

    def before_reset(self, model: ecole_Model) -> None:
        self.tracer = Tracer(model.as_pyscipopt())

    def extract(self, model: ecole_Model, fin: bool) -> ndarray:
        m = model.as_pyscipopt()
        self.tracer.update(m)

        if not fin:
            return None

        # the instance was pre-solved if the tracer could not find the root
        T = self.tracer.T
        if T.graph["root"] is None:
            return np.zeros(0, dtype=np.float32)

        # the list of visited nodes ordered according to visitation sequence
        n_visits = nx.get_node_attributes(T, "n_visits")
        visited = [n for n in T if n_visits[n] > 0]

        n_order = nx.get_node_attributes(T, "n_order")
        visited = sorted(visited, key=n_order.get)
        assert all(n_order[n] >= 0 for n in visited)

        # Get the ell-1 norm of the coefficients of the linear objective
        # XXX we hope that SCIP does not re-scale the sub-problems
        obj = {v.name: v.getObj() for v in m.getVars(transformed=True)}
        scale = max(map(abs, obj.values()))  # sum

        # compute the normalized lp gains on edges between visited nodes
        lps, scores = nx.get_node_attributes(T, "lp"), []
        for u in visited:
            score, k = 1.0, 0
            for v, dt in T[u].items():
                if n_visits[v] < 1 or dt["g"] <= 0.0:
                    continue

                if self.pseudocost:
                    s = dt["g"] / dt["f"]
                    # XXX what is a good natural scale for the pseudocosts?
                    # s = min(max(dt["g"] / dt["f"], 0.0), 1.0)

                else:
                    # \Delta_\pm = c^\top (x_\pm - x) \leq \|c\|_p \|x_\pm - x\|_q
                    # get the ell-infty norm between nested solutions
                    vx, ux = lps[v].x, lps[u].x
                    s = dt["g"] / (scale * sum(abs(vx[k] - ux[k]) for k in obj))

                score *= s
                k += 1

            # compute the geometric mean (`k` is at most 2)
            scores.append(score ** (1 / max(k, 1.0)))  # XXX in [0, 1]
            # XXX after tree tracing we may end up with visited leaves, or visited
            #  nodes with just one child, because SCIP can immediately fathom
            #  children after branching, without ever informing us, due to primal
            #  bound cutoff, infeasibility, or interger-feasibility
            #  - such nodes are good, because they lead to immediate fathoming,
            #    hence they get a +1 reward

        return np.array(scores, dtype=np.float32) ** self.gamma


if __name__ == '__main__':
    import tqdm
    import ecole as ec
    from ecole.environment import Branching

    rew_tracer = LPGains()
    gasse_2019_scip_params = {
        'separating/maxrounds': 0,  # separate (cut) only at root node
        'presolving/maxrestarts': 0,  # disable solver restarts
        'limits/time': 20 * 60,  # solver time limit
        'limits/gap': 3e-3,
        'limits/nodes': 25,
    }

    env = Branching(
        observation_function=ec.observation.StrongBranchingScores(),
        information_function=rew_tracer,
        scip_params=gasse_2019_scip_params
    )

    res = []
    it = ec.instance.CombinatorialAuctionGenerator(n_items=50, n_bids=250)
    # see we throw anything
    for mod, _ in zip(tqdm.tqdm(it, ncols=70), range(100)):
        # mod.disable_presolve()

        n_steps = 0
        obs, act, rew, fin, nfo = env.reset(mod)
        while not fin:
            # use sb scores and branch
            var = act[obs[act].argmax(-1)]
            n_steps += 1

            obs, act, rew, fin, nfo = env.step(var)

        if n_steps > 0:
            res.append(nfo)
