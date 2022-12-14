from .obs import NodeBipariteWith24VariableFeatures
from .reward import RetroBranching
import ecole


class EcoleBranching(ecole.environment.Branching):
    def __init__(self, instance_gen, obs_function=None):
        # init default rewards
        reward_function = RetroBranching()

        # reward_function['retro_binary_fathomed'] = RetroBranching()

        information_function = {
            "num_nodes": ecole.reward.NNodes().cumsum(),
            "lp_iterations": ecole.reward.LpIterations().cumsum(),
            "solving_time": ecole.reward.SolvingTime().cumsum(),
            # 'primal_integral': ecole.reward.PrimalIntegral().cumsum(),
            # 'dual_integral': ecole.reward.DualIntegral().cumsum(),
            # 'primal_dual_integral': ecole.reward.PrimalDualIntegral(),
        }

        gasse_2019_scip_params = {
            "separating/maxrounds": 0,  # separate (cut) only at root node
            "presolving/maxrestarts": 0,  # disable solver restarts
            "limits/time": 60 * 60,  # solver time limit
            "timing/clocktype": 1,  # 1: CPU user seconds, 2: wall clock time
            # 'limits/gap': 3e-4,  # 0.03% relative primal-dual gap (default: 0.0)
            # 'limits/nodes': -1,
        }

        if obs_function is None:
            obs_function = NodeBipariteWith24VariableFeatures()

        super(EcoleBranching, self).__init__(
            observation_function=obs_function,
            information_function=information_function,
            reward_function=reward_function,
            scip_params=gasse_2019_scip_params,
            pseudo_candidates=False,
        )

        self.instance_gen = instance_gen

    def reset_basic(self, instance):
        return super().reset(instance)

    def reset(self):
        for instance in self.instance_gen:
            obs, act_set, reward, done, info = super().reset(instance.copy_orig())
            if not done:
                info["instance"] = instance
                return obs, act_set, reward, done, info
        raise StopIteration
