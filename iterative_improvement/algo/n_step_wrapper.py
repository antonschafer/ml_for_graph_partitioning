class NStepWrapper:
    def __init__(
        self,
        algo,
        r_steps,
    ):
        assert algo.use_progress or r_steps == 1, "Not implemented"

        self.algo = algo
        self.r_steps = r_steps
        self.stats_info = []  # stats: predictions, rewards, cut decrease normalized, next qs
        self.observations = []  # state action reward

        self.best_cut_weight = self.algo.best_cut_weight
        self.last_algo_weights = []

        for i in range(r_steps):
            self._run_step()

        self.steps = 0

        self.static_rep_timestamp = 0

    def _run_step(self):
        if not self.algo.done():
            self.algo.step()
            self.last_algo_weights.append(self.algo.best_cut_weight)
            self.stats_info.append(self.algo.get_x_r_cdn_nextq())
            self.observations.append(self.algo.get_state_action_reward())
        else:
            self.stats_info.append((None, 0, 0, 0))
            empty_ob = {
                "next_n2p": None,
                "next_add_in": None,
                "next_q": 0,
                "last_step": True,
                "reward": 0,
            }
            self.observations.append(empty_ob)

    def _current_cum_reward(self):
        assert len(self.stats_info) == self.r_steps
        return sum(stats[1] for stats in self.stats_info)

    def get_x_r_cdn_nextq(self):
        assert self.steps > 0, "Perform step first"

        cdn = self.stats_info[0][2]
        pred = self.stats_info[0][0]

        next_q = self.stats_info[-1][3]

        return pred, self._current_cum_reward(), cdn, next_q

    def get_state_action_reward(self):
        assert self.steps > 0, "Perform step first"
        assert len(self.observations) == self.r_steps

        res_newest = self.observations[-1]
        res_oldest = self.observations[0]

        from_oldest = ["node_2_partition", "swap", "explore", "additional_in", "G", "params_static"]
        from_newest = ["next_n2p", "next_add_in", "next_q", "last_step"]

        return {
            **{k: res_oldest[k] for k in from_oldest},
            **{k: res_newest[k] for k in from_newest if k in res_newest},
            "reward": self._current_cum_reward(),
        }

    def set_static_rep_dirty(self):
        self.algo.set_static_rep_dirty()
        self.static_rep_timestamp = self.steps

    def step(self):
        assert not self.done()
        if self.steps != 0:  # first step already computed
            self.stats_info.pop(0)
            self.observations.pop(0)
            self._run_step()
        self.steps += 1

        self.best_cut_weight = self.last_algo_weights.pop(0)
        self.static_rep_timestamp = self.algo.static_rep_timestamp + self.steps - self.algo.steps  #

    def done(self):
        return self.algo.done() and self.steps == self.algo.steps

