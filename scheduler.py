class StairScheduler:
    def __init__(self, start_val, decrease, interval, min_val):
        self.n_steps = 0
        self.value = start_val
        self.interval = interval
        self.linear = LinearScheduler(start_val=start_val, decrease=decrease, min_val=min_val)

    def step(self):
        self.n_steps += 1
        if self.n_steps % self.interval == 0:
            self.linear.step()

    def val(self):
        return self.linear.val()


class CustomScheduler:
    def __init__(self, vals, steps):
        self.vals = vals
        self.steps = steps

        self.val_idx =  0
        assert steps[0] == 0
        self.n_steps = 0

    def step(self):
        self.n_steps += 1
        if self.val_idx < len(self.steps) - 1 and self.steps[self.val_idx + 1] == self.n_steps:
            self.val_idx += 1

    def val(self):
        return self.vals[self.val_idx]


class LinearScheduler:
    def __init__(self, start_val, decrease, min_val):
        self.value = start_val
        self.decrease = decrease
        self.min_val = min_val

    def step(self):
        self.value = max(self.min_val, self.value - self.decrease)
        print(self.value)

    def val(self):
        return self.value
