import time


class PID:
    def __init__(
        self,
        kP=1.0,
        kI=0.0,
        kD=0.0,
        output_bound_low=-5.0,
        output_bound_high=5.0,
        integral_bound_low=-2000.0,
        integral_bound_high=2000.0,
    ):
        self.kP = float(kP)
        self.kI = float(kI)
        self.kD = float(kD)
        self.output_bound_low = float(output_bound_low)
        self.output_bound_high = float(output_bound_high)
        self.integral_bound_low = float(integral_bound_low)
        self.integral_bound_high = float(integral_bound_high)
        self.prev_time = time.time()
        self.prev_error = 0.0
        self.prev_output = 0.0
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0

    def reset(self):
        self.prev_time = time.time()
        self.prev_error = 0.0
        self.prev_output = 0.0
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0

    def set_gains(self, kP=None, kI=None, kD=None):
        if kP is not None:
            self.kP = float(kP)
        if kI is not None:
            self.kI = float(kI)
        if kD is not None:
            self.kD = float(kD)

    def set_output_bounds(self, low=None, high=None):
        if low is not None:
            self.output_bound_low = float(low)
        if high is not None:
            self.output_bound_high = float(high)

    def update(self, error, dt=None):
        now = time.time()
        if dt is None:
            dt = now - self.prev_time
        dt = max(float(dt), 1e-4)
        error = float(error)
        delta_error = error - self.prev_error
        self.p_term = error
        self.d_term = delta_error / dt
        output_saturated = self.prev_output <= self.output_bound_low or self.prev_output >= self.output_bound_high
        same_sign = (self.prev_output >= 0.0 and error >= 0.0) or (self.prev_output < 0.0 and error < 0.0)
        if not (output_saturated and same_sign):
            self.i_term += error * dt
            self.i_term = max(self.integral_bound_low, min(self.integral_bound_high, self.i_term))
        adaptive_ki = self.kI / (1.0 + 0.0001 * abs(error))
        output = self.kP * self.p_term + adaptive_ki * self.i_term + self.kD * self.d_term
        output = max(self.output_bound_low, min(self.output_bound_high, output))
        self.prev_time = now
        self.prev_error = error
        self.prev_output = output
        return output
    
