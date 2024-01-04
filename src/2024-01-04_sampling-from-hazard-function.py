"""
Sampling from an arbitrary hazard function

Reference: https://gist.github.com/jcrudy/10481743

"""

import numpy
import scipy.integrate
from matplotlib import pyplot
from statsmodels.distributions import ECDF


class HazardSampler(object):
    def __init__(self, hazard, start=0.0, step=None):
        self.hazard = hazard
        if step is None:
            h0 = hazard(0.0)
            if h0 > 0:
                step = 2.0 / hazard(0.0)
            else:
                # Reasonable default.  Not efficient in some cases.
                step = 200.0 / scipy.integrate.quad(hazard, 0.0, 100.0)
        self.cumulative_hazard = CumulativeHazard(hazard)
        self.survival_function = SurvivalFunction(self.cumulative_hazard)
        self.cdf = Cdf(self.survival_function)
        self.inverse_cdf = InverseCdf(self.cdf, start=start, step=step, lower=0.0)
        self.sampler = InversionTransformSampler(self.inverse_cdf)

    def draw(self):
        return self.sampler.draw()


class InversionTransformSampler(object):
    def __init__(self, inverse_cdf):
        self.inverse_cdf = inverse_cdf

    def draw(self):
        u = numpy.random.uniform(0, 1)
        return self.inverse_cdf(u)


class CumulativeHazard(object):
    def __init__(self, hazard):
        self.hazard = hazard

    def __call__(self, t):
        return scipy.integrate.quad(self.hazard, 0.0, t)[0]


class SurvivalFunction(object):
    def __init__(self, cumulative_hazard):
        self.cumulative_hazard = cumulative_hazard

    def __call__(self, t):
        return numpy.exp(-self.cumulative_hazard(t))


class Cdf(object):
    def __init__(self, survival_function):
        self.survival_function = survival_function

    def __call__(self, t):
        return 1.0 - self.survival_function(t)


class InverseCdf(object):
    def __init__(
        self, cdf, start, step, precision=1e-8, lower=float("-inf"), upper=float("inf")
    ):
        self.cdf = cdf
        self.precision = precision
        self.start = start
        self.step = step
        self.lower = lower
        self.upper = upper

    def __call__(self, p):
        last_diff = None
        step = self.step
        current = self.start
        while True:
            value = self.cdf(current)
            diff = value - p
            if abs(diff) < self.precision:
                break
            elif diff < 0:
                current = min(current + step, self.upper)
                if last_diff is not None and last_diff > 0:
                    step *= 0.5
                last_diff = diff
            else:
                current = max(current - step, self.lower)
                if last_diff is not None and last_diff < 0:
                    step *= 0.5
                last_diff = diff
        return current


if __name__ == "__main__":
    # Set a random seed and sample size
    numpy.random.seed(1)
    m = 1000

    # Use this totally crazy hazard function
    def hazard(t):
        return numpy.exp(numpy.sin(t) - 2.0)

    # Sample failure times from the hazard function
    sampler = HazardSampler(hazard)
    failure_times = numpy.array([sampler.draw() for _ in range(m)])

    # Apply some non-informative right censoring, just to demonstrate how it's done
    censor_times = numpy.random.uniform(0.0, 25.0, size=m)
    y = numpy.minimum(failure_times, censor_times)
    c = 1.0 * (censor_times > failure_times)

    # Make some plots of the simulated data
    # Plot a histogram of failure times from this hazard function
    pyplot.hist(failure_times, bins=50)
    pyplot.title("Uncensored Failure Times")
    pyplot.savefig("uncensored_hist.png")
    pyplot.show()

    # Plot a histogram of censored failure times from this hazard function
    pyplot.hist(y, bins=50)
    pyplot.title("Non-informatively Right Censored Failure Times")
    pyplot.savefig("censored_hist.png")
    pyplot.show()

    # Plot the empirical survival function (based on the censored sample) against the
    # actual survival function
    t = numpy.arange(0, 20.0, 0.1)
    S = numpy.array([sampler.survival_function(t[i]) for i in range(len(t))])
    S_hat = 1.0 - ECDF(failure_times)(t)
    pyplot.figure()
    pyplot.title("Survival Function Comparison")
    pyplot.plot(t, S, "r", lw=3, label="True survival function")
    pyplot.plot(t, S_hat, "b--", lw=3, label="Sampled survival function (1 - ECDF)")
    pyplot.legend()
    pyplot.xlabel("Time")
    pyplot.ylabel("Proportion Still Alive")
    pyplot.savefig("survival_comp.png")
    pyplot.show()

    print("done")
