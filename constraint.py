from enum import Enum


class TDBMetric(Enum):
    ERROR, FEEDBACK, RUNTIME = range(3)


class TDBConstraint:
    def __init__(self, constraint):
        # ('error', 0.1, 1)
        if len(constraint) != 3:
            raise ValueError(f'Invalid constraint: {constraint}')
        
        if constraint[0] == 'error':
            self.metric = TDBMetric.ERROR
        elif constraint[0] == 'feedback':
            self.metric = TDBMetric.FEEDBACK
        elif constraint[0] == 'runtime':
            self.metric = TDBMetric.RUNTIME
        else:
            raise ValueError(f'Invalid constraint metric: {constraint[0]}')
        self.threshold = constraint[1]
        self.weight = constraint[2]

    def __repr__(self):
        return f'({self.metric.name.lower()}, {self.threshold}, {self.weight})'

    def check_continue(self, error, nr_feedbacks, runtime, nr_nl_filters):
        if self.metric == TDBMetric.ERROR:
            return error > self.threshold
        elif self.metric == TDBMetric.FEEDBACK:
            return nr_feedbacks < self.threshold * nr_nl_filters and error > 0
        elif self.metric == TDBMetric.RUNTIME:
            return runtime < self.threshold and error > 0
        else:
            raise ValueError(f'Invalid constraint metric: {self.metric}')