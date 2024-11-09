import pandas as pd

class OptimizationTracker:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.iteration_count = 0
        self.lcoe_values = []

    def callback(self, xk, lcoe):
        self.iteration_count += 1
        self.lcoe_values.append((self.iteration_count, lcoe))

    def save_to_csv(self):
        df = pd.DataFrame(self.lcoe_values, columns=['Iteration', 'LCOE'])
        df.to_csv(self.csv_file_path, index=False)

    def reset(self):
        self.iteration_count = 0
        self.lcoe_values = []