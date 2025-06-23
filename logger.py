import os
import csv

class Logger:
    def __init__(self, path:str):
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.path = os.path.join(self.path, "logs.csv")

        self.fields = ('''eid
                       time
                       num_actions
                       reward
                       throughput
                       tardiness
                       policy_loss
                       value_loss
                       total_rewards
                       done_lots
                       ''').split('\n')
        
        self.fields = [x.strip() for x in self.fields]
        if not os.path.isfile(self.path):
            self.write()
            
        self.reset_pool()

    def write(self) -> None:
        with open(self.path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.fields)

    def reset_pool(self) -> None:
        self.pool = {k: None for k in self.fields}

    def commit(self) -> None:
        with open(self.path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.pool.values())
        self.reset_pool()

    def add_to_pool(self, **kwargs) -> None:
        for var, val in kwargs.items():
            if var in self.fields:
                self.pool[var] = val
            else:
                raise Exception('Unexpected field ', var)
            