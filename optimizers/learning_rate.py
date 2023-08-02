class StepLR:
    
    def __init__(self, initial_lr, step_size, gamma):
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
        
    def set_learning_rate(self, opt, epoch):
        opt.lr = self.initial_lr * self.gamma**(epoch/self.step_size)
        
    def get_lr(self, epoch):
        return self.initial_lr * self.gamma**(epoch/self.step_size)