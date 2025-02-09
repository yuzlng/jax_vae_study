class Trainer:
    def __init__(self):
        """
        Initialize parameters
        """
        
    def loss_fn(self):
        '''
        ELBO loss function
        '''
        return NotImplementedError
    
    def optimizer(self):
        return NotImplementedError
    
    def logging(self):
        """
        Log training process
        Use tensor board
        """
        return NotImplementedError
    
    def model_save(self):
        """
        Save the model
        """
        return NotImplementedError