class Hyperparameters:
    def __init__(self):
        self.batch_size = 64
        self.channel_feature_size = 32
        self.channel_out_size = 1
        self.ckpt_path = './ckpt/'
        self.data_path = './data/'
        self.data_name = 'MNIST'
        self.hidden_layer_size = 256
        self.learning_rate = 0.001        
        self.num_epochs = 10
        self.seed = 0
        