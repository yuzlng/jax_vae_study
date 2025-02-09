import einops
import jax
from flax import linen as nn
import jax.numpy as jnp
import jax.random as random
from hparam import Hyperparameters 

hps = Hyperparameters()

class Encoder(nn.Module):
    """ 
    input -> latent space 
    """
    features: int = hps.channel_feature_size
    training: bool = True
    latent_dim: int = hps.hidden_layer_size

    @nn.compact
    def __call__(self, x):
        z1 = nn.Conv(self.features, kernel_size=(3, 3))(x)
        z1 = nn.relu(z1)
        z1 = nn.Conv(self.features, kernel_size=(3, 3))(z1)
        z1 = nn.BatchNorm(use_running_average=not self.training)(z1)
        z1 = nn.relu(z1)
        z1_pool = nn.max_pool(z1, window_shape=(2, 2), strides=(2, 2))
        
        z2 = nn.Conv(self.features * 2, kernel_size=(3, 3))(z1_pool)
        z2 = nn.relu(z2)
        z2 = nn.Conv(self.features * 2, kernel_size=(3, 3))(z2)
        z2 = nn.BatchNorm(use_running_average=not self.training)(z2)
        z2 = nn.relu(z2)
        z2_pool = nn.max_pool(z2, window_shape=(2, 2), strides=(2, 2))
		
        z3 = nn.Conv(self.features * 4, kernel_size=(3, 3))(z2_pool)
        z3 = nn.relu(z3)
        z3 = nn.Conv(self.features * 4, kernel_size=(3, 3))(z3)
        z3 = nn.BatchNorm(use_running_average=not self.training)(z3)
        z3 = nn.relu(z3)
        z = einops.rearrange(z3, 'b c h w -> b (c h w)')

        # Latent space
        mean_z = nn.Dense(features=self.latent_dim)(z)
        logvar_z = nn.Dense(features=self.latent_dim)(z)

        return mean_z, logvar_z, z3.shape   # z3.shape : decoder가 복원에 사용할 정보 


class Decoder(nn.Module):
    """ 
    latent space -> input 
    """
    features: int = hps.channel_feature_size
    training: bool = True
    channel_out_size: int = hps.channel_out_size
    
    @nn.compact
    def __call__(self, z, *z_shape):
        z_up = nn.Dense(z_shape[1] * z_shape[2]* z_shape[3])(z)
        z3 = einops.rearrange(z_up, 'b (c h w) -> b c h w', c=z_shape[1], h=z_shape[2], w=z_shape[3])
        z3 = nn.ConvTranspose(self.features * 4, kernel_size=(3, 3), strides=2)(z3)
        z3 = nn.relu(z3)
        z3 = nn.Conv(self.features * 4, kernel_size=(3, 3))(z3)
        z3 = nn.BatchNorm(use_running_average=not self.training)(z3)
        z3 = nn.relu(z3)
        
        z4 = nn.ConvTranspose(self.features * 2, kernel_size=(3, 3), strides=2)(z3)
        z4 = nn.relu(z4)
        z4 = nn.Conv(self.features * 2, kernel_size=(3, 3))(z4)
        z4 = nn.BatchNorm(use_running_average=not self.training)(z4)
        z4 = nn.relu(z4)

        z5 = nn.ConvTranspose(self.features, kernel_size=(3, 3))(z4)
        z5 = nn.relu(z5)
        z5 = nn.Conv(self.channel_out_size, kernel_size=(3, 3))(z5)
        z5 = nn.BatchNorm(use_running_average=not self.training)(z5)
        x = nn.tanh(z5)

        return x


#TODO: HOMEWORK!!!!!
class VAE(nn.Module):
    """
    - Encoder, Decoder 함수를 기반으로 VAE class 완성
    - DataLoader 로부터 batch data 를 입력으로 받아서 VAE 출력이 가능하도록 만들기
    """
    features: int = hps.channel_feature_size
    latent_dim: int = hps.hidden_layer_size
    channel_out_size: int = hps.channel_out_size
    key: int = random.PRNGKey(hps.seed)

    def rng_generator(self):
        self.key, subkey = random.split(self.key)
        return subkey

    def setup(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    ## reparametrize
    def reparametrize(self, mean, logvar):
        std = jnp.exp(0.5 * logvar)
        epsilon = random.normal(self.rng_generator(), mean.shape)
        z = mean + epsilon * std
        return z

    def __call__(self, x):
        # 1. Amortized Inference (Encoder)
        mean, logvar, z_shape = self.encoder(x)

        # 2. Pathwise Estimator (Reparameterization Trick)
        z = self.reparametrize(mean, logvar)

        # 3. Generator (Decoder)
        x_recon = self.decoder(z, *z_shape)

        return x_recon, mean, logvar

if __name__ == "__main__":
    key = random.PRNGKey(0)
    key, rng = random.split(key, 2)
    x = jax.random.normal(rng, (hps.batch_size, 28, 28, hps.channel_out_size))

    encoder = Encoder()
    decoder = Decoder()
    encoder_params = encoder.init(rng, x)['params']
    encoder_out = encoder.apply({'params': encoder_params}, x)
    decoder_params = decoder.init(rng, encoder_out[0], *encoder_out[2])['params']
    decoder_out = decoder.apply({'params': decoder_params}, encoder_out[0], *encoder_out[2])

    if decoder_out.shape == x.shape:
        print('VAE is working')
    else:
        print('VAE is not working')
