from trainer import Trainer
from model import VAE

def main():
    trainer = Trainer()
    vae = VAE()
    trainer.train(vae)

if __name__ == "__main__":
    print(0)
    main()
