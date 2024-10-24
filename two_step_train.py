import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import wandb
from tqdm import tqdm
from robot_clip.model import RobotCLIP
import os

def load_and_normalize_data(config: DictConfig):
    data = np.load(config.data.source_file, allow_pickle=True).item()
    normalized_data = {}
    normalization_params = {}

    for modality, tensor in data.items():
        mean = np.mean(tensor, axis=0)
        std = np.std(tensor, axis=0)
        normalized_tensor = (tensor - mean) / (std + 1e-8)
        normalized_data[modality] = torch.tensor(normalized_tensor, dtype=torch.float32)
        normalization_params[modality] = {'mean': mean, 'std': std}

    return normalized_data, normalization_params

def create_optimizer(config: DictConfig, model_params, optimizer_config):
    optimizer_name = optimizer_config.name
    optimizer_params = {k: v for k, v in optimizer_config.items() if k != "name"}
    
    if optimizer_name == "Adam":
        return torch.optim.Adam(model_params, **optimizer_params)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(model_params, **optimizer_params)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(model_params, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def train_encoders(model, optimizer, train_loader, config, device):
    model.train()
    epoch_losses = {}

    for batch in tqdm(train_loader, desc="Training Encoders"):
        batch = [t.to(device) for t in batch]
        batch_dict = {
            "mano": batch[0],
            "faive": batch[1],
            "simple_gripper": batch[2]
        }
        optimizer.zero_grad()
        embeddings = model.encode(batch_dict)
        contrastive_loss = model.contrastive_loss(embeddings)
        loss = config.model.contrastive_loss_weight * contrastive_loss
        loss.backward()
        optimizer.step()

        if "contrastive_loss" not in epoch_losses:
            epoch_losses["contrastive_loss"] = 0
        epoch_losses["contrastive_loss"] += contrastive_loss.item()

    for k in epoch_losses:
        epoch_losses[k] /= len(train_loader)

    return epoch_losses

def train_decoders(model, optimizer, train_loader, config, device):
    model.train()
    epoch_losses = {}

    for batch in tqdm(train_loader, desc="Training Decoders"):
        batch = [t.to(device) for t in batch]
        batch_dict = {
            "mano": batch[0],
            "faive": batch[1],
            "simple_gripper": batch[2]
        }
        optimizer.zero_grad()
        with torch.no_grad():
            embeddings = model.encode(batch_dict)
        reconstructions = model.decode(embeddings)
        reconstruction_losses = model.reconstruction_loss(batch_dict, reconstructions)
        loss = sum(reconstruction_losses.values())
        loss.backward()
        optimizer.step()

        for k, v in reconstruction_losses.items():
            if k not in epoch_losses:
                epoch_losses[k] = 0
            epoch_losses[k] += v.item()

    for k in epoch_losses:
        epoch_losses[k] /= len(train_loader)

    return epoch_losses

def evaluate(model, test_loader, config, device):
    model.eval()
    epoch_losses = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = [t.to(device) for t in batch]
            batch_dict = {
                "mano": batch[0],
                "faive": batch[1],
                "simple_gripper": batch[2]
            }
            embeddings = model.encode(batch_dict)
            contrastive_loss = model.contrastive_loss(embeddings)
            reconstructions = model.decode(embeddings)
            reconstruction_losses = model.reconstruction_loss(batch_dict, reconstructions)

            if "contrastive_loss" not in epoch_losses:
                epoch_losses["contrastive_loss"] = 0
            epoch_losses["contrastive_loss"] += contrastive_loss.item()

            for k, v in reconstruction_losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0
                epoch_losses[k] += v.item()

    for k in epoch_losses:
        epoch_losses[k] /= len(test_loader)

    return epoch_losses

def save_model(model, encoder_optimizer, decoder_optimizer, epoch, config, run_name, normalization_params):
    save_dir = os.path.join(config.training.save_path, run_name)
    print(f"Saving model to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
        'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
        'config': OmegaConf.to_container(config, resolve=True),
        'normalization_params': normalization_params
    }
    torch.save(checkpoint, os.path.join(save_dir, f'model_epoch_{epoch}.pth'))

@hydra.main(config_path="config", config_name="two_step_config", version_base="1.1")
def train(config: DictConfig):
    debug_mode = config.training.debug

    os.makedirs(config.training.save_path, exist_ok=True)

    wandb_config = OmegaConf.to_container(config, resolve=True)
    wandb.init(project=config.wandb.project,
                entity=config.wandb.entity, 
                config=wandb_config, 
                mode="disabled" if debug_mode else None,
                dir=config.training.save_path)

    device = torch.device(f"cuda:{config.training.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data, normalization_params = load_and_normalize_data(config)
    
    dataset = TensorDataset(*[tensor for tensor in data.values()])
    
    train_size = int(config.data.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)

    model = RobotCLIP(config).to(device)
    encoder_optimizer = create_optimizer(config, model.encoders.parameters(), config.optimizer.encoder)
    decoder_optimizer = create_optimizer(config, model.decoders.parameters(), config.optimizer.decoder)

    # Step 1: Train Encoders
    print("Training Encoders")
    for epoch in range(config.training.encoder_epochs):
        train_losses = train_encoders(model, encoder_optimizer, train_loader, config, device)
        test_losses = evaluate(model, test_loader, config, device)

        print(f"Encoder Epoch {epoch+1}/{config.training.encoder_epochs}")
        print("Train Losses:", {k: f"{v:.4f}" for k, v in train_losses.items()})
        print("Test Losses:", {k: f"{v:.4f}" for k, v in test_losses.items()})

        if not debug_mode:
            wandb.log({
                "encoder/epoch": epoch,
                **{f"encoder/train/{k}": v for k, v in train_losses.items()},
                **{f"encoder/test/{k}": v for k, v in test_losses.items()}
            })

        if (epoch + 1) % config.training.save_interval == 0:
            save_model(model, encoder_optimizer, decoder_optimizer, epoch + 1, config, wandb.run.name, normalization_params)

    # Step 2: Train Decoders
    print("Training Decoders")
    for param in model.encoders.parameters():
        param.requires_grad = False

    for epoch in range(config.training.decoder_epochs):
        train_losses = train_decoders(model, decoder_optimizer, train_loader, config, device)
        test_losses = evaluate(model, test_loader, config, device)

        print(f"Decoder Epoch {epoch+1}/{config.training.decoder_epochs}")
        print("Train Losses:", {k: f"{v:.4f}" for k, v in train_losses.items()})
        print("Test Losses:", {k: f"{v:.4f}" for k, v in test_losses.items()})

        if not debug_mode:
            wandb.log({
                "decoder/epoch": epoch,
                **{f"decoder/train/{k}": v for k, v in train_losses.items()},
                **{f"decoder/test/{k}": v for k, v in test_losses.items()}
            })

        if (epoch + 1) % config.training.save_interval == 0:
            save_model(model, encoder_optimizer, decoder_optimizer, config.training.encoder_epochs + epoch + 1, config, wandb.run.name, normalization_params)

    if not debug_mode:
        wandb.finish()

if __name__ == "__main__":
    train()
