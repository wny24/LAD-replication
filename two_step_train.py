import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from tqdm import tqdm
from robot_clip.model import RobotCLIP
from robot_clip.data_loading import get_dataloaders
from robot_clip.utils import create_optimizer, save_model, get_temperature
import os
from hydra.utils import to_absolute_path

def train_encoders(model, optimizer, train_loader, config, device, current_epoch):
    model.train()
    epoch_losses = {}
    
    # Get current temperature for this epoch
    current_temp = get_temperature(config, current_epoch)

    for batch in tqdm(train_loader, desc="Training Encoders"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        embeddings = model.encode(batch)
        # Pass current temperature to contrastive loss
        contrastive_loss = model.contrastive_loss(embeddings, temperature=current_temp)
        loss = config.model.contrastive_loss_weight * contrastive_loss
        loss.backward()
        optimizer.step()

        if "contrastive_loss" not in epoch_losses:
            epoch_losses["contrastive_loss"] = 0
        epoch_losses["contrastive_loss"] += contrastive_loss.item()
        # Track current temperature
        if "temperature" not in epoch_losses:
            epoch_losses["temperature"] = current_temp

    for k in epoch_losses:
        if k != "temperature":  # Don't average the temperature
            epoch_losses[k] /= len(train_loader)

    return epoch_losses

def train_decoders(model, optimizer, train_loader, config, device):
    model.train()
    epoch_losses = {}

    for batch in tqdm(train_loader, desc="Training Decoders"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        with torch.no_grad():
            embeddings = model.encode(batch)
        reconstructions = model.decode(embeddings)
        reconstruction_losses = model.reconstruction_loss(batch, reconstructions)
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

def evaluate(model, test_loader, config, device, current_epoch):
    model.eval()
    epoch_losses = {}
    
    # Get current temperature for evaluation
    current_temp = get_temperature(config, current_epoch)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            embeddings = model.encode(batch)
            contrastive_loss = model.contrastive_loss(embeddings, temperature=current_temp)
            reconstructions = model.decode(embeddings)
            reconstruction_losses = model.reconstruction_loss(batch, reconstructions)

            if "contrastive_loss" not in epoch_losses:
                epoch_losses["contrastive_loss"] = 0
            epoch_losses["contrastive_loss"] += contrastive_loss.item()

            for k, v in reconstruction_losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0
                epoch_losses[k] += v.item()

    for k in epoch_losses:
        epoch_losses[k] /= len(test_loader)
    epoch_losses["temperature"] = current_temp

    return epoch_losses

@hydra.main(config_path="config", config_name="two_step_config", version_base="1.1")
def train(config: DictConfig):
    debug_mode = config.training.debug

    # Use to_absolute_path for save_path
    save_path = to_absolute_path(config.training.save_path)
    os.makedirs(save_path, exist_ok=True)

    wandb_config = OmegaConf.to_container(config, resolve=True)
    wandb.init(project=config.wandb.project,
                entity=config.wandb.entity, 
                config=wandb_config, 
                mode="disabled" if debug_mode else None,
                dir=save_path)

    device = torch.device(f"cuda:{config.training.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, normalization_params = get_dataloaders(config)

    model = RobotCLIP(config).to(device)
    encoder_optimizer = create_optimizer(config, model.encoders.parameters(), config.optimizer.encoder)
    decoder_optimizer = create_optimizer(config, model.decoders.parameters(), config.optimizer.decoder)

    # Step 1: Train Encoders
    print("Training Encoders")
    for epoch in range(config.training.encoder_epochs):
        train_losses = train_encoders(model, encoder_optimizer, train_loader, config, device, epoch)
        test_losses = evaluate(model, test_loader, config, device, epoch)

        print(f"Encoder Epoch {epoch+1}/{config.training.encoder_epochs}")
        print("Train Losses:", {k: f"{v:.4f}" for k, v in train_losses.items()})
        print("Test Losses:", {k: f"{v:.4f}" for k, v in test_losses.items()})
        print(f"Current Temperature: {train_losses['temperature']:.6f}")

        if not debug_mode:
            wandb.log({
                "encoder/epoch": epoch,
                "encoder/temperature": train_losses["temperature"],
                **{f"encoder/train/{k}": v for k, v in train_losses.items() if k != "temperature"},
                **{f"encoder/test/{k}": v for k, v in test_losses.items() if k != "temperature"}
            })

        if (epoch + 1) % config.training.save_interval == 0:
            save_model(model, (encoder_optimizer, decoder_optimizer), epoch + 1, config, wandb.run.name, normalization_params, is_two_step=True)

    # Step 2: Train Decoders
    print("Training Decoders")
    for param in model.encoders.parameters():
        param.requires_grad = False

    for epoch in range(config.training.decoder_epochs):
        train_losses = train_decoders(model, decoder_optimizer, train_loader, config, device)
        test_losses = evaluate(model, test_loader, config, device, epoch)

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
            save_model(model, (encoder_optimizer, decoder_optimizer), config.training.encoder_epochs + epoch + 1, config, wandb.run.name, normalization_params, is_two_step=True)

    if not debug_mode:
        wandb.finish()

if __name__ == "__main__":
    train()
