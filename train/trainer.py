from tqdm import tqdm
import torch
import os
from utils.helpers import aleatoric_loss, compute_weight_decay, load_config, deep_supervision_loss, ensure_dir, enable_dropout, compute_smoothness_loss, normalize_for_vis
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from eval.eval import evaluate_model


config = load_config()
deep_supervision = config.model.deep_supervision
use_smooth_loss = config.train.use_smooth_loss
smoothness_weight = config.train.smooth_loss_weight
patience_limit = config.train.early_stopping.patience
mc_dropout = config.eval.mc_dropout

# Initialize TensorBoard writer
log_dir=os.path.join(config.logging.tensorboard_dir, "depth_run")
ensure_dir(log_dir)
writer = SummaryWriter(log_dir=log_dir)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, results_dir, mc_training=False):
    """Train the model and save the best based on validation metrics"""
    best_val_loss = float('inf')
    best_epoch = 0
    best_sirmse = float('inf')
    train_losses = []
    val_losses = []

    global_step = 0
    patience_counter = 0    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0

        for i, (inputs, targets, _) in enumerate(tqdm(train_loader, desc="Training")):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            if deep_supervision:
                outputs, aux_outputs = model(inputs)
                loss = deep_supervision_loss(outputs, aux_outputs, targets, criterion, weights=[0.25, 0.25])
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if mc_training:
                loss += compute_weight_decay(model, config.model.dropout_p, targets.size(0))

            if use_smooth_loss:
                smoothness_loss = compute_smoothness_loss(outputs[:, 0:1, :, :], inputs)
                loss += smoothness_weight * smoothness_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            # Log training loss to TensorBoard
            writer.add_scalar("Loss/Train", loss.item(), global_step)
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar("LearningRate", current_lr, epoch)


            # Log images every 50 steps
            if global_step % 50 == 0:
                writer.add_images("Train/Input", inputs, global_step)
                writer.add_images("Train/Target", normalize_for_vis(targets), global_step)
                
                if isinstance(outputs, tuple):  # deep supervision case
                    writer.add_images("Train/Prediction", normalize_for_vis(outputs[0][:, 0:1]), global_step)
                else:
                    writer.add_images("Train/Prediction", normalize_for_vis(outputs[:, 0:1]), global_step)


            global_step += 1

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)               
                loss = 0

                if mc_training:
                    enable_dropout(model)
                    loss += compute_weight_decay(model, config.model.dropout_p, targets.size(0))

                if deep_supervision:
                    outputs, _ = model(inputs)
                else:
                    outputs = model(inputs)

                loss += criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()


        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Log validation loss to TensorBoard
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        # Save the best model
        if val_loss < best_val_loss:
            patience_counter = 0
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))
            print(f"New best model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
        # Early stopping logic
        else:
            print(f"No improvement in validation loss at epoch {epoch+1}. Best so far: {best_val_loss:.4f} at epoch {best_epoch+1}")
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience_limit}")
            if patience_counter >= patience_limit:
                print(f"\nEarly stopping triggered. Best model was from epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
                writer.close()
                model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))
                return model

        if (epoch+1) % 5 == 0:
            # strongly evaluate the model every 5 epochs
            print("Evaluating model on validation set...")
            metrics = evaluate_model(model, val_loader, device, results_dir, mc_predictions=mc_dropout)
            print("\nValidation Metrics:")
            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")
            # Save evaluation results to TensorBoard
            writer.add_scalar("Metrics/MAE", metrics['MAE'], epoch)
            writer.add_scalar("Metrics/RMSE", metrics['RMSE'], epoch)
            writer.add_scalar("Metrics/Rel", metrics['REL'], epoch)
            writer.add_scalar("Metrics/Delta1", metrics['Delta1'], epoch)
            writer.add_scalar("Metrics/Delta2", metrics['Delta2'], epoch)
            writer.add_scalar("Metrics/Delta3", metrics['Delta3'], epoch)
            writer.add_scalar("Metrics/siRMSE", metrics['siRMSE'], epoch)

            # Save metrics to file
            path = os.path.join(results_dir, f"metrics_epoch_{epoch+1}.txt")
            with open(path, 'w') as f:
                for name, value in metrics.items():
                    f.write(f"{name}: {value:.4f}\n")

            # Save best model based on SIRMSE
            sirmse = metrics["siRMSE"]
            if sirmse < best_sirmse:
                patience_counter = 0
                best_sirmse = sirmse
                torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_sirmse.pth'))
                print(f"New best sirmse-model saved at epoch {epoch+1} with sirmse: {sirmse:.4f}")

            # Save model checkpoint every 5 epochs
            checkpoint_path = os.path.join(results_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")


    print(f"\nBest model was from epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
    writer.close()

    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))

    return model
