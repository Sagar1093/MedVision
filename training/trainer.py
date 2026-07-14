import torch
from tqdm import tqdm
from torch.amp import autocast,GradScaler
from configs.config import *

def train_one_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
        scaler,
        use_amp
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for images,labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type,enabled=use_amp):

            outputs = model(images)

            loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        running_loss += loss.item()
        _,predicted = torch.max(outputs,1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        progress_bar.set_postfix(
            loss = loss.item()
        )
    epoch_loss = running_loss/len(dataloader)
    epoch_accuracy = 100*correct/total
    return epoch_loss,epoch_accuracy


def validate_one_epoch(
        model,
        dataloader,
        criterion,
        device,
        use_amp
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc = "Validation",
            leave = False
            )
        for images,labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            with autocast(device_type=device.type,enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs,labels)

            running_loss += loss.item()

            _,predicted = torch.max(outputs,1)

            correct += (predicted == labels).sum().item()
            
            total += labels.size(0)
            
            progress_bar.set_postfix(
                loss = loss.item()
            )

    epoch_loss = running_loss/len(dataloader)
    epoch_accuracy = 100* correct/total
    return epoch_loss,epoch_accuracy

def fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs,
        checkpoint_path
):
    history = {
        "train_loss":[],
        "train_acc":[],
        "val_loss":[],
        "val_acc":[],
        "learning_rate":[]
    }

    use_amp = device.type == "cuda"
    
    best_val_acc = 0.0
    scaler = GradScaler("cuda",enabled=use_amp)
    early_stop_counter = 0

    for epoch in range(epochs):
        print(f"Epoch:{epoch+1}/{epochs}\n")

        train_loss,train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp
            )
        val_loss,val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_amp
            )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.2f}%")
        scheduler.step(val_acc)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            early_stop_counter = 0

            torch.save(
                 {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "history":history
                 },
                checkpoint_path
            )
        else:
            early_stop_counter += 1
            
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print("Early Stopping Triggered")
                break
    return history