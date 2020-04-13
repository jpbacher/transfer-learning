import time
import numpy as np
import pandas as pd
import torch


def train(model,
          loss_fn,
          optimizer,
          train_loader,
          val_loader,
          file_name_save,
          n_epochs=10,
          patience=3):

    device = torch.device('cpu')
    epochs = 0
    epochs_patience = 0
    val_loss_min = np.Inf
    val_acc_max = 0
    history = []
    start = time.time()

    for epoch in range(1, n_epochs +1):
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0
        val_acc = 0
        model.train()
        start = time.time()
        for batch, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            top_p, preds = torch.max(output, dim=1)
            correct = preds.eq(target.data.view_as(preds))
            accuracy = torch.mean(correct.type(torch.FloatTensor))
            train_acc += accuracy.item() * inputs.size(0)
        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_acc / len(train_loader.sampler)
        epochs += 1

        with torch.no_grad():
            model.eval()
            for inputs, target in val_loader:
                output = model.forward(inputs)
                loss = loss_fn(output, target)
                val_loss += loss.item() * inputs.size(0)
                top_p, preds = torch.max(output, dim=1)
                correct = preds.eq(target.data.view_as(preds))
                accuracy = torch.mean(
                    correct.type(torch.FloatTensor))
                val_acc += accuracy.item() * inputs.size(0)
            val_loss = val_loss / len(val_loader.sampler)
            val_acc = val_acc / len(val_loader.sampler)
            history.append([train_loss, val_loss, train_acc, val_acc])
            print(f'[Epoch: {epoch}/{n_epochs} train loss {train_loss:.6f}'
                  f' train acc {train_acc:.3f} val loss {val_loss:.6f}'
                  f' val acc {val_acc:.3f}]')
            if val_loss < val_loss_min:
                torch.save(model.state_dict(), file_name_save)
                epochs_patience = 0
                val_loss_min = val_loss
                val_acc_max = val_acc
                best_epoch = epoch
            else:
                epochs_patience += 1
                if epochs_patience >= patience:
                    print(
                        f'Early stopping - total epochs: {epoch} - best epoch: {best_epoch}'
                        f' min loss: {val_loss_min:.6f} - best acc: {val_acc_max:.3f}'
                    )
                    model.load_state_dict(torch.load(file_name_save))
                    model.optimizer = optimizer
                    history = pd.DataFrame(
                        history,
                        columns=['train_loss', 'val_loss', 'train_acc', 'val_acc']
                    )
                    return model, history

    model.optimizer = optimizer
    total_time = (time.time() - start) / 60
    print(f'best epoch: {best_epoch} - loss:{val_loss_min:.6f} - acc: {val_acc_max:.3f}')
    print(f'total training time: {total_time:.2f} minutes')
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'val_loss', 'train_acc', 'val_acc'])

    return model, history
