import numpy as np
from src.utils.crossentropy_loss import crossentropy_loss
from src.utils.accuracy import accuracy
import time
import os

class Trainer:
    def __init__(self, seed = 42):
        self.seed = seed
    
    def setup_seed(self):
        np.random.seed(self.seed)

    def train_epoch(self, model, optimizer, X, y, batch_size, m):

        indices = np.random.permutation(m)  # Mini-Batch SGD
        
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            y_pred = model.feedforward(X_batch)
            

            _, err = crossentropy_loss(y_pred, y_batch)

            model.backward(err)

            optimizer.step()
            optimizer.zero_grad() 

    def train(self, model, optimizer, X_train, y_train, X_val = None, y_val = None, epochs=50, batch_size=32):
        #self.setup_seed()
        m = len(X_train)
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        start_time = time.time()
        for epoch in range(epochs):

            y_pred_train = model.feedforward(X_train)

            train_acc = accuracy(y_train, y_pred_train)
            train_loss, _ = crossentropy_loss(y_pred_train, y_train)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss}, Train Accuracy:{train_acc}")

            if X_val is not None and y_val is not None:
                y_pred_val = model.feedforward(X_val)

                val_acc = accuracy(y_val, y_pred_val)
                val_loss, _ = crossentropy_loss(y_pred_val, y_val)

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

            self.train_epoch(model, optimizer, X_train, y_train, batch_size, m)

 
        total_time = time.time() - start_time
        print("Time:",total_time)

        return history
    


def train_mlp(model, optimizer, X, y , epochs, batch_size):
    loss = []
    m = len(X)
    y_initial = model.feedforward(X)
    initial_loss, _ = crossentropy_loss(y_initial,y)
    loss.append(initial_loss)
    print("Initial loss:", loss[0])
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        indices = np.random.permutation(m)  # Mini-Batch SGD
        
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            y_new = model.feedforward(X_batch)

            batch_loss, err = crossentropy_loss(y_new,y_batch)
            epoch_loss += batch_loss
            num_batches += 1

            model.backward(err)

            optimizer.step()
            optimizer.zero_grad() 

        avg_loss = epoch_loss / num_batches
        loss.append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss}")

    return loss

