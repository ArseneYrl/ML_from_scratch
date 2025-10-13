import numpy as np
from src.utils.crossentropy_loss import crossentropy_loss
from src.utils.accuracy import accuracy
import time
import os

class Trainer:
    def __init__(self, model, optimizer, seed = 42):
        self.seed = seed
        self.model = model
        self.optimizer = optimizer
    
    def setup_seed(self):
        np.random.seed(self.seed)

    def train_epoch(self, X, y, batch_size, m):
        #What is done during 1 epoch

        indices = np.random.permutation(m)  # Mini-Batch SGD
        
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            y_pred = self.model.feedforward(X_batch)
            

            _, err = crossentropy_loss(y_pred, y_batch)

            self.model.backward(err)

            self.optimizer.step()
            self.optimizer.zero_grad() 

    def train(self, X_train, y_train, X_val = None, y_val = None, epochs=50, batch_size=32, patience = 5):

        self.setup_seed()
        m = len(X_train)

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        start_time = time.time()

        if X_val is not None and y_val is not None:
                y_pred_val = self.model.feedforward(X_val)
                min_val_loss, _ = crossentropy_loss(y_pred_val, y_val)
                patience_count = 0

        for epoch in range(epochs):

            y_pred_train = self.model.feedforward(X_train)

            train_acc = accuracy(y_train, y_pred_train)
            train_loss, _ = crossentropy_loss(y_pred_train, y_train)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss}, Train Accuracy:{train_acc}")

            if X_val is not None and y_val is not None:
                y_pred_val = self.model.feedforward(X_val)

                val_acc = accuracy(y_val, y_pred_val)
                val_loss, _ = crossentropy_loss(y_pred_val, y_val)

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                if epoch % 10 == 0:
                    print(f"Validation Loss: {val_loss}, Validation Accuracy:{val_acc}")

                if val_loss < min_val_loss - 1e-6:  # Amélioration significative
                    min_val_loss = val_loss
                    patience_count = 0
                    best_epoch = epoch

                else: 
                    patience_count +=1
                    if patience_count >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        print(f"Best validation loss: {min_val_loss:.4f} at epoch {best_epoch}")
                        break


            self.train_epoch(X_train, y_train, batch_size, m)

 
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        return history
    
    def fit(self, X_test, y_test):
        y_pred = self.model.feedforward(X_test)
        return accuracy(y_test, y_pred) 

