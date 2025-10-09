import numpy as np
from src.utils.crossentropy_loss import crossentropy_loss

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