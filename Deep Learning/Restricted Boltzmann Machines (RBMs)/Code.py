import torch
import torch.nn as nn
import numpy as np

# Define the RBM class
class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        
        # Initialize the weight and bias parameters
        self.W = nn.Parameter(torch.randn(visible_units, hidden_units))
        self.v_bias = nn.Parameter(torch.zeros(visible_units))
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))
        
    def sample_hidden(self, visible):
        # Compute the probabilities of the hidden units
        hidden_probs = torch.sigmoid(torch.matmul(visible, self.W) + self.h_bias)
        
        # Sample the hidden units from the probabilities
        hidden = torch.bernoulli(hidden_probs)
        return hidden, hidden_probs
    
    def sample_visible(self, hidden):
        # Compute the probabilities of the visible units
        visible_probs = torch.sigmoid(torch.matmul(hidden, self.W.t()) + self.v_bias)
        
        # Sample the visible units from the probabilities
        visible = torch.bernoulli(visible_probs)
        return visible, visible_probs
    
    def gibbs_sampling(self, visible):
        # Run the Gibbs sampling chain to generate a sample of the hidden units
        hidden, _ = self.sample_hidden(visible)
        visible_rec, _ = self.sample_visible(hidden)
        return visible_rec
        
    def forward(self, visible):
        # Compute the free energy of the visible units
        visible_bias_term = torch.matmul(visible, self.v_bias)
        hidden_term = torch.log(1 + torch.exp(torch.matmul(visible, self.W) + self.h_bias))
        free_energy = -torch.sum(visible_bias_term + hidden_term, dim=1)
        
        # Compute the contrastive divergence algorithm
        k = 1
        for _ in range(k):
            visible_rec = self.gibbs_sampling(visible)
            
        hidden_rec, _ = self.sample_hidden(visible_rec)
        
        positive_grad = torch.matmul(visible.t(), hidden)
        negative_grad = torch.matmul(visible_rec.t(), hidden_rec)
        delta_W = (positive_grad - negative_grad) / visible.shape[0]
        delta_v_bias = torch.mean(visible - visible_rec, dim=0)
        delta_h_bias = torch.mean(hidden - hidden_rec, dim=0)
        
        return free_energy.mean(), delta_W, delta_v_bias, delta_h_bias

# Define the training loop function
def train_rbm(rbm, train_data, learning_rate=0.1, num_epochs=10, batch_size=10):
    optimizer = torch.optim.Adam(rbm.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        num_batches = len(train_data) // batch_size
        
        for i in range(num_batches):
            batch = train_data[i*batch_size : (i+1)*batch_size]
            batch = torch.tensor(batch, dtype=torch.float32)
            optimizer.zero_grad()
            free_energy, delta_W, delta_v_bias, delta_h_bias = rbm(batch)
            loss = free_energy.neg().mean()
            loss.backward()
            optimizer.step()
            
        print("Epoch: {}, Loss: {}".format(epoch+1, loss.item()))
        
# Example usage
visible_units = 784
hidden_units = 100
rbm = RBM(visible_units, hidden_units)
train_data = np.random.binomial(1, 0.5, (1000, visible_units))
train_rbm(rbm, train_data, num_epochs=10)
