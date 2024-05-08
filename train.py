import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import shuffle_data,plot_results,ToVariable,use_cuda
from sklearn.metrics import mean_squared_error

def train_sfm(model,x_train,y_train,epochs=10):
    optimizer = optim.Adam(model.parameters())
    # Loss function: Mean Squared Error
    criterion = nn.MSELoss()

    # Shuffle data to prevent order bias
    X,Y = shuffle_data(x_train,y_train)

    x_len = X.shape[1]

    # Convert data to torch Variable
    X = ToVariable(X)
    Y = ToVariable(Y)

    for epoch in range(0,epochs):
        # Initialize model state
        h,c,re_s,im_s,time = model.init_state()
        for step in range(0,x_len):
            # Select input at current step
            x = X[:,step,:]
            y = Y[:,step,:]

            # Adjust dimensions for model input
            x = x.unsqueeze(1)

            # Model forward pass
            output,h,c,re_s,im_s,time = model(x,h,c,re_s,im_s,time)

            # Detach state variables to prevent backprop through entire sequence
            h = h.data
            c = c.data
            re_s = re_s.data
            im_s = im_s.data
            time = time.data

            # Compute loss
            loss = criterion(output.squeeze(0), y)

            # Clear gradients
            optimizer.zero_grad()
            # Backpropagate error
            loss.backward(retain_graph=True)

            # Update model weights
            optimizer.step()
            print('Epoch: ', epoch+1, '| step: ', step+1, '| Loss: ',loss.detach())

    torch.save(model, 'model/model2.pkl') # Save the trained model

def test_sfm(model,x_test,y_test,opt,denoise):
    # Load the trained model
    model = torch.load('model/model2.pkl')

    # Set model to evaluation mode
    model.eval()
    pred_dat = []

    # Initialize model state
    h,c,re_s,im_s,time = model.init_state()
    # Get the sequence length of test data
    seq_len = x_test.shape[1]

    for i in range(0,seq_len):
        x = ToVariable(x_test[:,i,:]) # Convert test data to Variable
        x = x.view(-1,1,1) # Adjust dimensions for model input

        # Model forward pass
        pre_out,h,c,re_s,im_s,time =  model(x,h,c,re_s,im_s,time)
        # Detach state variables
        h = h.data
        c = c.data
        re_s = re_s.data
        im_s = im_s.data
        time = time.data
        if use_cuda:
            pred_dat.append(pre_out.data.cpu().numpy())
        else:
            pred_dat.append(pre_out.data.numpy())

    pred_dat=np.array(pred_dat)
    pred_dat = pred_dat.transpose(1,0,2)

    # Scale back the data
    pred_dat = (pred_dat[:,:, 0] * (opt.max_data - opt.min_data) + (opt.max_data + opt.min_data))/2
    y_test = (y_test[:,:, 0] * (opt.max_data - opt.min_data) + (opt.max_data + opt.min_data))/2

    # error = np.sum((pred_dat[:,-opt.test_len:] - y_test[:,-opt.test_len:])**2) / (opt.test_len* pred_dat.shape[0])
    if denoise:
        error = mean_squared_error(pred_dat[-1, -opt.test_len+1:] , y_test[6, -opt.test_len+1:])
    else:
        error = mean_squared_error(pred_dat[6, -opt.test_len+1:] , y_test[6, -opt.test_len+1:])
    variance_y = np.var(y_test[6, -opt.test_len + 1:])
    relative_mse = error / variance_y
    print('The mean square error is: %f' % error)
    print('The relative mean square error is: %f' % relative_mse)
    # Plotting the Prediction Results
    if denoise:
        plot_results(pred_dat[-1, -opt.test_len+1:], y_test[6, -opt.test_len+1:])
    else:
        plot_results(pred_dat[6,-opt.test_len+1:],y_test[6,-opt.test_len+1:])