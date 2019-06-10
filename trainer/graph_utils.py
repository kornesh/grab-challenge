import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def prediction_graph(Xs, ys, yhat, i, past_steps, future_steps):
    past = Xs[i][:, -1].reshape(past_steps)[1:]
    for c in Xs[i]:
        print(c)

    print("past", past)
    truth = ys[i].reshape(future_steps)
    past = np.append(past, truth[0])
    predicted = yhat[i].reshape(future_steps)
    #predicted = test_y[-5:]
    #predicted = ys[-1:].reshape(5)


    #print("past",past.shape, past)
    #print("truth", truth.shape, truth)
    #print("predicted", predicted.shape, predicted)

    plt.figure(figsize=(10,6))   
    plt.plot(range(0, past_steps), past)
    plt.plot(range(past_steps - 1, past_steps + future_steps - 1), truth, color='orange')
    plt.plot(range(past_steps - 1, past_steps + future_steps - 1), predicted, color='teal', linestyle='--')

    plt.legend(['Truth','Target','Predictions'])
    plt.show()
