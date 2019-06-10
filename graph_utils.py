import matplotlib.pyplot as plt
import seaborn as sns

def predict(test_y, yhat, i):
  truth = test_y[i:i+15].reshape(15)
  predicted = yhat[i].reshape(5)
  #predicted = test_y[-5:]
  #predicted = test_ys[-1:].reshape(5)


  print(truth[0:10])
  print(truth[10:])
  print(predicted)

  plt.figure(figsize=(10,6))   
  plt.plot(range(0, 11), truth[0:11])
  plt.plot(range(10,15), truth[10:], color='orange')
  plt.plot(range(10,15), predicted, color='teal',linestyle='--')

  plt.legend(['Truth','Target','Predictions'])
  plt.show