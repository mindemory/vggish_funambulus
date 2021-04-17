def mahalanobis_distance(x, df):
  mu = np.mean(df)
  difference = x - mu
  cov = np.cov(df.values.T)
  if np.linalg.det(cov) == 0:
    inv_cov = np.linalg.pinv(cov)
  else:
    inv_cov = np.linalg.inv(cov)
  mahal_matrix = np.dot(np.dot(difference, inv_cov), difference.T)
  #fig = plt.figure()
  #plt.hist(mahal_matrix.diagonal(), bins = 500)
  return mahal_matrix.diagonal()
  

def classifier(x_train, x_test, y_train, categories):
  distance_df = pd.DataFrame(columns = categories)
  probability_df = pd.DataFrame(columns = categories)
  
  for cat in categories:
    globals()['x_train_' + cat] = x_train.loc[y_train == cat, :]
    distance_df[cat] = mahalanobis_distance(x_test, globals()['x_train_' + cat])

  probability_df = 1 - distance_df.div(distance_df.sum(axis = 1), axis = 0)
  predictions_df = probability_df.idxmax(axis = 1)
  return probability_df, predictions_df