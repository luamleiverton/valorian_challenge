Class

@jit(target_backend ="cuda")
def compute_models(model, x_train, x_test, y_train, y_test):
  inicio = time.time()
  # Initialize the model
  temp = model
  model_fitted = temp.fit(x_train, y_train)
  # Fit the model
  print('Model fitted:', model_fitted)
  # Make predictions
  y_pred = model_fitted.predict(x_test)
  y_pred = np.round(y_pred)
  return y_pred