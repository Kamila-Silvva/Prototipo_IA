# Forçar o TensorFlow a usar a CPU
tf.config.set_visible_devices([], 'GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def add_exogenous_variables(data, variables):
    for var in variables:
        if var == 'PIB':
            data['PIB'] = np.random.normal(loc=10000, scale=100, size=len(data))
        elif var == 'Cambio':
            data['Cambio'] = np.random.normal(loc=5, scale=0.1, size=len(data))
    return data

def remove_outliers(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data

def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def build_model(optimizer='adam', neurons=50, dropout_rate=0.2, l2_reg=0.01):
    model = Sequential()
    model.add(InputLayer(shape=(look_back, 6)))
    model.add(LSTM(neurons, return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(neurons, kernel_regularizer=l1_l2(l1=0.01, l2=l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def perform_grid_search(X_train, y_train):
    model = KerasRegressor(model=build_model, verbose=0)
    param_grid = {
        'batch_size': [16, 32],
        'epochs': [50, 100],
        'optimizer': ['adam', 'nadam'],
        'neurons': [50, 100],
        'dropout_rate': [0.2, 0.3],
        'l2_reg': [0.01, 0.02]
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)
    return grid_result.best_params_

def train_and_predict(ticker, variables, look_back):
    today = pd.Timestamp.now()
    start_date = today - pd.DateOffset(years=2)  # Aumentar o período de treinamento para 2 anos
    data = yf.download(ticker, start=start_date, end=today)['Close']
    data = data.dropna()

    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Adicionar variáveis exógenas
    data = add_exogenous_variables(data, variables)

    # Remover valores ausentes e infinidades
    data = data.dropna()
    data = data[~data.isin([np.inf, -np.inf]).any(axis=1)]

    # Remover outliers
    data = remove_outliers(data, 'Close')

    # Decomposição sazonal
    decomposition = seasonal_decompose(data['Close'], model='multiplicative', period=30)
    data['trend'] = decomposition.trend
    data['seasonal'] = decomposition.seasonal
    data['resid'] = decomposition.resid
    data = data.dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = create_dataset(scaled_data, look_back)

    print(f"Data shape: {data.shape}")
    print(f"Scaled data shape: {scaled_data.shape}")
    print(f"X shape before reshape: {X.shape}")
    print(f"y shape: {y.shape}")

    if X.shape[0] == 0 or X.shape[1] == 0 or X.shape[2] == 0:
        raise ValueError("O conjunto de dados criado está vazio ou com formato incorreto. Verifique os dados de entrada e o parâmetro look_back.")

    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

    # Dividindo os dados em conjunto de treino e validação
    tscv = TimeSeriesSplit(n_splits=5)
    train_index, val_index = list(tscv.split(X))[-1]
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Busca em Grade para Hiperparâmetros
    best_params = perform_grid_search(X_train, y_train)
    print(f"Melhores parâmetros encontrados: {best_params}")

    model = build_model(
        optimizer=best_params['optimizer'],
        neurons=best_params['neurons'],
        dropout_rate=best_params['dropout_rate'],
        l2_reg=best_params['l2_reg']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_loss', mode='min')

    print("Iniciando treinamento do modelo...")
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=2,
                        validation_data=(X_val, y_val), callbacks=[early_stopping, checkpoint])
    end_time = time.time()
    print(f"Tempo de treinamento: {end_time - start_time} segundos")

    model.load_weights("best_model.keras")

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros((predictions.shape[0], data.shape[1] - 1)))))

    last_samples = scaled_data[-look_back:]
    last_samples = np.reshape(last_samples, (1, last_samples.shape[0], last_samples.shape[1]))

    num_days = 7  # Prevendo para os próximos 7 dias
    future_predictions = []

    for _ in range(num_days):
        next_prediction = model.predict(last_samples)
        next_prediction_full = np.zeros((1, next_prediction.shape[0], last_samples.shape[2]))
        next_prediction_full[:, :, 0] = next_prediction
        future_predictions.append(next_prediction[0])
        next_input = np.concatenate((last_samples[:, 1:, :], next_prediction_full), axis=1)
        last_samples = next_input

    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(np.column_stack((future_predictions, np.zeros((future_predictions.shape[0], data.shape[1] - 1)))))

    future_dates = pd.date_range(data.index[-1], periods=num_days + 1, freq='D')[1:]

    return data, predictions, future_dates, future_predictions, look_back

if _name_ == "_main_":
    look_back = 10
    ticker = 'PETR4.SA'
    selected_variables = ['PIB', 'Cambio']
    data, predictions, future_dates, future_predictions, look_back = train_and_predict(ticker, selected_variables, look_back)

    plt.figure(figsize=(12, 6))
    plt.plot(data.index[look_back + 1:], data['Close'][look_back + 1:], label='Original PETR4 Data')
    plt.plot(data.index[look_back + 1:], predictions[:, 0], label='Predicted PETR4 Price', linestyle='-')
    plt.plot(future_dates, future_predictions[:, 0], label='Forecasted PETR4 Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()