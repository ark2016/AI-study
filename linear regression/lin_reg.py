class LinearRegression:

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
        """
        Инициализирует модель гиперпараметрами:
        - fit_intercept: учитывать ли свободный член
        - normalize: нормировать ли X перед обучением
        - copy_X: копировать ли входные данные
        """  
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X

    def set_params(self, **params):
        """
        Устанавливает гиперпараметры модели.
        """
        self.fit_intercept = params.get('fit_intercept', self.fit_intercept)
        self.normalize = params.get('normalize', self.normalize)
        self.copy_X = params.get('copy_X', self.copy_X)
        return self

    def get_params(self, deep=True):
        """
        Возвращает текущие гиперпараметры модели.
        """
        return {"fit_intercept": self.fit_intercept, "normalize": self.normalize, "copy_X": self.copy_X}

    def clone(self):
        """
        Возвращает копию модели с теми же параметрами.
        """
        return self.__class__(**self.get_params())

    def fit(self, X, y):
        """
        Обучает модель на данных (X, y). 
        """
        if self.copy_X:
            X = X.copy()
        if self.normalize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        # Вычисляем коэффициенты модели по методу наименьших квадратов
        # Формула: β = (X^T * X)^(-1) * X^T * y
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X):
        """
        Предсказывает целевую переменную для новых данных X.
        """
        if self.copy_X:
            X = X.copy()
        if self.normalize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.coef_

    def partial_fit(self, X, y):
        """
        Онлайн-обучение: обновляет параметры на очередной порции данных.
        """
        if self.copy_X:
            X = X.copy()
        if self.normalize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def score(self, X, y, scoring='r2'):
        """
        Вычисляет метрику качества (по умолчанию R²).
        """
        if scoring == 'r2':
            return r2_score(y, self.predict(X))
        else:
            raise ValueError(f"Unknown scoring method: {scoring}")

    def cross_validate(self, X, y, cv=5, scoring='r2'):
        """
        Выполняет кросс-валидацию модели.
        """
        pass

    def grid_search(self, param_grid, X, y, cv=5, scoring='r2'):
        """
        Подбирает гиперпараметры через GridSearchCV.
        """
        pass

    def compute_loss(self, X, y):
        """
        Вычисляет значение функции потерь (MSE).
        """
        return mean_squared_error(y, self.predict(X))

    def compute_gradient(self, X, y):
        """
        Вычисляет градиент функции потерь для стохастического градиентного спуска.
        """
        pass


    def update_params(self, gradient, lr):
        """
        Обновляет параметры модели по формуле θ := θ - lr * gradient. 
        """
        pass

    def save_model(self, filepath):
        """
        Сохраняет модель на диск (joblib или pickle).
        """
        joblib.dump(self.coef_, filepath)

    def load_model(self, filepath):
        """
        Загружает модель из файла. 
        """
        self.coef_ = joblib.load(filepath)

    def summary(self):
        """
        Выводит краткий отчёт: коэффициенты, intercept, статистику. 
        """
        print("Coef: ", self.coef_)
        print("Intercept: ", self.intercept_)
        print("R2: ", self.score(self.X, self.y))

    def plot_learning_curve(self, X, y, cv=5):
        """
        Строит кривые обучения через sklearn.model_selection.learning_curve.
        Позволяет визуально оценить переобучение/недообучение. 
        """
        predicted = self.predict(X)
        plt.plot(X, y, 'o')
        plt.plot(X, predicted, '-')
        plt.show()
        
