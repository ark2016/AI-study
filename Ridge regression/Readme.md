## Гребневая регрессия (Ridge)

### 1. Понятное описание

Ridge-регрессия вводит штраф за величину коэффициентов, сглаживая оценку и уменьшая разброс при сильно коррелированных признаках ([en.wikipedia.org](https://en.wikipedia.org/wiki/Ridge_regression?utm_source=chatgpt.com "Ridge regression - Wikipedia")).  
Она предпочтительна, когда требуется стабильная оценка, хотя и приводит к смещению параметров (bias–variance tradeoff) ([mathworks.com](https://www.mathworks.com/discovery/regularization.html?utm_source=chatgpt.com "Regularization - MATLAB & Simulink - MathWorks")).

### 2. Математическая формулировка

Пусть всё как в линейной регрессии. Вводим гиперпараметр λ≥0\lambda\ge0.

### 3. Функция потерь
$$
J(\theta)=\frac{1}{2m}\|X\theta - y\|_2^2 + \frac{\lambda}{2m}\|\theta\|_2^2.
$$
Градиент:
$$
\nabla_\theta J(\theta) =\frac{1}{m}X^\top(X\theta - y) + \frac{\lambda}{m}\theta.
$$
Приравняв к нулю, получаем нормальное уравнение
$$
(X^\top X + \lambda I)\theta = X^\top y.
$$
**Доказательство:** аналогично линейной регрессии с добавлением производной по $\theta$ от $\|\theta\|_2^2$ ([mathworks.com](https://www.mathworks.com/matlabcentral/cody/problems/44734?utm_source=chatgpt.com "Solve the 2-norm Regularization Problem - MATLAB Cody"), [mathworks.com](https://www.mathworks.com/matlabcentral/answers/410642-how-to-reduce-the-large-condition-number-of-matrix?utm_source=chatgpt.com "How to reduce the large condition number of matrix - MathWorks")).

### 4. Правило предсказания

- **Закрытое выражение:**
    

$\hat\theta = (X^\top X + \lambda I)^{-1}X^\top y.$

- **Псевдокод (градиентный спуск):**
    

```plaintext
Вход: X, y, λ, шаг α, T
Инициализация: θ=0
Для t=1…T:
    θ ← θ - α * [ (1/m)Xᵀ(Xθ - y) + (λ/m)θ ]
Возврат θ
```

- Реализация в scikit-learn: `sklearn.linear_model.Ridge(alpha=λ)` ([scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?utm_source=chatgpt.com "Ridge — scikit-learn 1.7.0 documentation"), [scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ridge_regression.html?utm_source=chatgpt.com "ridge_regression — scikit-learn 1.7.0 documentation")).
    

### 5. Пример вычислений

Пусть те же данные, добавим λ=1:

$X=\begin{pmatrix}1\\2\end{pmatrix},\;y=\begin{pmatrix}2\\3\end{pmatrix},\;\lambda=1.$

Тогда

$\hat\theta =\bigl(X^\top X + \lambda I\bigr)^{-1}X^\top y =\bigl(5+1\bigr)^{-1}\cdot8=\tfrac{8}{6}\approx1.333.$
Предсказания: 1.333 и 2.667.

### 6. Практические советы и варианты

- **Выбор λ:** кросс-валидация (Grid Search или Random Search).
    
- **Нормировка:** масштабирование признаков обязательно.
    
- **Мультивыход:** поддерживается в `Ridge` для многомерного y ([scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?utm_source=chatgpt.com "Ridge — scikit-learn 1.7.0 documentation")).
    
