import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Вспомогательная функция для вычисления B-сплайнов
# Это и есть та самая "готовая реализация", которую мы напишем сами.
# Она вычисляет значения базисных функций B-сплайна.
# ----------------------------------------------------------------------------

def b_spline_basis(x, grid, k):
    """
    Вычисляет значения базисных функций B-сплайна.
    
    Args:
        x (torch.Tensor): Входной тензор. Ожидаемая форма (batch_size, 1).
        grid (torch.Tensor): Тензор с узлами сетки.
        k (int): Порядок сплайна.
        
    Returns:
        torch.Tensor: Значения базисных функций.
    """

    # Начинаем с B-сплайнов 0-го порядка (кусочно-постоянные функции)
    # x имеет форму (batch_size, 1), grid[:-1] - (N,). Результат будет (batch_size, N)
    B = ((x >= grid[:-1]) & (x < grid[1:])).float()
    
    # Рекурсивно вычисляем B-сплайны более высоких порядков
    for order in range(1, k + 1):
        # Вычисляем левую часть рекуррентной формулы
        term1_num = x - grid[:-order-1]
        term1_den = grid[order:-1] - grid[:-order-1]
        term1_den[term1_den == 0] = 1e-8
        term1 = (term1_num / term1_den) * B[:, :-1]
        
        # Вычисляем правую часть рекуррентной формулы
        term2_num = grid[order+1:] - x
        term2_den = grid[order+1:] - grid[1:-order]
        term2_den[term2_den == 0] = 1e-8
        term2 = (term2_num / term2_den) * B[:, 1:]
        
        B = term1 + term2
        
    return B

# ----------------------------------------------------------------------------
# УРОВЕНЬ 1: "КИРПИЧИК" - SplineActivation
# Становится полноценным модулем PyTorch
# ----------------------------------------------------------------------------
class SplineActivation(nn.Module):
    def __init__(self, grid_size=5, spline_order=3):
        super().__init__()
        self.spline_order = spline_order
        
        h = (2.0) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h - 1.0
        self.register_buffer('grid', grid)
        
        self.spline_coeffs = nn.Parameter(torch.randn(1, len(self.grid) - spline_order - 1) * 0.1)
        self.w_base = nn.Parameter(torch.randn(1))
        # В статье w_s инициализируется как 1, но в официальной реализации он тоже случаен.
        # Оставим его случайным для большей стабильности в начале обучения.
        self.w_spline = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # Убедимся, что x имеет форму (batch_size, 1)
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
            
        # 1. Базовая активация SiLU
        base_activation_val = F.silu(x)
        
        # 2. Значение сплайна
        spline_basis_vals = b_spline_basis(x, self.grid, self.spline_order)
        spline_val = F.linear(spline_basis_vals, self.spline_coeffs)
        
        # 3. Комбинируем и возвращаем результат
        # Сначала складываем тензоры формы [batch, 1], а потом сжимаем до [batch]
        combined = self.w_base * base_activation_val + self.w_spline * spline_val
        return combined.squeeze(-1) # squeeze(-1) "сжимает" последнюю размерность
    
# ----------------------------------------------------------------------------
# УРОВЕНЬ 2: СЛОЙ KAN
# ----------------------------------------------------------------------------
class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, grid_size=5, spline_order=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Вместо обычного списка Python используем nn.ModuleList.
        # Это КРИТИЧЕСКИ важно, чтобы PyTorch видел параметры внутри SplineActivation.
        self.activations = nn.ModuleList([
            SplineActivation(grid_size, spline_order) for _ in range(in_dim * out_dim)
        ])

    def forward(self, x):
        # x имеет форму (batch_size, in_dim)
        
        # Подготавливаем выходной тензор
        output = torch.zeros(x.shape[0], self.out_dim, device=x.device)
        
        for j in range(self.out_dim):
            for i in range(self.in_dim):
                # Получаем соответствующую активацию
                activation_func = self.activations[j * self.in_dim + i]
                
                # Применяем ее к i-му входу
                phi_ji_x_i = activation_func(x[:, i])
                
                # Суммируем на j-м выходном нейроне
                output[:, j] += phi_ji_x_i
                
        return output

# ----------------------------------------------------------------------------
# УРОВЕНЬ 3: ВСЯ МОДЕЛЬ KAN
# ----------------------------------------------------------------------------
class KAN(nn.Module):
    def __init__(self, shape, grid_size=5, spline_order=3):
        super().__init__()
        
        # Опять используем nn.ModuleList
        self.layers = nn.ModuleList()
        
        for i in range(len(shape) - 1):
            in_dim = shape[i]
            out_dim = shape[i+1]
            self.layers.append(KANLayer(in_dim, out_dim, grid_size, spline_order))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) # nn.Module вызываются как функции
        return x