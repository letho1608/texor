# Texor AI Library

Texor là một thư viện AI mạnh mẽ kết hợp những tính năng tốt nhất của TensorFlow và PyTorch. Thư viện cung cấp API cao cấp, dễ sử dụng trong khi vẫn duy trì tính linh hoạt và hiệu suất thông qua hệ thống backend lai.

## Tính Năng Chính

### 1. Backend Lai (Hybrid Backend)
- Tận dụng sức mạnh của cả TensorFlow và PyTorch
- Chuyển đổi linh hoạt giữa các backend
- Tối ưu hóa tự động dựa trên use case

### 2. Core API
```python
from texor.core import Tensor

# Tạo tensor từ nhiều nguồn khác nhau
x = Tensor([[1, 2], [3, 4]])  # Từ Python list
x = Tensor(numpy_array)        # Từ NumPy array
x = Tensor(tf_tensor)         # Từ TensorFlow tensor
x = Tensor(torch_tensor)      # Từ PyTorch tensor

# Truy cập dữ liệu ở nhiều định dạng
numpy_data = x.numpy()
tf_data = x.tensorflow()
torch_data = x.pytorch()
```

### 3. Neural Network Layers
```python
from texor.nn import Sequential, Linear, Conv2D, MaxPool2D, ReLU, Dropout

model = Sequential([
    Conv2D(in_channels=1, out_channels=32, kernel_size=3),
    ReLU(),
    MaxPool2D(kernel_size=2),
    Conv2D(in_channels=32, out_channels=64, kernel_size=3),
    ReLU(),
    MaxPool2D(kernel_size=2),
    Linear(in_features=1600, out_features=10)
])
```

### 4. Optimizers
```python
from texor.optim import SGD, Adam, RMSprop

# Tạo optimizer
optimizer = Adam(model.parameters(), lr=0.001)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### 5. Loss Functions
```python
from texor.nn import MSELoss, CrossEntropyLoss, BCELoss

# Sử dụng loss functions
criterion = CrossEntropyLoss()
loss = criterion(predictions, targets)
```

## Cài Đặt

```bash
pip install texor
```

## Command Line Interface (CLI)

texor cung cấp giao diện dòng lệnh (CLI) mạnh mẽ với các tính năng trực quan:

```bash
# Xem thông tin về môi trường và cài đặt
texor info

# Liệt kê các module có sẵn
texor list

# Tìm kiếm module cụ thể
texor list resnet

# Kiểm tra môi trường và dependencies
texor check
```

### Tính năng CLI:

- **Hiển thị màu sắc**: Thông báo, cảnh báo và lỗi được hiển thị với màu sắc rõ ràng
- **Thanh tiến trình**: Hiển thị tiến độ cho các tác vụ dài
- **Giao diện tương tác**: Các lệnh CLI thân thiện với người dùng
- **Thông tin hệ thống**: Hiển thị chi tiết về môi trường và cấu hình

## Ví Dụ Cơ Bản

```python
from texor.nn import Sequential, Linear, ReLU
from texor.core import Tensor
import numpy as np

# Tạo model
model = Sequential([
    Linear(input_size=784, output_size=256),
    ReLU(),
    Linear(input_size=256, output_size=10)
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy'
)

# Tạo dữ liệu mẫu
x = np.random.randn(100, 784)
y = np.random.randint(0, 10, size=(100,))
y = np.eye(10)[y]  # One-hot encode

# Huấn luyện model
model.fit(
    x=Tensor(x),
    y=Tensor(y),
    epochs=10,
    batch_size=32
)
```

## Ví Dụ MNIST
Xem file `examples/mnist_example.py` để thấy ví dụ đầy đủ về việc train một CNN trên MNIST dataset.

## API Documentation

### Core Module
- `Tensor`: Class cơ bản cho các phép toán tensor
- `zeros`, `ones`, `randn`: Các hàm tạo tensor
- `from_numpy`, `from_tensorflow`, `from_pytorch`: Chuyển đổi từ các định dạng khác

### Neural Network (nn) Module
- Layers: `Linear`, `Conv2D`, `MaxPool2D`, `Dropout`
- Activations: `ReLU`, `Sigmoid`, `Tanh`
- Loss Functions: `MSELoss`, `CrossEntropyLoss`, `BCELoss`
- Model: `Sequential` - API dễ sử dụng cho việc xây dựng model

### Optimizers Module
- `SGD`: Stochastic Gradient Descent với momentum
- `Adam`: Adam optimizer
- `RMSprop`: RMSprop optimizer

## Đóng Góp

Chúng tôi luôn chào đón sự đóng góp! Vui lòng tham khảo `CONTRIBUTING.md` để biết thêm chi tiết.

## License

MIT License - xem file `LICENSE` để biết thêm chi tiết.
