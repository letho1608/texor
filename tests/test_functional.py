"""Tests for texor.nn.functional module"""
import pytest
import numpy as np
from texor.core import Tensor
from texor.nn import functional as F


class TestActivations:
    """Test activation functions"""
    
    def test_relu(self):
        x = Tensor(np.array([-1, 0, 1, 2]))
        result = F.relu(x)
        expected = np.array([0, 0, 1, 2])
        assert np.allclose(result.data, expected)
    
    def test_relu_inplace_error(self):
        x = Tensor(np.array([-1, 0, 1, 2]))
        with pytest.raises(NotImplementedError):
            F.relu(x, inplace=True)
    
    def test_sigmoid(self):
        x = Tensor(np.array([0, 1, -1]))
        result = F.sigmoid(x)
        expected = 1 / (1 + np.exp(-np.array([0, 1, -1])))
        assert np.allclose(result.data, expected)
    
    def test_tanh(self):
        x = Tensor(np.array([0, 1, -1]))
        result = F.tanh(x)
        expected = np.tanh(np.array([0, 1, -1]))
        assert np.allclose(result.data, expected)
    
    def test_leaky_relu(self):
        x = Tensor(np.array([-1, 0, 1]))
        result = F.leaky_relu(x, negative_slope=0.01)
        expected = np.where(x.data > 0, x.data, 0.01 * x.data)
        assert np.allclose(result.data, expected)
    
    def test_elu(self):
        x = Tensor(np.array([-1, 0, 1]))
        result = F.elu(x, alpha=1.0)
        expected = np.where(x.data > 0, x.data, np.exp(x.data) - 1)
        assert np.allclose(result.data, expected)
    
    def test_gelu_exact(self):
        x = Tensor(np.array([0, 1, -1]))
        result = F.gelu(x, approximate='none')
        from scipy.special import erf
        expected = 0.5 * x.data * (1 + erf(x.data / np.sqrt(2)))
        assert np.allclose(result.data, expected)
    
    def test_gelu_tanh(self):
        x = Tensor(np.array([0, 1, -1]))
        result = F.gelu(x, approximate='tanh')
        expected = 0.5 * x.data * (1 + np.tanh(np.sqrt(2 / np.pi) * (x.data + 0.044715 * x.data**3)))
        assert np.allclose(result.data, expected)
    
    def test_softmax(self):
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        result = F.softmax(x, dim=1)
        assert np.allclose(result.data.sum(axis=1), 1.0)
    
    def test_softplus(self):
        x = Tensor(np.array([0, 1, -1]))
        result = F.softplus(x)
        expected = np.log(1 + np.exp(x.data))
        assert np.allclose(result.data, expected)
    
    def test_mish(self):
        x = Tensor(np.array([0, 1, -1]))
        result = F.mish(x)
        expected = x.data * np.tanh(np.log(1 + np.exp(x.data)))
        assert np.allclose(result.data, expected)


class TestDropout:
    """Test dropout functions"""
    
    def test_dropout_inference(self):
        x = Tensor(np.ones(100))
        result = F.dropout(x, p=0.5, training=False)
        assert np.allclose(result.data, x.data)
    
    def test_dropout_zero_p(self):
        x = Tensor(np.ones(100))
        result = F.dropout(x, p=0, training=True)
        assert np.allclose(result.data, x.data)


class TestNormalization:
    """Test normalization functions"""
    
    def test_layer_norm_1d(self):
        x = Tensor(np.random.randn(10, 20))
        result = F.layer_norm(x, (20,))
        assert result.data.shape == (10, 20)
        assert np.abs(result.data.mean(axis=1)).max() < 0.1
    
    def test_layer_norm_2d(self):
        x = Tensor(np.random.randn(10, 20, 30))
        result = F.layer_norm(x, (20, 30))
        assert result.data.shape == (10, 20, 30)
    
    def test_layer_norm_with_weight_bias(self):
        x = Tensor(np.random.randn(10, 20))
        weight = Tensor(np.ones(20))
        bias = Tensor(np.zeros(20))
        result = F.layer_norm(x, (20,), weight, bias)
        assert result.data.shape == (10, 20)
    
    def test_batch_norm_4d(self):
        x = Tensor(np.random.randn(2, 3, 4, 4))
        running_mean = Tensor(np.zeros(3))
        running_var = Tensor(np.ones(3))
        weight = Tensor(np.ones(3))
        bias = Tensor(np.zeros(3))
        result = F.batch_norm(x, running_mean, running_var, weight, bias, training=True)
        assert result.data.shape == (2, 3, 4, 4)
    
    def test_batch_norm_inference(self):
        x = Tensor(np.random.randn(2, 3, 4, 4))
        running_mean = Tensor(np.random.randn(3))
        running_var = Tensor(np.abs(np.random.randn(3)) + 0.1)
        weight = Tensor(np.ones(3))
        bias = Tensor(np.zeros(3))
        result = F.batch_norm(x, running_mean, running_var, weight, bias, training=False)
        assert result.data.shape == (2, 3, 4, 4)
    
    def test_instance_norm_4d(self):
        x = Tensor(np.random.randn(2, 3, 4, 4))
        result = F.instance_norm(x)
        assert result.data.shape == (2, 3, 4, 4)
        # Each channel should have mean close to 0
        for c in range(3):
            channel_mean = result.data[:, c, :, :].mean()
            assert np.abs(channel_mean) < 0.1
    
    def test_instance_norm_2d(self):
        x = Tensor(np.random.randn(2, 3))
        result = F.instance_norm(x)
        assert result.data.shape == (2, 3)
    
    def test_group_norm_4d(self):
        x = Tensor(np.random.randn(2, 4, 4, 4))
        result = F.group_norm(x, num_groups=2)
        assert result.data.shape == (2, 4, 4, 4)
    
    def test_group_norm_2d(self):
        x = Tensor(np.random.randn(2, 4))
        result = F.group_norm(x, num_groups=2)
        assert result.data.shape == (2, 4)


class TestLinearOperations:
    """Test linear operations"""
    
    def test_linear(self):
        x = Tensor(np.random.randn(10, 20))
        weight = Tensor(np.random.randn(30, 20))
        bias = Tensor(np.random.randn(30))
        result = F.linear(x, weight, bias)
        expected = x.data @ weight.data.T + bias.data
        assert np.allclose(result.data, expected)
    
    def test_linear_no_bias(self):
        x = Tensor(np.random.randn(10, 20))
        weight = Tensor(np.random.randn(30, 20))
        result = F.linear(x, weight)
        expected = x.data @ weight.data.T
        assert np.allclose(result.data, expected)


class TestConvolution:
    """Test convolution operations"""
    
    def test_conv1d_stride(self):
        x = Tensor(np.random.randn(1, 1, 10))
        weight = Tensor(np.random.randn(1, 1, 3))
        result = F.conv1d(x, weight, stride=2, padding=0)
        expected_length = (10 - 3) // 2 + 1
        assert result.data.shape[2] == expected_length
    
    def test_conv2d(self):
        x = Tensor(np.random.randn(2, 3, 10, 10))
        weight = Tensor(np.random.randn(4, 3, 3, 3))
        bias = Tensor(np.random.randn(4))
        result = F.conv2d(x, weight, bias, padding=1)
        assert result.data.shape == (2, 4, 10, 10)


class TestPooling:
    """Test pooling operations"""
    
    def test_avg_pool2d(self):
        x = Tensor(np.random.randn(2, 3, 4, 4))
        result = F.avg_pool2d(x, kernel_size=2, stride=2)
        assert result.data.shape == (2, 3, 2, 2)
    
    def test_max_pool2d(self):
        x = Tensor(np.random.randn(2, 3, 4, 4))
        result = F.max_pool2d(x, kernel_size=2, stride=2)
        assert result.data.shape == (2, 3, 2, 2)


class TestUtility:
    """Test utility functions"""
    
    def test_softmax_stable(self):
        x = Tensor(np.array([[1, 2, 3]]))
        result = F.softmax(x, dim=1)
        assert np.allclose(result.data.sum(axis=1), 1.0)
        # Values should be in valid probability range
        assert result.data.max() <= 1.0
        assert result.data.min() >= 0.0