import numpy as np
from torch.optim import RMSprop


class AdaGrad:
	def __init__(self, num_row, num_col, eta=0.1, epsilon=1e-8):
		self.eta = eta
		self.epsilon = epsilon
		self.cache = np.zeros((num_row, num_col))

	def update(self, W, gradient):
		self.cache += gradient ** 2
		W -= self.eta / np.sqrt(self.cache + self.epsilon) * gradient
		return W


class RMSProp:
	def __init__(self, num_row, num_col, lr=0.01, alpha=0.99, epsilon=1e-8):
		self.history = np.zeros((num_row, num_col))
		self.lr = lr
		self.alpha = alpha
		self.epsilon = epsilon

	def update(self, W, gradient):
		self.history = self.alpha * self.history + (1 - self.alpha) * gradient ** 2

		W -= gradient / (np.sqrt(self.history) + self.epsilon)
		return W


class Adam:
	def __init__(self, num_row, num_col, eta=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
		self.M = np.zeros((num_row, num_col), dtype=np.float64)
		self.V = np.zeros((num_row, num_col), dtype=np.float64)
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.eta = eta

	def update(self, t, W, gradient):
		self.M = self.beta_1 * self.M + (1 - self.beta_1) * gradient
		self.V = self.beta_2 * self.V + (1 - self.beta_2) * gradient ** 2

		M_corr = self.M / (1 - self.beta_1 ** t)
		V_corr = self.V / (1 - self.beta_2 ** t)

		W -= self.eta * M_corr / (np.sqrt(V_corr) + self.epsilon)
		return W


class ConstantLr:
	def __init__(self, lr=0.001):
		self.lr = lr

	def update(self, W, gradient):
		return W - self.lr - gradient
