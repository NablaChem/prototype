# %%
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def get_training_data(ndim, npts):
    def testfun(x):
        return np.sin(x.sum(axis=-1))

    X = np.random.uniform(0, 1, size=(npts, ndim))
    y = testfun(X)
    return X, y


# %%
class Kernel:
    @staticmethod
    def _check_shape(arr: np.ndarray, shape: tuple[int]):
        if arr.shape != shape:
            raise ValueError(
                f"Shape mismatch. Please check kernel implementation. Expected {shape}, got {arr.shape}."
            )

        return arr

    @staticmethod
    def _check_args(X: np.ndarray, Xother: np.ndarray, args):
        if len(args) != 0:
            raise ValueError("Function requires explicit keyword-arguments.")

        if Xother is None:
            Xother = X

        return X, Xother

    def value(self, X: np.ndarray, Xother: np.ndarray = None, *args, **kwargs):
        X, Xother = Kernel._check_args(X, Xother, args)

        vals = np.array(self.callable(X, Xother, kwargs))

        return Kernel._check_shape(vals, (X.shape[0], Xother.shape[0]))

    def grad(self, X: np.ndarray, Xother: np.ndarray = None, *args, **kwargs):
        X, Xother = Kernel._check_args(X, Xother, args)

        jac = jax.jacfwd(self.callable, argnums=2)(X, Xother, kwargs)
        jacs = []
        for parameter in self._parameters:
            jacs.append(np.array(jac[parameter]))

        return Kernel._check_shape(
            np.array(jacs), (len(self._parameters), X.shape[0], Xother.shape[0])
        )


class RBF(Kernel):
    _parameters = ["sigma"]

    def callable(self, X, Xother, kwargs):
        distances = jnp.linalg.norm((Xother - X[:, None]), axis=-1)
        return jnp.exp(-(distances**2) / (2 * kwargs["sigma"] ** 2))


def test_gaussian_finite_diff():
    def one_run(kernel, Xtrain, ytrain, Xtest, ytest, sigma, lval):
        npts, _ = Xtrain.shape
        K = kernel.value(Xtrain, Xtrain, sigma=sigma)
        alpha = np.linalg.inv(K.T + lval * np.identity(npts)) @ ytrain

        K = kernel.value(Xtrain, Xtest, sigma=sigma)
        pred = np.sum(K.T * alpha, axis=1)
        return np.average((pred - ytest) ** 2)

    kernel = RBF()
    ndim, npts, sigma, lval = 2, 10, 1e-1, 1e-10

    ntest = 100
    train = get_training_data(ndim, npts)
    test = get_training_data(ndim, ntest)

    # finite diff
    delta = 0.001
    down = one_run(kernel, *train, *test, sigma + delta, lval)
    up = one_run(kernel, *train, *test, sigma - delta, lval)
    deriv = (up - down) / (2 * delta)

    # analytical
    K = kernel.value(train[0], train[0], sigma=sigma)
    beta = np.linalg.inv(K + lval * np.identity(npts))
    alpha = beta @ train[1]

    gradK = kernel.grad(train[0], train[0], sigma=sigma)[0]
    gradbeta = -np.linalg.inv(K) @ gradK @ np.linalg.inv(K)
    K = kernel.value(train[0], test[0], sigma=sigma)
    gradK = kernel.grad(train[0], test[0], sigma=sigma)[0]
    pred = np.sum(K.T * alpha, axis=1)

    gradhat = gradbeta @ train[1] @ K + beta @ train[1] @ gradK
    deriv_analytical = -(2 * (pred - test[1]) * gradhat).sum() / ntest

    assert abs(deriv - deriv_analytical) < 1e-3


test_gaussian_finite_diff()

# %%
