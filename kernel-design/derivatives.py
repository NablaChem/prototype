# %%
import numpy as np
import jax
import jax.numpy as jnp


def testfun(x):
    return np.sin(x.sum(axis=-1))


def get_training_data(ndim, npts):
    X = np.random.uniform(0, 1, size=(npts, ndim))
    y = testfun(X)
    return X, y


def gaussian_kernel(X, Xprime, sigma):
    return np.exp(-np.linalg.norm((X - Xprime[:, None]), axis=-1) ** 2 / (2 * sigma**2))


def grad_gaussian_kernel(X, Xprime, sigma):
    return jnp.exp(
        -jnp.linalg.norm((X - Xprime[:, None]), axis=-1) ** 2 / (2 * sigma**2)
    )


def one_run(Xtrain, ytrain, Xtest, ytest, sigma, lval):
    npts, _ = Xtrain.shape
    K = gaussian_kernel(Xtrain, Xtrain, sigma)
    alpha = np.linalg.inv(K + lval * np.identity(npts)) @ ytrain

    K = gaussian_kernel(Xtrain, Xtest, sigma)
    pred = np.sum(K * alpha, axis=1)
    return np.average((pred - ytest) ** 2)


def value_and_grad(ndim, npts, sigma, lval):
    ntest = 100
    train = get_training_data(ndim, npts)
    test = get_training_data(ndim, ntest)

    # finite diff
    delta = 0.001
    down = one_run(*train, *test, sigma + delta, lval)
    up = one_run(*train, *test, sigma - delta, lval)
    deriv = (up - down) / (2 * delta)

    # analytical
    K = gaussian_kernel(train[0], train[0], sigma)
    beta = np.linalg.inv(K + lval * np.identity(npts))
    alpha = beta @ train[1]

    gradK = np.array(
        jax.jacfwd(grad_gaussian_kernel, argnums=2)(train[0], train[0], sigma)
    )
    gradbeta = -np.linalg.inv(K) @ gradK @ np.linalg.inv(K)
    K = gaussian_kernel(train[0], test[0], sigma)
    gradK = np.array(
        jax.jacfwd(grad_gaussian_kernel, argnums=2)(train[0], test[0], sigma)
    )
    pred = np.sum(K * alpha, axis=1)

    gradhat = gradbeta @ train[1] @ K.T + beta @ train[1] @ gradK.T
    deriv_analytical = -(2 * (pred - test[1]) * gradhat).sum() / ntest

    return up, deriv, deriv_analytical


value_and_grad(2, 10, 1e-1, 1e-10)

# %%
