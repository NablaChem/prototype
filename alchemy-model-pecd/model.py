#%%
BASEDIR = "/home/ferchault/wrk/prototype/alchemy-model-pecd"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import findiff
import math
import functools
import scipy.interpolate as sci
from sklearn.kernel_ridge import KernelRidge


def get_data(kind: str, reduced: bool = False) -> pd.DataFrame:
    """
    Loads an origin data file and returns it as a pandas DataFrame.

    Parameters
    ----------
    kind : str
        The kind of data to load. This should correspond to the basename of a text file
        in the directory specified by the BASEDIR global variable.
    reduced : bool, optional
        Whether to return a reduced version of the DataFrame, by dropping its first
        and last columns. Default is False.

    Returns
    -------
    df : pandas.DataFrame
        The loaded data as a pandas DataFrame. The DataFrame's columns correspond to
        the numerical values extracted from the header of the input text file.
        The DataFrame's rows correspond to the data points contained in the file.
        If `reduced` is True, the DataFrame has its first and last columns removed.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    df = pd.read_csv(f"{BASEDIR}/{kind}.txt", index_col=0, delim_whitespace=True)
    df.columns = [float(_.split("=")[1]) for _ in df.columns]
    if reduced:
        df = df.iloc[:, 1:-1]
    return df


# compare to KRR/GPR
# uncertainty from distance in dq/dr
# discuss error as function of displacement

# %%
def central_finite_difference(displacement: float, values: np.ndarray) -> np.poly1d:
    """
    Computes a polynomial approximation of the derivative of a function
    using central finite differences.

    Parameters
    ----------
    displacement : float
        The spacing between adjacent values in the `values` array.
    values : numpy.ndarray
        An array of function values at equidistant points.

    Returns
    -------
    p : numpy.poly1d
        A polynomial function that approximates the derivative of the input function
        using central finite differences. The polynomial's coefficients are determined
        by fitting it to the input data. The polynomial's degree is determined by the
        number of derivative orders that can be computed using the input values.

    Raises
    ------
    AssertionError
        If the length of the input `values` array is not odd.
    """
    assert len(values) % 2 == 1
    order = 0
    pcoeffs = [values[len(values) // 2]]
    while True:
        order += 1
        for acc in (4, 2):
            coeffs = findiff.coefficients(deriv=order, acc=acc)["center"][
                "coefficients"
            ]
            if len(coeffs) == len(values):
                break
        else:
            break
        pcoeffs.append(
            np.sum(coeffs * values) / displacement**order / math.factorial(order)
        )
    return np.poly1d(pcoeffs[::-1])


def alchemical_pecd(
    energy: float, dr: float, dq: float, reduced: bool = False
) -> float:
    """
    Computes the percentage change in PECD resulting from
    a simultaneous displacement and charge perturbation of a molecule, using the
    alchemical perturbation approach.

    Parameters
    ----------
    energy : float
        The photoelectron energy, in eV.
    dr : float
        The displacement distance of the perturbed molecule, in a.u..
    dq : float
        The charge change of the perturbed molecule, in elementary charges.
    reduced : bool, optional
        Whether to use a reduced version of the input data, by dropping its first
        and last columns. Default is False.

    Returns
    -------
    pecd : float
        The percentage change in electrostatic potential energy resulting from the
        specified displacement and charge perturbation, as predicted by the alchemical
        perturbation method.
    """
    line = get_data("charge", reduced=reduced).loc[energy].to_list()
    taylor_charge = central_finite_difference(0.1, np.array(line))
    line = get_data("position", reduced=reduced).loc[energy].to_list()
    taylor_pos = central_finite_difference(0.1, np.array(line))
    return (
        line[len(line) // 2]
        + taylor_charge(dq)
        - taylor_charge(0)
        + taylor_pos(dr)
        - taylor_pos(0)
    ) * 100


#%%
def estimated_uncertainty(energy: float, dr: float, dq: float) -> float:
    """
    Estimates the uncertainty in the percentage change of electrostatic potential energy
    resulting from a simultaneous displacement and charge perturbation of a molecule,
    using the alchemical perturbation approach.

    Parameters
    ----------
    energy : float
        The photoelectron energy, in eV.
    dr : float
        The displacement distance of the perturbed molecule, in a.u..
    dq : float
        The charge change of the perturbed molecule, in elementary charges.

    Returns
    -------
    unc : float
        An estimate of the uncertainty in the percentage change of electrostatic potential
        energy resulting from the specified perturbations. The uncertainty is defined as
        the absolute difference between the alchemical perturbation estimates obtained
        using the full and reduced versions of the input data.

    Notes
    -----
    This function estimates the uncertainty in the alchemical perturbation estimates
    by computing the difference between two estimates obtained using the full and reduced
    versions of the input data. The full version is expected to be more accurate, but also
    more computationally expensive, as it includes all available data. The reduced version
    is expected to be less accurate, but also faster to compute, as it excludes some of the
    data at the edges of the electrostatic potential energy surface. By comparing the two
    estimates, this function provides a rough estimate of the uncertainty in the alchemical
    perturbation approach, which can be used to assess the reliability of the results
    obtained using this method.
    """
    return abs(
        alchemical_pecd(energy, dr, dq, reduced=False)
        - alchemical_pecd(energy, dr, dq, reduced=True)
    )


def plot_results_per_energy() -> None:
    """
    Plots the alchemical pecd and estimated uncertainty surfaces for a range of energies,
    using a grid of displacement and charge perturbations.

    Returns
    -------
    None

    Notes
    -----
    This function generates a set of contour plots that visualize the alchemical pecd and
    estimated uncertainty surfaces for a range of electrostatic potential energies, using
    a grid of displacement and charge perturbations. The range of energies is defined by
    the integer values between 4 and 10, inclusive. For each energy value, the function
    computes the pecd and estimated uncertainty at 20x20 evenly spaced grid points in
    the [-0.5, 0.5] range for both the displacement and charge perturbations. It then
    generates a contour plot of the pecd surface using a color map, and superimposes a
    contour plot of the estimated uncertainty surface using white contour lines. The
    maximum value is highlighted using a yellow dot. The x and y axes show
    the displacement and charge perturbations, respectively, in atomic units.
    """
    for energy in range(4, 11):
        target = functools.partial(alchemical_pecd, energy)
        drs = np.linspace(-0.5, 0.5, 20)
        dqs = np.linspace(-0.5, 0.5, 20)
        X, Y = np.meshgrid(drs, dqs)
        Z = np.array(
            [(target(dr, dq)) for dr, dq in zip(X.ravel(), Y.ravel())]
        ).reshape(X.shape)
        print(np.amax(Z), np.amin(Z))
        levels = range(-10, 11)
        plt.contourf(X, Y, Z, levels=levels, cmap="RdBu")
        plt.colorbar()

        target2 = functools.partial(estimated_uncertainty, energy)
        Z2 = np.array(
            [(target2(dr, dq)) for dr, dq in zip(X.ravel(), Y.ravel())]
        ).reshape(X.shape)
        cs = plt.contour(
            X,
            Y,
            Z2,
            levels=[
                1,
            ],
            colors="white",
            linewidths=0.8,
        )
        maxval = 0
        pos = None
        for item in cs.collections:
            for i in item.get_paths():
                v = i.vertices
                xs = v[:, 0]
                ys = v[:, 1]
                for x, y in zip(xs, ys):
                    value = abs(target(x, y))
                    if value > maxval:
                        maxval = value
                        pos = (x, y)
        print(pos, maxval)
        plt.scatter(
            (pos[0],), (pos[1],), color="yellow", s=50, edgecolors="grey", zorder=100
        )

        plt.xlabel("dr [a.u.]")
        plt.ylabel("dq [a.u.]")
        plt.title("$\\beta_1$@E={}".format(energy))
        plt.scatter(
            (0, 0, 0, 0, 0),
            (-0.2, -0.1, 0, 0.1, 0.2),
            color="white",
            edgecolors="grey",
            zorder=100,
        )
        plt.scatter(
            (-0.2, -0.1, 0, 0.1, 0.2),
            (0, 0, 0, 0, 0),
            color="white",
            edgecolors="grey",
            zorder=100,
        )
        plt.show()


# %%


def compare_to_lower_order(dr: bool) -> None:
    """
    Compares the alchemical pecd estimates obtained using the full and reduced versions
    of the input data, for a range of photoelectron energies and perturbation
    parameters.

    Parameters
    ----------
    dr : bool
        Whether to compare the alchemical pecd estimates for displacement or charge
        perturbations. If True, the function compares the estimates obtained by varying
        the displacement parameter and fixing the charge parameter at zero. If False,
        the function compares the estimates obtained by varying the charge parameter and
        fixing the displacement parameter at zero.

    Returns
    -------
    None

    Notes
    -----
    This function compares the alchemical pecd estimates obtained using the full and
    reduced versions of the input data, for a range of photoelectron energies
    and perturbation parameters. Specifically, it computes the alchemical pecd estimates
    for the specified perturbation parameter (displacement or charge) at a range of
    energies between 4 and 10, inclusive, using both the full and reduced versions of
    the input data. It then plots the resulting estimates as a function of energy, using
    different colors for the full and reduced estimates, and labels them accordingly.
    Additionally, it plots the estimates obtained by reversing the signs of the
    perturbation parameters, to allow for a comparison with the original estimates. The
    resulting plots show how the alchemical pecd estimates vary with the perturbation
    parameters and the accuracy of the input data, and can be used to assess the
    reliability of the alchemical perturbation method for different scenarios.
    """
    if dr:
        plt.title("+- 0.2 from (+- 0.1, 0) for geometry")
        dr, dq = 0.2, 0.0
    else:
        plt.title("+- 0.2 from (+- 0.1, 0) for charges")
        dr, dq = 0.0, 0.2
    Es = range(4, 11)
    betas = [alchemical_pecd(_, dr, dq, reduced=False) for _ in Es]
    plt.plot(Es, betas, color="C0", label="reduced")
    betas = [alchemical_pecd(_, dr, dq, reduced=True) for _ in Es]
    plt.plot(Es, betas, color="C1", label="full")

    betas = [alchemical_pecd(_, -dr, -dq, reduced=False) for _ in Es]
    plt.plot(Es, betas, color="C0")
    betas = [alchemical_pecd(_, -dr, -dq, reduced=True) for _ in Es]
    plt.plot(Es, betas, color="C1")
    plt.legend()
    plt.show()


compare_to_lower_order(True)
compare_to_lower_order(False)
# %%
def through_energy(dr: float, dq: float) -> None:
    """
    Plots the alchemical pecd estimates obtained using the full and reduced versions of
    the input data, as a function of photoelectron energy, for a fixed set of
    displacement and charge perturbation parameters.

    Parameters
    ----------
    dr : float
        The displacement distance of the perturbed molecule, in Angstroms.
    dq : float
        The charge change of the perturbed molecule, in elementary charges.

    Returns
    -------
    None

    Notes
    -----
    This function computes the alchemical pecd estimates obtained using the full and
    reduced versions of the input data, as a function of photoelectron energy,
    for a fixed set of displacement and charge perturbation parameters. Specifically,
    it computes the alchemical pecd estimates for the specified perturbation parameters
    at a range of energies between 4 and 10, inclusive, using both the full and reduced
    versions of the input data. It then plots the resulting estimates as a function of
    energy, using different colors for the full and reduced estimates, and labels them
    accordingly. The resulting plot shows how the alchemical pecd estimates vary with
    the photoelectron energy, and can be used to assess the reliability of
    the alchemical perturbation method for the specified perturbation parameters.
    """
    Es = range(4, 11)
    betas = [alchemical_pecd(_, dr, dq, reduced=False) for _ in Es]
    plt.plot(Es, betas, color="C0", label="reduced")
    betas = [alchemical_pecd(_, dr, dq, reduced=True) for _ in Es]
    plt.plot(Es, betas, color="C1", label="full")
    plt.legend()
    plt.show()


through_energy(0.5, 0.3)
# %%


def build_krr(
    energies: np.ndarray, drs: np.ndarray, dqs: np.ndarray, betas: np.ndarray
) -> KernelRidge:
    """
    Builds a kernel ridge regression model to learn the relationship between the
    photoelectron energies, displacement and charge perturbations, and the
    corresponding alchemical pecd values.

    Parameters
    ----------
    energies : np.ndarray
        A 1D array of photoelectron energies used in the training set.
    drs : np.ndarray
        A 1D array of displacement perturbations used in the training set.
    dqs : np.ndarray
        A 1D array of charge perturbations used in the training set.
    betas : np.ndarray
        A 1D array of alchemical pecd values corresponding to the input energies, drs,
        and dqs.

    Returns
    -------
    model : sklearn.kernel_ridge.KernelRidge
        A trained kernel ridge regression model that can be used to predict alchemical
        pecd values for new sets of energy, displacement, and charge parameters.
    """
    # Stack the input arrays into a single feature matrix
    X = np.column_stack([energies, drs, dqs])

    # Create a kernel ridge regression model with a Gaussian radial basis function (RBF) kernel
    model = KernelRidge(kernel="rbf")

    # Train the model on the input data
    model.fit(X, betas)

    return model


#%%
def get_long_format_data():
    A = (
        get_data("charge")
        .reset_index()
        .melt(id_vars=["Energy"], var_name="dq", value_name="beta")
    )
    B = (
        get_data("position")
        .reset_index()
        .melt(id_vars=["Energy"], var_name="dr", value_name="beta")
    )
    return pd.concat([A, B], ignore_index=True).fillna(0)


df = get_long_format_data()


# %%
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge


def build_krr_model():
    # Define the features and target
    X = df.drop("beta", axis=1)
    X = pd.DataFrame(X, columns=df.columns.drop("beta"))
    y = df["beta"]

    # Split the data into training, validation, and test sets
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42
    )

    # Define the column transformer to standardize the features
    ct = ColumnTransformer([("scaler", StandardScaler(), X.columns)])

    # Define the pipeline for kernel ridge regression
    pipeline = Pipeline([("transformer", ct), ("model", KernelRidge(kernel="rbf"))])

    # Fit the column transformer on the training set only
    X_train_transformed = ct.fit_transform(X_train)

    # Define the hyperparameters to optimize
    param_grid = {
        "model__alpha": np.logspace(-5, 2, 8),
        "model__gamma": np.logspace(-5, 2, 8),
    }

    # Perform a grid search with cross-validation on the training set to optimize the hyperparameters
    grid_search = GridSearchCV(
        pipeline, param_grid=param_grid, cv=10, scoring="neg_mean_squared_error"
    )
    grid_search.fit(pd.DataFrame(X_train_transformed, columns=X.columns), y_train)

    # Transform the validation and test sets using the fitted column transformer
    X_val_transformed = ct.transform(X_val)
    X_test_transformed = ct.transform(X_test)

    # Evaluate the pipeline on the validation and test sets
    score_val = grid_search.score(
        pd.DataFrame(X_val_transformed, columns=X.columns), y_val
    )
    score_test = grid_search.score(
        pd.DataFrame(X_test_transformed, columns=X.columns), y_test
    )

    # Print the best hyperparameters and the corresponding score
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best score on training set: {grid_search.best_score_}")
    print(f"Score on validation set: {score_val}")
    print(f"Score on test set: {score_test}")
    return grid_search, ct


def apply_krr_model(grid_search, energy, dr, dq) -> float:
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({"Energy": [energy], "dq": [dq], "dr": [dr]})
    input_data_transformed = grid_search[1].transform(input_data)

    # Make a prediction using the fitted pipeline
    prediction = grid_search[0].predict(
        pd.DataFrame(input_data_transformed, columns=input_data.columns)
    )[0]

    return prediction


# %%


def plot_krr_results():
    model = build_krr_model()
    for energy in range(4, 11):
        target = functools.partial(apply_krr_model, model, energy)
        drs = np.linspace(-0.5, 0.5, 20)
        dqs = np.linspace(-0.5, 0.5, 20)
        X, Y = np.meshgrid(drs, dqs)
        Z = np.array(
            [(target(dr, dq) * 100) for dr, dq in zip(X.ravel(), Y.ravel())]
        ).reshape(X.shape)
        print(np.amax(Z), np.amin(Z))
        levels = range(-10, 11)
        plt.contourf(X, Y, Z, levels=levels, cmap="RdBu")
        plt.colorbar()

        plt.xlabel("dr [a.u.]")
        plt.ylabel("dq [a.u.]")
        plt.title("$\\beta_1$@E={}".format(energy))
        plt.scatter(
            (0, 0, 0, 0, 0),
            (-0.2, -0.1, 0, 0.1, 0.2),
            color="white",
            edgecolors="grey",
            zorder=100,
        )
        plt.scatter(
            (-0.2, -0.1, 0, 0.1, 0.2),
            (0, 0, 0, 0, 0),
            color="white",
            edgecolors="grey",
            zorder=100,
        )
        plt.show()
