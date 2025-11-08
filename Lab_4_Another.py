import matplotlib.pyplot as plt
import random
from scipy.stats import shapiro
import numpy as np
import cvxopt
import sklearn.svm
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

 
cvxopt.solvers.options.update({
    "show_progress": False,
    "abstol": 1e-10,  # absolute accuracy
    "reltol": 1e-10,  # relative acc
    "feastol": 1e-10,  # feasibility conditions
})

def get_a(B):
    A = np.zeros(shape=B.shape, dtype=B.dtype)
    A[0, 0] = np.sqrt(B[0, 0])
    A[0, 1] = 0
    A[1, 0] = B[0, 1] / A[0, 0]
    A[1, 1] = np.sqrt(B[1, 1] - np.power(B[0, 1], 2) / B[0, 0])
    return A


def get_standard_normal_distribution(N, size):
    vector = np.zeros(shape=N)
    N = N
    # M = shape[1]
    for i in range(N):
        vector[i] = (np.sum(np.random.uniform(low=0, high=1, size=size)) - size/2) * np.sqrt(12) / np.sqrt(size)
    return vector


def get_normal_distribution(MX, A, N, cpt_size):

    std_norm_vector = get_standard_normal_distribution(N, cpt_size)
    std_norm_vector = std_norm_vector.reshape((N, 1))

    needed_vector = np.dot(A, std_norm_vector) + MX
    return needed_vector


def get_normal_distribution_vector(MX, B, N, vector_length, cpt_size):
    A = get_a(B)
    final_vector = np.ndarray(shape=(0, 0), dtype="float64")
    for i in range(vector_length):
        vector_part = get_normal_distribution(MX, A, N, cpt_size)
        final_vector = np.concatenate((final_vector, vector_part), axis=1) if len(final_vector) > 0 else vector_part
    return final_vector


def get_edge_between_two_vectors(lhs_vector, lhs_M, lhs_B, rhs_vector, rhs_M, rhs_B, X):
    """
    граница для Байеса
    """
    C = 1/2 * (rhs_M.T @ np.linalg.inv(rhs_B) @ rhs_M
         - lhs_M.T @ np.linalg.inv(lhs_B) @ lhs_M
         + np.log(np.linalg.det(lhs_B) / np.linalg.det(rhs_B)))
    lin_coef = lhs_M.T @ np.linalg.inv(lhs_B) - rhs_M.T @ np.linalg.inv(rhs_B)
    lin_coef = lin_coef.reshape((2,))
    double_coef = 1 / 2 * (np.linalg.inv(rhs_B) - np.linalg.inv(lhs_B))
    y0_array = []
    y1_array = []
    for x in X:
        D = (np.power((x*double_coef[1][0] + x*double_coef[0][1] + lin_coef[1]), 2)
             - 4 * double_coef[1][1] * (x ** 2 * double_coef[0][0] + x * lin_coef[0] + C))
        D = D[0][0]
        y0 = -(x*double_coef[1][0] + x*double_coef[0][1] + lin_coef[1]) + np.sqrt(D)
        y0 /= (2 * double_coef[1][1])
        y1 = -(x*double_coef[1][0] + x*double_coef[0][1] + lin_coef[1]) - np.sqrt(D)
        y1 /= (2 * double_coef[1][1])
        y0_array.append(y0)
        y1_array.append(y1)
    # print(y0_array)
    return y0_array, y1_array


def dlj(lhs_vector, lhs_M, lhs_B, rhs_vector, rhs_M, rhs_B):
    """
    считаем ошибки первого и второго рода для Байеса
    """
    first_vector_check = []
    eps_needed = 0.05
    N = lhs_vector.shape[1]
    for j in range(N):
        x = [lhs_vector[0][j], lhs_vector[1][j]]
        x = np.array(x)
        x = x.reshape((2, 1))
        C = 1/2 * (rhs_M.T @ np.linalg.inv(rhs_B) @ rhs_M
             - lhs_M.T @ np.linalg.inv(lhs_B) @ lhs_M
             + np.log(np.linalg.det(lhs_B) / np.linalg.det(rhs_B)))
        lin_coef = (lhs_M.T @ np.linalg.inv(lhs_B) - rhs_M.T @ np.linalg.inv(rhs_B)) @ x
        double_coef = 1 / 2 * x.T @ (np.linalg.inv(rhs_B) - np.linalg.inv(lhs_B)) @ x
        first_vector_check.append(double_coef + lin_coef + C)
    first_vector_check = np.array(first_vector_check)
    p0 = np.count_nonzero(first_vector_check[first_vector_check < 0]) / N
    print(f"Вероятность ошибки первого рода: {p0}")

    second_vector_check = []
    N = rhs_vector.shape[1]
    for j in range(N):
        x = [rhs_vector[0][j], rhs_vector[1][j]]
        x = np.array(x)
        x = x.reshape((2, 1))
        C = 1 / 2 * (rhs_M.T @ np.linalg.inv(rhs_B) @ rhs_M
                     - lhs_M.T @ np.linalg.inv(lhs_B) @ lhs_M
                     + np.log(np.linalg.det(lhs_B) / np.linalg.det(rhs_B)))
        lin_coef = (lhs_M.T @ np.linalg.inv(lhs_B) - rhs_M.T @ np.linalg.inv(rhs_B)) @ x
        double_coef = 1 / 2 * x.T @ (np.linalg.inv(rhs_B) - np.linalg.inv(lhs_B)) @ x
        second_vector_check.append(double_coef + lin_coef + C)
    second_vector_check = np.array(second_vector_check)
    p1 = np.count_nonzero(second_vector_check[second_vector_check > 0]) / N
    print(f"Вероятность ошибки второго рода: {p1}")

def solve_qp(*args):
    args = [cvxopt.matrix(x) for x in args]
    res = cvxopt.solvers.qp(*args)

    if res["status"] != "optimal":
        raise ValueError("No solution!")

    return np.array(res["x"]).ravel()

def decision_funct_gauss(x, r, lambdas, xs, w_N):
    sigma_gauss = 1
    disper = sigma_gauss ** 2
    for_gauss = np.array([[xs[j, :] - x[i, :] for i in range(x.shape[0])] for j in range(xs.shape[0])])

    euclid = for_gauss[:, :, 0] ** 2 + for_gauss[:, :, 1] ** 2
    return (r * lambdas) @ np.exp(- 1 / (2 * disper) * euclid) + w_N

def decision_function(r, lambdas, xs, w_N, kernel):
    if kernel == "linear":
        return lambda x: np.dot((r * lambdas), np.dot(xs, x.T)) + w_N
    elif kernel == "poly":
        return lambda x: np.dot((r * lambdas), np.dot(xs, x.T) ** 3) + w_N
    elif kernel == "gauss":
        return lambda  x : decision_funct_gauss(x, r, lambdas, xs, w_N)
        # return lambda x: np.dot((r * lambdas), np.dot(xs, x.transpose()) ** 3) + w_N
    # np.exp(- 1 / (2 * disper) * (xs - x))


def svm(x, r, C, kernel, eps_zero=1e-11, eps_c=1e-10):
    if kernel == "linear":
        P = np.outer(r, r) * np.dot(x, x.T)

    if kernel == "poly":
        P = np.outer(r, r) * (np.dot(x, x.T) ** 3)

    if kernel == "gauss":
        # P = np.outer(r, r) * np.exp(- (np.dot(x, x.transpose()) ** 3))
        sigma_gauss = 1
        dx = sigma_gauss ** 2
        for_gauss = np.array([[x[j, :] - x[i, :] for i in range(x.shape[0])] for j in range(x.shape[0])])
        # x_for_test = np.copy(x)
        # x_for_test, _ = np.meshgrid(x_for_test, x_for_test)
        euclid = for_gauss[:,:,0] ** 2 + for_gauss[:,:, 1] ** 2
        P = np.outer(r, r) * np.exp(- 1 / (2 * dx) * euclid)

    q = -np.ones(2 * N)

    A = r.reshape(1, -1)
    b = np.zeros(1)

    G = -np.eye(2 * N)
    h = np.zeros(2 * N)

    if C != np.inf:
        G = np.concatenate((G, np.eye(2 * N)))
        h = np.concatenate((h, np.ones(2 * N) * C))

    lambdas = solve_qp(P, q, G, h, A, b)
    s = (0 + eps_zero < lambdas) & (lambdas < C - eps_c)  # поскольку в массиве нет нулей, а значения e-14 и т.д.
    support_x = x[s]
    support_r = r[s]

    wt_x = decision_function(r, lambdas, x, 0, kernel)
    wN = (support_r - wt_x(support_x)).mean()
    dx = decision_function(r, lambdas, x, wN, kernel)

    return support_x, dx


def show_classes(x0, x1, marked, title=""):
    plt.scatter(x0[:, 0], x0[:, 1], color="red", label="x0", alpha=0.7)
    plt.scatter(x1[:, 0], x1[:, 1], color="green", label="x1", alpha=0.7)
    plt.title(title)

    if marked is not None:
        plt.plot(marked[:, 0], marked[:, 1], "o", color="b",
                  markersize=9, fillstyle='none', markeredgewidth=2)
    plt.legend()


def show_clf(x0, x1, support, d, title=""):
    show_classes(x0, x1, support, title)
    plt.ylim(-2, 2.5)
    plt.xlim(-2, 1)
    x = np.linspace(-2, 1, 100)
    y = np.linspace(-2, 2.5, 100)
    x, y = np.meshgrid(x, y)
    xy = np.dstack((x, y))
    z = d(xy.reshape(-1, 2)).reshape(x.shape)
    print(f"неправильно проклассиф. из x0: {calc_error(d, x0)} т.е. ошибка={calc_error(d, x0)/N}")
    print(f"неправильно проклассиф. из x1: {N - calc_error(d, x1)} т.е. ошибка={(N - calc_error(d, x1))/N}")
    plt.contour(x, y, z, levels=[-1, 0, 1], colors="k", alpha=0.5, linestyles=["--", "-", "--"])


def calc_error(d, x):
    line = d(x.reshape(-1, 2))
    counter = 0
    for i in line:
        if i > 0:
            counter += 1
    return counter


def show_case_lab(x0, x1, kernel, C, title, start=False, eps_zero=1e-11, eps_c=1e-10):
    """
    Собственная реализация для ядер и т.д.
    :param x0:
    :param x1:
    :param kernel:
    :param C:
    :param title:
    :param start:
    :return:
    """
    x = np.concatenate((x0, x1))
    support, df = svm(x, r, C, kernel, eps_zero=eps_zero, eps_c=eps_c)
    if start:
        show_clf(x0, x1, support, df, "Собств.реализация\n" + title + "C=" + str(C))
    else:
        show_clf(x0, x1, support, df, "C=" + str(C))


def show_case_sklearn(x0, x1, kernel, C, title, start=False):
    """
    Определение параметров и границ с помощью sklearn
    :param x0:
    :param x1:
    :param kernel:
    :param C:
    :param title:
    :param start:
    :return:
    """
    x = np.concatenate((x0, x1))
    if C == np.inf:
        sklearn_clf = sklearn.svm.SVC(kernel=kernel, degree=3)
    else:
        sklearn_clf = sklearn.svm.SVC(kernel=kernel, degree=3, C=C)
    sklearn_clf.fit(x, r)
    support = sklearn_clf.support_vectors_
    df = sklearn_clf.decision_function

    if start:
        show_clf(x0, x1, support, df, title="sklearn.SVC\n" + title + "C=" + str(C))
    else:
        show_clf(x0, x1, support, df, title="C=" + str(C))


def show_case_linearsvc(x0, x1, kernel, C, title, start=False):
    """
    Определение параметров и границ с помощью sklearn
    :param x0:
    :param x1:
    :param kernel:
    :param C:
    :param title:
    :param start:
    :return:
    """
    x = np.concatenate((x0, x1))
    clf = make_pipeline(StandardScaler(), LinearSVC(dual="auto", random_state=0, tol=1e-5))
    clf.fit(x, r)
    df = clf.decision_function
    support = None
    show_clf(x0, x1, None, df, title="LinearSVC\nC=" + str(np.inf))

    if start:
        show_clf(x0, x1, support, df, title="LinearSVC\n" + title + "C=" + str(C))
    else:
        show_clf(x0, x1, support, df, title="C=" + str(C))


def show_bayes(x0, MX1, B1, x1, MX2, B2, title="", start=False):
    plt.ylim(-2, 2.5)
    plt.xlim(-2, 1)
    x_axis = np.linspace(-2, 1, 250)
    dlj(x0.T, MX1, B1, x1.T, MX2, B2)
    y0, y1 = get_edge_between_two_vectors(x0.T, MX1, B1, x1.T, MX2, B2, x_axis)

    plt.plot(x_axis, y0, color="black", label="line_0_1", lw=1)
    # plt.scatter(X, y1, color="Yellow", label="line_0_1")
    plt.plot(x_axis, y1, color="black", label="line_0_1", lw=1)
    show_classes(x0, x1, marked=None, title=title)

def print_tasks(S1_sep, S2_sep, S1_nsep, S2_nsep):
    """
    Выполнение пунктов лабораторной с 2 до 4-го
    :param S1_sep: линейно разделимый класс: выборка 1
    :param S2_sep: линейно разделимый класс: выборка 2
    :param S1_nsep: линейно неразделимый класс: выборка 1
    :param S2_nsep: линейно неразделимый класс: выборка 2
    """
    B1 = np.array([[0.04, -0.01], [-0.03, 0.03]])
    B2 = np.array([[0.03, -0.01], [-0.02, 0.06]])
    N = 2
    MX1 = np.array([[0], [0]])
    MX2 = np.array([[-1], [1]])


    print("Пункт 2")
    plt.gcf().set_size_inches(12, 6)
    plt.subplots_adjust(hspace=0.25)
    title = "Пункт 2 графики: Линейно разделимые классы "
    plt.subplot(1, 4, 1)
    show_case_lab(S1_sep, S2_sep, kernel="linear", C=np.inf, title=title, start=True)
    plt.subplot(1, 4, 2)
    show_case_sklearn(S1_sep, S2_sep, kernel="linear", C=np.inf, title=title, start=True)
    plt.subplot(1, 4, 3)
    show_case_linearsvc(S1_sep, S2_sep, None, None, "", True)
    # байес:


    plt.subplot(1, 4, 4)
    print("Результат для БК:")
    show_bayes(S1_sep, MX1, B1, S2_sep, MX2, B2, title="БК")
    plt.show()

    # punkt 3
    B1 = np.array([[0.11, -0.05], [-0.06, 0.11]])
    B2 = np.array([[0.11, -0.07], [-0.05, 0.09]])
    MX1 = np.array([[0], [0]])
    MX2 = np.array([[-1], [1]])
    print("Пункт 3")
    plt.gcf().set_size_inches(12, 18)
    plt.subplots_adjust(hspace=0.35)
    title = "Метод опорных векторов для лин.неразделимых классов\n"
    plt.subplot(4, 2, 1)
    print("результаты для C=0.1")
    show_case_lab(S1_nsep, S2_nsep, kernel="linear", C=0.1, title=title, start=True, eps_zero=1e-12)
    plt.subplot(4, 2, 2)
    show_case_sklearn(S1_nsep, S2_nsep, kernel="linear", C=0.1, title=title, start=True)
    plt.subplot(4, 2, 3)
    print("результаты для C=1")
    show_case_lab(S1_nsep, S2_nsep, kernel="linear", C=1, title=title)
    plt.subplot(4, 2, 4)
    show_case_sklearn(S1_nsep, S2_nsep, kernel="linear", C=1, title=title)
    plt.subplot(4, 2, 5)
    print("результаты для C=10")
    show_case_lab(S1_nsep, S2_nsep, kernel="linear", C=10, title=title)
    plt.subplot(4, 2, 6)
    show_case_sklearn(S1_nsep, S2_nsep, kernel="linear", C=10, title=title)

    C = 5
    add_C = 2
    print(f"\nрезультаты для C={C}")
    plt.subplot(4, 2, 7)
    show_case_lab(S1_nsep, S2_nsep, kernel="linear", C=C, title=title)
    """
    print(f"\nрезультаты для C={C+add_C}")
    plt.subplot(4, 2, 8)
    show_case_lab(S1_nsep, S2_nsep, kernel="linear", C=C+add_C, title=title)
    plt.show()
    """
    print("\nРезультат для БК:")
    plt.subplot(4, 2, 8)
    show_bayes(S1_nsep, MX1, B1, S2_nsep, MX2, B2, title="БК")
    plt.show()
    # punkt 4
    plt.gcf().set_size_inches(12, 18)
    plt.subplots_adjust(hspace=0.35)
    print("Пункт 4: полином")
    title = "Метод через ядра: kernel=poly\n"
    plt.subplot(4, 2, 1)
    print("\nрезультаты для C=0.1")
    show_case_lab(S1_nsep, S2_nsep, kernel="poly", C=0.1, title=title, start=True)
    plt.subplot(4, 2, 2)
    show_case_sklearn(S1_nsep, S2_nsep, kernel="poly", C=0.1, title=title, start=True)
    plt.subplot(4, 2, 3)
    print("\nрезультаты для C=1")
    show_case_lab(S1_nsep, S2_nsep, kernel="poly", C=1, title=title)
    plt.subplot(4, 2, 4)
    show_case_sklearn(S1_nsep, S2_nsep, kernel="poly", C=1, title=title)
    plt.subplot(4, 2, 5)
    print("\nрезультаты для C=10")
    show_case_lab(S1_nsep, S2_nsep, kernel="poly", C=10, title=title)
    plt.subplot(4, 2, 6)
    show_case_sklearn(S1_nsep, S2_nsep, kernel="poly", C=10, title=title)

    C = 5
    add_C = 2
    print(f"\nрезультаты для C={C}")
    plt.subplot(4, 2, 7)
    show_case_lab(S1_nsep, S2_nsep, kernel="poly", C=C, title=title)
    print("\nРезультат для БК:")
    plt.subplot(4, 2, 8)
    show_bayes(S1_nsep, MX1, B1, S2_nsep, MX2, B2, title="БК")
    plt.show()
    """
    print(f"\nрезультаты для C={C+add_C}")
    plt.subplot(4, 2, 8)
    show_case_lab(S1_nsep, S2_nsep, kernel="poly", C=C+add_C, title=title)
    """

    # gauss:
    print("Пункт 4: гаусс")
    plt.figure()
    plt.gcf().set_size_inches(12, 18)
    plt.subplots_adjust(hspace=0.35)
    title = "Метод через ядра: kernel=gauss\n"
    print("\nрезультаты для C=0.1")
    plt.subplot(4, 2, 1)
    show_case_lab(S1_nsep, S2_nsep, kernel="gauss", C=0.1, title=title, start=True)
    plt.subplot(4, 2, 2)
    show_case_sklearn(S1_nsep, S2_nsep, kernel="rbf", C=0.1, title=title, start=True)
    plt.subplot(4, 2, 3)
    print("\nрезультаты для C=1")
    show_case_lab(S1_nsep, S2_nsep, kernel="gauss", C=1, title=title)
    plt.subplot(4, 2, 4)
    show_case_sklearn(S1_nsep, S2_nsep, kernel="rbf", C=1, title=title)
    plt.subplot(4, 2, 5)
    print("\nрезультаты для C=10")
    show_case_lab(S1_nsep, S2_nsep, kernel="gauss", C=10, title=title)
    plt.subplot(4, 2, 6)
    show_case_sklearn(S1_nsep, S2_nsep, kernel="rbf", C=10, title=title)
    C = 5
    add_C = 2
    print(f"\nрезультаты для C={C}")
    plt.subplot(4, 2, 7)
    show_case_lab(S1_nsep, S2_nsep, kernel="gauss", C=C, title=title)

    print("\nРезультат для БК:")
    plt.subplot(4, 2, 8)
    show_bayes(S1_nsep, MX1, B1, S2_nsep, MX2, B2, title="БК")
    plt.show()
    """
    print(f"\nрезультаты для C={C + add_C}")
    plt.subplot(4, 2, 8)
    show_case_lab(S1_nsep, S2_nsep, kernel="gauss", C=C+add_C, title=title)
    """
    plt.show()


if __name__ == "__main__":
    save = True
    if save:
        B1 = np.array([[0.04, -0.01], [-0.03, 0.03]])
        B2 = np.array([[0.03, -0.01], [-0.02, 0.06]])
        N = 2
        MX1 = np.array([[0], [0]])
        MX2 = np.array([[-1], [1]])
        first_vector = get_normal_distribution_vector(MX1, B1, N, 100, cpt_size=50).T
        second_vector = get_normal_distribution_vector(MX2, B2, N, 100, cpt_size=50).T
        np.save("Lab4_1.npy", first_vector)
        np.save("Lab4_2.npy", second_vector)

        B1 = np.array([[0.11, -0.05], [-0.06, 0.11]])
        B2 = np.array([[0.11, -0.07], [-0.05, 0.09]])
        MX1 = np.array([[0], [0]])
        MX2 = np.array([[-1], [1]])
        first_vector_nsep = get_normal_distribution_vector(MX1, B1, N, 100, cpt_size=50).T
        second_vector_nsep = get_normal_distribution_vector(MX2, B2, N, 100, cpt_size=50).T
        np.save("Lab4_3.npy", first_vector_nsep)
        np.save("Lab4_4.npy", second_vector_nsep)
    else:
        first_vector = np.load('fourth_lab_first_vector.npy')
        second_vector = np.load('fourth_lab_second_vector.npy')
        first_vector_nsep = np.load('fourth_lab_third_vector.npy')
        second_vector_nsep = np.load('fourth_lab_fourth_vector.npy')

    # Строим линейно разделимые

    """
    plt.figure()
    plt.scatter(first_vector[0, :], first_vector[1, :], s=5, color='red', marker='o', label="first_vector")
    plt.scatter(second_vector[0, :], second_vector[1, :], s=5, color='blue', marker='x', label="second_vector")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.legend()
    plt.show()
    """
    # Строим линейно неразделимые


    N = first_vector_nsep.shape[0]
    r = np.concatenate((-np.ones(N), np.ones(N)))
    print_tasks(first_vector, second_vector, first_vector_nsep, second_vector_nsep)
    """
    plt.figure()
    plt.scatter(first_vector[0, :], first_vector[1, :], s=5, color='red', marker='o', label="first_vector")
    plt.scatter(second_vector[0, :], second_vector[1, :], s=5, color='blue', marker='x', label="second_vector")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.legend()
    plt.show()
    """
