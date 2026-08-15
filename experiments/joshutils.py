import numpy as np
from qiskit.quantum_info import random_hermitian, Operator, SparsePauliOp
import scipy


def random_diagonal_matrix(n):
    rng = np.random.default_rng()
    diags = rng.normal(size=n)
    return np.diag(diags)


def prl_beta_gtau(h, o, beta, tau):
    """
    Returns G(tau) = <H_1(tau)H_1> - <H_1>^2 for
    <H_1> = Tr[H_1 rho(Bz, beta)] for rho the
    thermal state PRL model at inverse temp beta.
    """
    evals, evecs = np.linalg.eigh(h)

    z = np.sum(np.exp(-beta * evals))

    corr = (
        np.sum(
            [
                np.exp(-(beta - tau) * evals[i])
                * np.exp(-tau * evals[j])
                * np.abs(np.dot(np.conj(evecs[:, i].T), np.dot(o, evecs[:, j]))) ** 2
                for i in range(len(evals))
                for j in range(len(evals))
            ]
        )
        / z
    )

    avgO = (
        np.sum(
            [
                np.exp(-beta * evals[i])
                * np.dot(np.conj(evecs[:, i].T), np.dot(o, evecs[:, i]))
                for i in range(len(evals))
            ]
        )
        / z
    )

    gTau = corr - (avgO) ** 2
    if gTau.imag < 1e-8:
        gTau = gTau.real

    return gTau


def prl_beta_chiE(h, o, beta):
    """
    Integrates prl_beta_gtau over tau from [0, beta].
    """
    result, error = scipy.integrate.quad(
        lambda tau: prl_beta_gtau(h, o, beta, tau), 0, beta
    )
    return result, error


def prl_beta_chiF(h, o, beta):
    """
    Integrates tau*prl_beta_gtau over tau from [0, beta/2].
    """
    result, error = scipy.integrate.quad(
        lambda tau: tau * prl_beta_gtau(h, o, beta, tau), 0, beta / 2
    )
    return result, error


def calc_actual_gs_gap(h):
    evals, _ = np.linalg.eigh(h)
    return evals[1] - evals[0]


def calc_approx_gs_gap(h, o, beta):
    M_0, error_0 = prl_beta_chiE(h, o, beta)
    M_1, error_1 = prl_beta_chiF(h, o, beta)
    return M_0 / M_1 / 2


def write_as_paulis(filename, op):
    qop = Operator(op)
    pauli_op = SparsePauliOp.from_operator(qop)
    with open(filename, "w") as f:
        for coeff, gates in zip(pauli_op.coeffs, pauli_op.paulis):
            f.write(f"{np.real(coeff)} ")
            for index, gate in enumerate(gates):
                if str(gate) != "I":
                    f.write(f"{index + 1} {gate} ")
            f.write("\n")


def itay_weird_tfim():
    j = 1
    h = 0.7
    g = 0.13
    h1 = SparsePauliOp.from_list([("ZZII", 1), ("IZZI", 1), ("IIZZ", 1), ("ZIIZ", 1)])
    h2 = SparsePauliOp.from_list([("XIII", 1), ("IXII", 1), ("IIXI", 1), ("IIIX", 1)])
    h3 = SparsePauliOp.from_list([("ZIII", 1), ("IZII", 1), ("IIZI", 1), ("IIIZ", 1)])
    return -j * h1 - h * h2 - g * h3


if __name__ == "__main__":
    N = 4
    beta_min = 0.5
    beta_max = 20
    beta_steps = 10

    # hamiltonian = itay_weird_tfim()
    hamiltonian = random_hermitian(N).to_matrix()
    random_operator = random_hermitian(N).to_matrix()
    diagonal_operator = random_diagonal_matrix(N)

    evals, evecs = np.linalg.eigh(hamiltonian)
    evecs = evecs.T
    off_diag = np.kron(evecs[0], np.conj(evecs[1])).reshape(N, N)
    off_diag += np.conj(off_diag).T

    write_as_paulis("H.txt", hamiltonian)
    write_as_paulis("A.txt", random_operator)
    write_as_paulis("B.txt", off_diag)

    print("beta random_approx(A) diag_approx off_diag_approx(B)")
    for beta in np.linspace(beta_min, beta_max, beta_steps, dtype=np.float128):
        approx_gap_random = calc_approx_gs_gap(hamiltonian, random_operator, beta)
        approx_gap_diag = calc_approx_gs_gap(hamiltonian, diagonal_operator, beta)
        approx_gap_off_diag = calc_approx_gs_gap(hamiltonian, off_diag, beta)
        print(f"{beta} {approx_gap_random} {approx_gap_diag} {approx_gap_off_diag}")

    actual_gap = calc_actual_gs_gap(hamiltonian)
    print(f"actual gap: {actual_gap}")
