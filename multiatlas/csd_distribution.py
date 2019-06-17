import numpy as np
from scipy.special import sph_harm
from dipy.data import get_sphere

def cartesian_to_polar(directions):
    r = np.linalg.norm(directions, axis=1)
    th = np.arccos(directions[:, -1] / r)
    ph = np.arctan2(directions[:, 1], directions[:, 0])

    return th, ph

def B(th, ph):
    sh_ord = 8
    n_sph = np.sum([2 * i + 1 for i in range(0, sh_ord + 1, 2)])
    SPH_sph2sig = np.zeros((th.shape[0], n_sph), dtype='complex')
    n_sph = 0
    for i in range(0, sh_ord + 1, 2):
        ns = np.zeros(2 * i + 1) + i
        ms = np.arange(-i, i + 1)
        SPH_sph2sig[:, n_sph: n_sph + 2 * i + 1] = sph_harm(ms[None, :], ns[None, :], ph[:, None], th[:, None])
        n_sph += 2 * i + 1

    return SPH_sph2sig

th, ph = cartesian_to_polar(get_sphere('repulsion100').vertices)
vertices_b = B(th, ph)

def estimate_fodf(directions, sphere_vertices):
    directions = np.array(directions)
    th, ph = cartesian_to_polar(directions)
    directions_b = B(th, ph)

    fodf = np.dot(vertices_b, directions_b.conj().T).sum(1).real
    fodf[fodf < 0] = 0

    return fodf
