import numpy as np


def compute_overlaps(ev):
    return np.moveaxis(ev[1:], -1, -2) @ ev[:-1]


def patch_overlaps(overlaps):
    row_mask = overlaps.sum(axis=-1) == 0
    col_mask = overlaps.sum(axis=-2) == 0
    indices, rows = np.argwhere(row_mask).T
    _, cols = np.argwhere(col_mask).T
    overlaps[indices, rows, cols] = 1
    return overlaps


def eigensign(ev):
    ev = ev.copy()
    size = ev.shape[-1]

    overlaps = compute_overlaps(ev)
    sign_flips = np.diagonal(np.sign(overlaps.real), axis1=-2, axis2=-1)
    sign_flips = np.vstack((sign_flips, -np.ones((1, size))))

    for n in range(size):
        sign_flip_indices = np.where(sign_flips[:, n] < 0)[0] + 1
        if len(sign_flip_indices) % 2:
            sign_flip_indices = sign_flip_indices[:-1]
        for start, stop in sign_flip_indices.reshape(-1, 2):
            ev[start:stop, ..., n] *= -1
    return ev


def eigensort(ew, ev, sort_eigenvectors=True, sort_eigenvector_signs=True):
    ew = ew.copy()
    ev = ev.copy()
    size = ev.shape[-1]
    cumulative_transform = np.diag(np.ones(size, dtype=int))

    overlaps = compute_overlaps(ev)
    overlaps = patch_overlaps(np.round(np.conj(overlaps) * overlaps).real)

    # find the overlap matrices with off-diagonal elements
    diag = np.diagonal(overlaps, axis1=-2, axis2=-1).sum(axis=1)
    indices = np.where(diag < size)[0] + 1

    for i in indices:
        new_transform = overlaps[i - 1] @ cumulative_transform
        transform = np.linalg.inv(cumulative_transform) @ new_transform
        ew[i:] = ew[i:] @ transform
        if sort_eigenvectors:
            ev[i:] = ev[i:] @ transform
        cumulative_transform = new_transform

    if sort_eigenvectors and sort_eigenvector_signs:
        ev = eigensign(ev)

    return ew, ev
