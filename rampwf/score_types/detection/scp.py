import numpy as np

from .base import DetectionBaseScoreType


class SCP(DetectionBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, shape, name='scp', precision=3, conf_threshold=0.5,
                 minipatch=None):
        self.shape = shape
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        """
        Score based on a matching by reprojection of craters on mask-map.

        True craters are projected positively, predicted craters negatively,
        so they can cancel out. Then the sum of the absolute value of the
        residual map is taken.

        The best score value for a perfect match is 0.
        The worst score value for a given patch is the sum of all crater
        instances in both `y_true` and `y_pred`.

        Parameters
        ----------
        y_true : list of list of tuples (x, y, radius)
            List of coordinates and radius of actual craters for set of patches
        y_pred : list of list of tuples (x, y, radius)
            List of coordinates and radius of predicted craters for set of
            patches

        Returns
        -------
        float : score for a given patch, the lower the better

        """
        scps = np.array(
            [scp_single(t, p, self.shape, self.minipatch)
             for t, p in zip(y_true, y_pred)])
        return np.sum(scps[:, 0]) / np.sum(scps[:, 1:3])


def scp_single(y_true, y_pred, shape, minipatch=None):
    """
    L1 distance between superposing bounding box cylinder or prism maps.

    True craters are projected positively, predicted craters negatively,
    so they can cancel out. Then the sum of the absolute value of the
    residual map is taken.

    The best score value for a perfect match is 0.
    The worst score value for a given patch is the sum of all crater
    instances in both `y_true` and `y_pred`.

    Parameters
    ----------
    y_true : list of tuples (x, y, radius)
        List of coordinates and radius of actual craters in a patch
    y_pred : list of tuples (x, y, radius)
        List of coordinates and radius of craters predicted in the patch
    shape : tuple of int
        Shape of the main patch
    minipatch : [row_min, row_max, col_min, col_max], optional
        Bounds of the internal scoring patch (default is None)

    Returns
    -------
    float : score for a given patch, the lower the better

    """
    map_true, map_pred = circle_maps(y_true, y_pred, shape)
    if minipatch is not None:
        map_true = map_true[
            minipatch[0]:minipatch[1], minipatch[2]:minipatch[3]]
        map_pred = map_pred[
            minipatch[0]:minipatch[1], minipatch[2]:minipatch[3]]
    # Sum all the pixels
    score = np.abs(map_true - map_pred).sum()
    n_true = map_true.sum()
    n_pred = map_pred.sum()
    return score, n_true, n_pred


def project_circle(circle, image=None, shape=None,
                   normalize=True, negative=False):
    """
    Project circles on an image.

    Parameters
    ----------
    circle : array-like
        x, y, radius
    image : array-like, optional
        image on which to project the circle
    shape : tuple of ints, optional
        shape of the image on which to project the circle
    normalize : bool, optional (default is `True`)
        normalize the total surface of the circle to unity
    negative : bool, optional (default is `False`)
        subtract the circle instead of adding it

    Returns
    -------
    array-like : image with projected circle

    """
    if image is None:
        if shape is None:
            raise ValueError("Either `image` or `shape` must be defined")
        else:
            image = np.zeros(shape)
    else:
        shape = image.shape
    x, y, radius = circle
    coords = circle_coords(x, y, radius, shape=shape)
    value = 1

    if not coords[0].size:
        # corner case where circle is outside the image
        return image

    if normalize:
        value /= coords[0].size
    if negative:
        value = -value
    image[coords] += value
    return image


def circle_maps(y_true, y_pred, shape):
    """
    Create a map to compare true and predicted craters.

    The craters (circles) are projected on the map with a coefficient
    chosen so its sum is normalized to unity.

    True and predicted craters are projected with a different sign,
    so that good predictions tend to cancel out the true craters.

    Parameters
    ----------
    y_pred, y_true : array-like of shape (3, X)
        list of circle positions (x, y, radius)
    shape : tuple of ints, optional
        shape of image

    Returns
    -------
    array-like : image with projected true and predicted circles

    """
    map_true = np.zeros(shape)
    map_pred = np.zeros(shape)

    # Add true craters positively
    for circle in y_true:
        map_true = project_circle(
            circle, map_true, shape=shape, normalize=True)

    # Add predicted craters negatively
    for circle in y_pred:
        map_pred = project_circle(
            circle, map_pred, shape=shape, normalize=True)

    return map_true, map_pred


#
# Copyright scikit-image below
#
def _ellipse_in_shape(shape, center, radii, rotation=0.):
    """Generate coordinates of points within ellipse bounded by shape.

    Parameters
    ----------
    shape :  iterable of ints
        Shape of the input image.  Must be length 2.
    center : iterable of floats
        (row, column) position of center inside the given shape.
    radii : iterable of floats
        Size of two half axes (for row and column)
    rotation : float, optional
        Rotation of the ellipse defined by the above, in radians
        in range (-PI, PI), in contra clockwise direction,
        with respect to the column-axis.
    Returns
    -------
    rows : iterable of ints
        Row coordinates representing values within the ellipse.
    cols : iterable of ints
        Corresponding column coordinates representing values within the ellipse
    """
    r_lim, c_lim = np.ogrid[0:float(shape[0]), 0:float(shape[1])]
    r_org, c_org = center
    r_rad, c_rad = radii
    rotation %= np.pi
    sin_alpha, cos_alpha = np.sin(rotation), np.cos(rotation)
    r, c = (r_lim - r_org), (c_lim - c_org)
    distances = ((r * cos_alpha + c * sin_alpha) / r_rad) ** 2 \
        + ((r * sin_alpha - c * cos_alpha) / c_rad) ** 2
    return np.nonzero(distances < 1)


def ellipse(r, c, r_radius, c_radius, shape=None, rotation=0.):
    """Generate coordinates of pixels within ellipse.

    Parameters
    ----------
    r, c : double
        Centre coordinate of ellipse.
    r_radius, c_radius : double
        Minor and major semi-axes. ``(r/r_radius)**2 + (c/c_radius)**2 = 1``.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for ellipses which exceed the
        image size.
        By default the full extent of the ellipse are used.
    rotation : float, optional (default 0.)
        Set the ellipse rotation (rotation) in range (-PI, PI)
        in contra clock wise direction, so PI/2 degree means swap ellipse axis
    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of ellipse.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    Examples
    --------
    >>> from skimage.draw import ellipse
    >>> img = np.zeros((10, 12), dtype=np.uint8)
    >>> rr, cc = ellipse(5, 6, 3, 5, rotation=np.deg2rad(30))
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    Notes
    -----
    The ellipse equation::
        ((x * cos(alpha) + y * sin(alpha)) / x_radius) ** 2 +
        ((x * sin(alpha) - y * cos(alpha)) / y_radius) ** 2 = 1
    Note that the positions of `ellipse` without specified `shape` can have
    also, negative values, as this is correct on the plane. On the other hand
    using these ellipse positions for an image afterwards may lead to appearing
    on the other side of image, because ``image[-1, -1] = image[end-1, end-1]``
    >>> rr, cc = ellipse(1, 2, 3, 6)
    >>> img = np.zeros((6, 12), dtype=np.uint8)
    >>> img[rr, cc] = 1
    >>> img
    array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]], dtype=uint8)
    """
    center = np.array([r, c])
    radii = np.array([r_radius, c_radius])
    # allow just rotation with in range +/- 180 degree
    rotation %= np.pi

    # compute rotated radii by given rotation
    r_radius_rot = abs(r_radius * np.cos(rotation)) \
        + c_radius * np.sin(rotation)
    c_radius_rot = r_radius * np.sin(rotation) \
        + abs(c_radius * np.cos(rotation))
    # The upper_left and lower_right corners of the smallest rectangle
    # containing the ellipse.
    radii_rot = np.array([r_radius_rot, c_radius_rot])
    upper_left = np.ceil(center - radii_rot).astype(int)
    lower_right = np.floor(center + radii_rot).astype(int)

    if shape is not None:
        # Constrain upper_left and lower_right by shape boundary.
        upper_left = np.maximum(upper_left, np.array([0, 0]))
        lower_right = np.minimum(lower_right, np.array(shape[:2]) - 1)

    shifted_center = center - upper_left
    bounding_shape = lower_right - upper_left + 1

    rr, cc = _ellipse_in_shape(bounding_shape, shifted_center, radii, rotation)
    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += upper_left[0]
    cc += upper_left[1]
    return rr, cc


def circle_coords(r, c, radius, shape=None):
    """Generate coordinates of pixels within circle.

    Parameters
    ----------
    r, c : double
        Centre coordinate of circle.
    radius : double
        Radius of circle.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for circles that exceed the image
        size. If None, the full extent of the circle is used.
    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of circle.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    Examples
    --------
    >>> from skimage.draw import circle
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = circle(4, 4, 5)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    return ellipse(r, c, radius, radius, shape)
