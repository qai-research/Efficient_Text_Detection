import cv2
import numpy as np
from pathlib import Path


class Pipeline:
    """
    Pipeline is the glue that sticks all modules (detection, recognition, etc.) together.

    Attributes
    ----------
    config : dict
        depend on each subclass, this configuration can be anything

    Interfaces
    ----------
    run(input_folder, output_folder)
        execute the pipeline through and through
    """
    def __init__(self, **config):
        """
        Parameters
        ----------
        config : keyword arguments
            depend on each subclass, this configuration can be anything
        """
        self.config = config

    def run(self, input_folder, output_folder):
        """Interface for the pipeline execution

        Parameters
        ----------
        input_folder : str
            path to the input folder containing the input data
        output_folder : str
            path to the output folder containing output images and annotation file
        """
        pass


class Point:
    """
    Representation of a point in 2D space.

    Attributes
    ----------
    x : int
        x coordinate
    y : int
        y coordinate

    Methods
    -------
    to_array()
        get the numpy array version of the point

    distance(other_point)
        calculate the distance between this point and another
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_array(self):
        return np.array([self.x, self.y])

    def distance(self, other_point):
        """Calculate the distance between this point and another

        Parameters
        ----------
        other_point : [x, y] or Point object

        Returns
        ------
        int
            the distance
        """

        p1 = self.to_array()
        if type(other_point) == Point:
            p2 = other_point.to_array()
        else:
            p2 = other_point  # array of [x, y]
        return np.linalg.norm(p1 - p2)


class Zone:
    """
    Representation of a zone of interest within an image.

    Attributes
    ----------
    points : list
        a list of 4 Point objects representing the bounding polygon
    cls : any
        the type of the zone
    image : numpy array
        the image within the bounding polygon
    text : str
        the text contained in the image
    metadata: dict
        other properties of the zone

    Methods
    -------
    sort_points()
        sort the 4 points of the zone in clockwise order from the top-left point

    get_points(flatten=False)
        get the numpy array version of the zone

    points_from_box(x, y, w, h)
        get 4 points of the polygon given top-left point (x, y) and box's width and height
    """
    def __init__(self, points, cls=0, image=None, text="", **kwargs):
        """
        Parameters
        ----------
        points : list
            a list of 4 Point objects representing the bounding polygon
            points order: top-left -> top-right -> bottom-right -> bottom-left
        cls : any, optional, default: 0
            the type of the zone
        image : numpy array, optional, default: None
            the image within the bounding polygon
        text : str, optional, default: ""
            the text contained in the image
        kwargs: keyword arguments
            other properties of the zone to be stored in metadata

        Raises
        ------
        ValueError
            if points are invalid
        """

        self.points = points
        self.cls = cls
        self.image = image
        self.text = text
        self.metadata = kwargs

        # check points to ensure there is no negative coordinates
        if not self._check_points_valid():
            raise ValueError('Points are invalid.')

    def _check_points_valid(self):
        if not self.points:
            return False

        for i, p in enumerate(self.points):
            if p.x < 0:
                self.points[i].x = 0
            if p.y < 0:
                self.points[i].y = 0
        return True

    def get_points(self, flatten=False):
        """Get the numpy array version of the zone

        Parameters
        ----------
        flatten : bool, optional, default: False
            whether to return a flat array or not

        Returns
        ------
        numpy array
            array version of the zone's coordinates
        """

        points = np.array([self.points[0].to_array(),
                           self.points[1].to_array(),
                           self.points[2].to_array(),
                           self.points[3].to_array()])
        if flatten:
            points = points.flatten()
        return points

    @staticmethod
    def points_from_box(x, y, w, h):
        p1 = Point(x, y)
        p2 = Point(x + w, y)
        p3 = Point(x + w, y + h)
        p4 = Point(x, y + h)
        return p1, p2, p3, p4

    def to_box(self):
        x = self.points[0].x
        y = self.points[0].y
        w = int(self.points[0].distance(self.points[1]))
        h = int(self.points[1].distance(self.points[2]))
        return x, y, w, h

    def area(self):
        _, _, w, h = self.to_box()
        return w * h


class Form:
    """
    Representation of an image which is a collection of zones

    Attributes
    ----------
    file_path : Path
        the path to the image
    cls : any
        the type of the form
    zones : list
        list of Zone objects
    image : numpy array
        the form image
    metadata: dict
        other properties of the zone

    Methods
    -------
    load_image()
        load the image from the path to self.image

    crop(zone)
        crop a zone in the form
    """
    def __init__(self, file_path=None, image=None, cls=0, zones=None, **kwargs):
        """
        Parameters
        ----------
        file_path : Path object
            the path to the image
        cls : any, optional, default: 0
            the type of the form
        zones : list, optional, default: []
            list of Zone objects
        image : numpy array
            the form image
        kwargs: keyword arguments
            other properties of the zone

        Raises
        ------
        ValueError
            can't load the image for some reason
        """
        if file_path is not None:
            self.file_path = Path(file_path)
        self.cls = cls
        self.image = image
        self.zones = zones if zones is not None else []
        self.metadata = kwargs

    def load_image(self):
        """Load the image from the path to self.image

        Raises
        ------
        ValueError
            can't load the image for some reason
        """
        if self.file_path is None:
            raise ValueError("File path not provided.")

        try:
            self.image = cv2.imdecode(np.fromfile(str(self.file_path), dtype=np.uint8), cv2.IMREAD_COLOR)  # BRG image
        except Exception:
            self.image = cv2.imread(str(self.file_path), cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError("Load image failed.")

    def crop(self, zone):
        """Crop a zone in the form.

        The cropped image is assigned to the zone itself.

        Parameters
        ----------
        zone : Zone object
            the zone to be cropped

        Raises
        ------
        ValueError
            The form's image is unavailable
        """

        if self.image is None:
            raise ValueError("Image not loaded.")

        poly = np.reshape(zone.get_points(), (4, 1, 2))
        rect = cv2.minAreaRect(poly)
        poly = poly.reshape((4, 2))

        box = cv2.boxPoints(rect)
        box = np.int0(box).reshape((4, 2))

        ordered_box = []
        for i in range(4):
            ordered_box.append(self._find_closet(Point(*poly[i]), box))
        src_pts = np.array(ordered_box).astype("float32")

        width = int(Point(*ordered_box[0]).distance(ordered_box[1]))
        height = int(Point(*ordered_box[0]).distance(ordered_box[3]))

        # coordinate of the points in box points after the rectangle has been straightened
        dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                           dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        zone.image = cv2.warpPerspective(self.image, M, (width, height))

    @staticmethod
    def _find_closet(point, box):
        closest = box[0]
        min_dist = point.distance(box[0])
        for i in range(1, 4):
            d = point.distance(box[i])
            if d < min_dist:
                min_dist = d
                closest = box[i]
        return closest


class Detector:
    """
    Detector identifies where the interested zones are.

    Attributes
    ----------
    config : dict
        depend on each subclass, this configuration can be anything

    Interfaces
    ----------
    detect(form)
        execute detection algorithm on a form
    """
    def __init__(self, **config):
        """
        Parameters
        ----------
        config : keyword arguments
            depend on each subclass, this configuration can be anything
        """
        self.config = config

    def detect(self, form):
        """Interface for the detection of zones within a form.

        Assign the detected zones to form.zones

        Parameters
        ----------
        form: Form object

        Raises
        ------
        ValueError
            if form.image is None
        """
        pass


class Recognizer:
    """
    Recognizer transcribe whatever in the zone's image into text.

    Attributes
    ----------
    config : dict
        depend on each subclass, this configuration can be anything

    Interfaces
    ----------
    recognize(zone)
        execute recognition algorithm on a zone
    """
    def __init__(self, **config):
        """
        Parameters
        ----------
        config : keyword arguments
            depend on each subclass, this configuration can be anything
        """
        self.config = config

    def image_to_text(self, image):
        """Interface for the recognition of an image.

        Parameters
        ----------
        image: numpy array
            OpenCV loaded image (BGR)

        Returns
        ------
        str
            recognized text
        float (optional)
            the confidence of the prediction

        Raises
        ------
        ValueError
            if image is None
        """
        pass

    def recognize(self, zone):
        """Interface for the recognition of a zone.

        Assign the recognized text to zone.text

        Parameters
        ----------
        zone: Zone object

        Raises
        ------
        ValueError
            if zone.image is None
        """
        pass


class Preprocess:
    """
    Preprocess runs before the execution of an algorithm

    Interfaces
    ----------
    run(inputs, **config)
        execute pre-processing algorithm
    """
    @staticmethod
    def run(inputs, **config):
        """Interface for pre-processing

        Parameters
        ----------
        inputs: any
            this could be anything
        config : dict
            depend on each subclass, this configuration can be anything

        Returns
        ------
        any
            the outputs could be anything
        """
        pass


class Postprocess:
    """
    Postprocess runs after the execution of an algorithm

    Interfaces
    ----------
    run(inputs, **config)
        execute post-processing algorithm
    """
    @staticmethod
    def run(inputs, **config):
        """Interface for post-processing

        Parameters
        ----------
        inputs: any
            this could be anything
        config : dict
            depend on each subclass, this configuration can be anything

        Returns
        ------
        any
            the outputs could be anything
        """
        pass
