class MaxBoxGen():


    def findMaxRect(data):
        """http://stackoverflow.com/a/30418912/5008845"""

        nrows, ncols = data.shape
        w = np.zeros(dtype=int, shape=data.shape)
        h = np.zeros(dtype=int, shape=data.shape)
        skip = 1
        area_max = (0, [])

        for r in range(nrows):
            for c in range(ncols):
                if data[r][c] == skip:
                    continue
                if r == 0:
                    h[r][c] = 1
                else:
                    h[r][c] = h[r - 1][c] + 1
                if c == 0:
                    w[r][c] = 1
                else:
                    w[r][c] = w[r][c - 1] + 1
                minw = w[r][c]
                for dh in range(h[r][c]):
                    minw = min(minw, w[r - dh][c])
                    area = (dh + 1) * minw
                    if area > area_max[0]:
                        area_max = (area, [(r - dh, c - minw + 1, r, c)])

        return area_max


    ########################################################################
    def residual(angle, data):
        nx, ny = data.shape
        M = cv2.getRotationMatrix2D(((nx - 1) / 2, (ny - 1) / 2), int(angle), 1)
        RotData = cv2.warpAffine(
            data, M, (nx, ny), flags=cv2.INTER_NEAREST, borderValue=1
        )
        rectangle = findMaxRect(RotData)

        return 1.0 / rectangle[0]


    ########################################################################
    def residual_star(args):
        return residual(*args)


    ########################################################################
    def get_rectangle_coord(angle, data, flag_out=None):
        nx, ny = data.shape
        M = cv2.getRotationMatrix2D(((nx - 1) / 2, (ny - 1) / 2), angle, 1)
        RotData = cv2.warpAffine(
            data, M, (nx, ny), flags=cv2.INTER_NEAREST, borderValue=1
        )
        rectangle = findMaxRect(RotData)

        if flag_out:
            return rectangle[1][0], M, RotData
        else:
            return rectangle[1][0], M


    ########################################################################
    def findRotMaxRect(
            data_in,
            flag_opt=False,
            flag_parallel=False,
            nbre_angle=10,
            flag_out=None,
            flag_enlarge_img=False,
            limit_image_size=300,
    ):
        """
        flag_opt     : True only nbre_angle are tested between 90 and 180
                            and a opt descent algo is run on the best fit
                    False 100 angle are tested from 90 to 180.
        flag_parallel: only valid when flag_opt=False. the 100 angle are run on multithreading
        flag_out     : angle and rectangle of the rotated image are output together with the rectangle of the original image
        flag_enlarge_img : the image used in the function is double of the size of the original to ensure all feature stay in when rotated
        limit_image_size : control the size numbre of pixel of the image use in the function.
                        this speeds up the code but can give approximated results if the shape is not simple
        """

        # time_s = datetime.datetime.now()

        # make the image square
        # ----------------
        nx_in, ny_in = data_in.shape
        if nx_in != ny_in:
            n = max([nx_in, ny_in])
            data_square = np.ones([n, n])
            xshift = int((n - nx_in) / 2)
            yshift = int((n - ny_in) / 2)
            if yshift == 0:
                data_square[xshift: (xshift + nx_in), :] = data_in[:, :]
            else:
                data_square[:, yshift: yshift + ny_in] = data_in[:, :]
        else:
            xshift = 0
            yshift = 0
            data_square = data_in

        # apply scale factor if image bigger than limit_image_size
        # ----------------
        if data_square.shape[0] > limit_image_size:
            data_small = cv2.resize(
                data_square, (limit_image_size, limit_image_size), interpolation=0
            )
            scale_factor = 1.0 * data_square.shape[0] / data_small.shape[0]
        else:
            data_small = data_square
            scale_factor = 1

        # set the input data with an odd number of point in each dimension to make rotation easier
        # ----------------
        nx, ny = data_small.shape
        nx_extra = -nx
        ny_extra = -ny
        if nx % 2 == 0:
            nx += 1
            nx_extra = 1
        if ny % 2 == 0:
            ny += 1
            ny_extra = 1
        data_odd = np.ones(
            [
                data_small.shape[0] + max([0, nx_extra]),
                data_small.shape[1] + max([0, ny_extra]),
            ]
        )
        data_odd[:-nx_extra, :-ny_extra] = data_small
        nx, ny = data_odd.shape

        nx_odd, ny_odd = data_odd.shape

        if flag_enlarge_img:
            data = (
                    np.zeros([2 * data_odd.shape[0] + 1, 2 * data_odd.shape[1] + 1])
                    + 1
            )
            nx, ny = data.shape
            data[
            nx / 2 - nx_odd / 2: nx / 2 + nx_odd / 2,
            ny / 2 - ny_odd / 2: ny / 2 + ny_odd / 2,
            ] = data_odd
        else:
            data = np.copy(data_odd)
            nx, ny = data.shape

        # print((datetime.datetime.now()-time_s).total_seconds()

        if flag_opt:
            myranges_brute = [
                (90.0, 180.0),
            ]
            coeff0 = np.array(
                [
                    0.0,
                ]
            )
            coeff1 = optimize.brute(
                residual, myranges_brute, args=(data,), Ns=nbre_angle, finish=None
            )
            popt = optimize.fmin(
                residual, coeff1, args=(data,), xtol=5, ftol=1.0e-5, disp=False
            )
            angle_selected = popt[0]

        else:
            rotation_angle = np.linspace(90, 180, 100 + 1)[:-1]
            args_here = []
            for angle in rotation_angle:
                args_here.append([angle, data])

            if flag_parallel:

                # set up a pool to run the parallel processing
                cpus = multiprocessing.cpu_count()
                pool = multiprocessing.Pool(processes=cpus)

                # then the map method of pool actually does the parallelisation

                results = pool.map(residual_star, args_here)

                pool.close()
                pool.join()

            else:
                results = []
                for arg in args_here:
                    results.append(residual_star(arg))

            argmin = np.array(results).argmin()
            angle_selected = args_here[argmin][0]
        rectangle, M_rect_max, RotData = get_rectangle_coord(
            angle_selected, data, flag_out=True
        )

        M_invert = cv2.invertAffineTransform(M_rect_max)
        rect_coord = [
            rectangle[:2],
            [rectangle[0], rectangle[3]],
            rectangle[2:],
            [rectangle[2], rectangle[1]],
        ]
        
        rect_coord_ori = []
        for coord in rect_coord:
            rect_coord_ori.append(
                np.dot(M_invert, [coord[0], (ny - 1) - coord[1], 1])
            )

        # transform to numpy coord of input image
        coord_out = []
        for coord in rect_coord_ori:
            coord_out.append(
                [
                    scale_factor * round(coord[0] - (nx / 2 - nx_odd / 2), 0)
                    - xshift,
                    scale_factor
                    * round((ny - 1) - coord[1] - (ny / 2 - ny_odd / 2), 0)
                    - yshift,
                ]
            )

        coord_out_rot = []
        coord_out_rot_h = []
        for coord in rect_coord:
            coord_out_rot.append(
                [
                    scale_factor * round(coord[0] - (nx / 2 - nx_odd / 2), 0)
                    - xshift,
                    scale_factor * round(coord[1] - (ny / 2 - ny_odd / 2), 0)
                    - yshift,
                ]
            )
            coord_out_rot_h.append(
                [
                    scale_factor * round(coord[0] - (nx / 2 - nx_odd / 2), 0),
                    scale_factor * round(coord[1] - (ny / 2 - ny_odd / 2), 0),
                ]
            )
        if flag_out is None:
            return coord_out
        elif flag_out == "rotation":
            return coord_out, angle_selected, coord_out_rot
        else:
            print("bad def in findRotMaxRect input. stop")
            pdb.set_trace()


    ######################################################
    def factors(n):
        return set(
            reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0),
            )
        )


    # test scale poly
    def scale_polygon(path, offset):
        center = centroid_of_polygon(path)
        path_temp = path
        for i in path_temp:
            if i[0] > center[0]:
                i[0] += offset
            else:
                i[0] -= offset
            if i[1] > center[1]:
                i[1] += offset
            else:
                i[1] -= offset
        return path_temp


    def area_of_polygon(x, y):
        """Calculates the signed area of an arbitrary polygon given its verticies
        http://stackoverflow.com/a/4682656/190597 (Joe Kington)
        http://softsurfer.com/Archive/algorithm_0101/algorithm_0101.htm#2D%20Polygons
        """
        area = 0.0
        for i in range(-1, len(x) - 1):
            area += x[i] * (y[i + 1] - y[i - 1])
        return area / 2.0


    def centroid_of_polygon(points):
        """
        http://stackoverflow.com/a/14115494/190597 (mgamba)
        """
        area = area_of_polygon(*zip(*points))
        result_x = 0
        result_y = 0
        N = len(points)
        points = IT.cycle(points)
        x1, y1 = next(points)
        for i in range(N):
            x0, y0 = x1, y1
            x1, y1 = next(points)
            cross = (x0 * y1) - (x1 * y0)
            result_x += (x0 + x1) * cross
            result_y += (y0 + y1) * cross
        result_x /= (area * 6.0)
        result_y /= (area * 6.0)
        return (result_x, result_y)


    def perimiter(points):
        """ returns the length of the perimiter of some shape defined by a list of points """
        distances = get_distances(points)
        width=min(distances)
        length = 0
        for distance in distances:
            length = length + distance

        return length, width


    def get_distances(points):
        """ convert a list of points into a list of distances """
        i = 0
        distances = []
        for i in range(len(points)):
            point = points[i]

            if i + 1 < len(points):
                next_point = points[i + 1]
            else:
                next_point = points[0]

            x0 = point[0]
            y0 = point[1]
            x1 = next_point[0]
            y1 = next_point[1]

            point_distance = get_distance(x0, y0, x1, y1)
            distances.append(point_distance)

        return distances


    def get_distance(x0, y0, x1, y1):
        """ use pythagorean theorm to find distance between 2 points """
        a = x1 - x0
        b = y1 - y0
        c_2 = a * a + b * b

        return c_2 ** (.5)

    def random_coord(origin_coord, threshold):
        new_coord = origin_coord
        points = []
        for row in origin_coord:
            x = row[0]
            y = row[1]
            points.append((float(x), float(y)))
        peri, width = perimiter(points)
        threshold *= peri
        if threshold >= width/2:
            threshold = math.floor(width/2)

        print(peri, width, threshold)

        #x1y1-top left
        new_coord[0][0]=random.uniform(origin_coord[0][0], origin_coord[0][0]+threshold)
        new_coord[0][1]=random.uniform(origin_coord[0][1]-threshold, origin_coord[0][1])

        # x2y2-top right
        new_coord[1][0] = random.uniform(origin_coord[1][0] - threshold, origin_coord[1][0])
        new_coord[1][1] = random.uniform(origin_coord[1][1] - threshold, origin_coord[1][1])

        # x3y3-bottom right
        new_coord[2][0] = random.uniform(origin_coord[2][0] - threshold, origin_coord[2][0])
        new_coord[2][1] = random.uniform(origin_coord[2][1], origin_coord[2][1] + threshold)

        # x4y4-bottom left
        new_coord[3][0] = random.uniform(origin_coord[3][0], origin_coord[3][0] + threshold)
        new_coord[3][1] = random.uniform(origin_coord[3][1], origin_coord[3][1] + threshold)

        return new_coord

    def draw_rect_size(image, resize_amount, random_threshold):
        
        a = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[::-1].T
        idx_in = np.where(a == 255)
        idx_out = np.where(a == 0)
        aa = np.ones_like(a)
        aa[idx_in] = 0

        # get coordinate of biggest rectangle
        # ----------------
        time_start = datetime.datetime.now()

        rect_coord_ori, angle, coord_out_rot = findRotMaxRect(
            aa,
            flag_opt=True,
            nbre_angle=4,
            flag_parallel=False,
            flag_out="rotation",
            flag_enlarge_img=False,
            limit_image_size=100,
        )

        print(
            "time elapsed =",
            (datetime.datetime.now() - time_start).total_seconds(),
        )
        print("angle        =", angle)

        rect_coord_for_scale = rect_coord_ori

        #change size with amount as wish
        new_coord_1 = scale_polygon(rect_coord_for_scale, resize_amount)

        #deflect by random new coord with threshold*perimeter
        random_poly_cord = random_coord(new_coord_1, random_threshold)
        
        return random_poly_cord