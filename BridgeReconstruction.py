from laspy.file import File
import laspy
import time
import numpy as np
import random
#import pyflann as pf
import numpy as np
import shapefile
#import index
from scipy.spatial.kdtree import KDTree
import numpy as np
import pyrr
from pyrr import Matrix33, Vector3

def distanceBetwenTwoPoints(p0, p1):
    return np.linalg.norm(p0 - p1)

def writePointsToLasFile(points):
    # inFile = File("/path/to/lasfile", mode="r")
    #
    # # Get arrays which indicate VALID X, Y, or Z values.
    #
    # X_invalid = np.logical_and((inFile.header.min[0] <= inFile.x),
    #                            (inFile.header.max[0] >= inFile.x))
    # Y_invalid = np.logical_and((inFile.header.min[1] <= inFile.y),
    #                            (inFile.header.max[1] >= inFile.y))
    # Z_invalid = np.logical_and((inFile.header.min[2] <= inFile.z),
    #                            (inFile.header.max[2] >= inFile.z))
    # good_indices = np.where(np.logical_and(X_invalid, Y_invalid, Z_invalid))
    # good_points = inFile.points[good_indices]

    # output_file = File("data/out_0.las", mode="w", header=inFile.header)
    # #output_file.points = good_points
    # output_file.points = points
    # output_file.close()

    #hdr = laspy.header.Header()

    #outfile = laspy.file.File("data/out_example.las", mode="w", header=hdr)
    outfile = laspy.file.File("data/out_brudge_underside.las", mode="w", header=inFile.header)
    allx = np.array(points[0])
    ally = np.array(points[1])
    allz = np.array(points[2])

    xmin = np.floor(np.min(allx))
    ymin = np.floor(np.min(ally))
    zmin = np.floor(np.min(allz))

    outfile.header.offset = [xmin, ymin, zmin]
    outfile.header.scale = [1, 1, 1]
    #outfile.header.scale = [0.001, 0.001, 0.001]

    outfile.x = allx
    outfile.y = ally
    outfile.z = allz

    outfile.close()

def isBridgeInScope(bbox):
    bbox = [point*100 for point in bbox]
    bbox_x_0 = bbox[0]
    bbox_y_0 = bbox[1]
    bbox_x_1 = bbox[2]
    bbox_y_1 = bbox[3]
    return (bbox_x_0 > min_x and bbox_x_0 < max_x and
            bbox_y_0 > min_y and bbox_y_0 < max_y and
            bbox_x_1 > min_x and bbox_x_1 < max_x and
            bbox_y_1 > min_y and bbox_y_1 < max_y)

def getPointsInBbox(bbox):

    margin = 1000

    #points = [(random.random(), random.random()) for i in range(100)]
    points = [(x, y) for x, y in zip(inFile.X, inFile.Y)]

    pts = np.array(points)
    ll = np.array([bbox[0] * 100 - margin, bbox[1] * 100 - margin])  # lower-left
    ur = np.array([bbox[2] * 100 + margin, bbox[3] * 100 + margin])  # upper-right

    inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    #inbox = pts[inidx]
    #count_true = inidx.tolist().count(True)
    #inbox = inFile.points[inidx]
    #outbox = pts[np.logical_not(inidx)]

    #return inbox
    #return inidx
    return (inFile.X[inidx], inFile.Y[inidx], inFile.Z[inidx])


def getPointsWithinDistance(points, x, y, z):

    #all_points = inFile.points.copy()

    #print()
    #z=28836

    # xs = np.append(inFile.X[points], x*100)
    # ys = np.append(inFile.Y[points], y*100)
    # zs = np.append(inFile.Z[points], z)

    # xs = np.append(inFile.x.copy(), x)
    # ys = np.append(inFile.y.copy(), y)
    # zs = np.append(inFile.z.copy(), z)
    #
    # coords_0 = np.vstack((xs, ys, zs)).transpose()

    # xs = [point_x for point_x in points[0]].append(x)
    # ys = [point_y for point_y in points[1]].append(y)
    # zs = [point_z for point_z in points[2]].append(z)

    xs = np.append(points[0], x*100)
    ys = np.append(points[1], y*100)
    zs = np.append(points[2], z)

    #coords = np.vstack((inFile.x[:].append(x), inFile.y[:].append(y), inFile.z[:].append(z))).transpose()
    coords = np.vstack((xs, ys, zs)).transpose()

    first_point = coords[-1, :]
    #distances = np.sum((coords - first_point) ** 2, axis=1)
    distances = np.sqrt(np.sum((coords - first_point) ** 2, axis=1))
    distances.sort()
    d = np.flip(distances)
    keep_points = distances < 50 #radius

    c = keep_points.tolist().count(True)
    i_s = []
    for i, b in enumerate(keep_points.tolist()):
        if b: i_s.append((i, b))

    # Grab an array of all points which meet this threshold
    #points_kept = inFile.points[keep_points[:-1]]
    #points_kept = points[keep_points[:-1]]

    points_kept = (points[0][keep_points[:-1]], points[1][keep_points[:-1]], points[2][keep_points[:-1]])

    #print("We're keeping %i points out of %i total" % (len(points_kept[0]), len(points[0])))

    #return keep_points
    return points_kept

def getLocalMaxZ(points, path_point):
    print()
    print("searching for z of:")
    print("x: ", path_point[0])
    print("y: ", path_point[1])
    z_found = None
    for z in np.arange(max_z, min_z - 1, -10):
        close_points = getPointsWithinDistance(points, path_point[0], path_point[1], z)

        if len(close_points[0]) > 5: # number of points inside radius
        #if close_points.tolist().count(True) > 50:
            #writePointsToLasFile(inFile.points[close_points])
            #print("z: ", z)
            print("found z: ", (z-25)/100)
            z_found = (z-25)/100
            #print("i have 20 points within 100 units")
            break

    return z_found

print("Loading LAS file ...")
inFile = File('data/TM_463_104.las', mode='r')

#data = inFile.points.tolist()

# Grab a numpy dataset of our clustering dimensions:
#dataset = np.vstack([inFile.X, inFile.Y, inFile.Z]).transpose()
# Load only points in bbox


# print("Building the KD Tree...")
# # Build the KD Tree
# tree = KDTree(dataset)
#
# print("Query")
# neighbors = tree.query(dataset[100,], k = 5)
#
# print("Five nearest neighbors of point 100: ")
# print(neighbors[1])
# print("Distances: ")
# print(neighbors[0])

# all_nns = [[dataset[idx] for idx in nn_indices if idx != i] for i, nn_indices in enumerate(neighbors[1])]
# for nns in all_nns:
#     print(nns)

max_x = inFile.header.max[0] * 100
max_y = inFile.header.max[1] * 100
max_z = inFile.header.max[2] * 100
min_x = inFile.header.min[0] * 100
min_y = inFile.header.min[1] * 100
min_z = inFile.header.min[2] * 100

debelina_mostu = 1

print("Loading shp file ...")
with shapefile.Reader("shapefiles/TN_CESTE_L.shp") as shp:

    shapes = shp.shapes()
    fields = shp.fields
    records = shp.records()

    for i, shape in enumerate(shapes, 0):
        tipobj_ces = records[i][5]
        if (tipobj_ces in [3, 4, 5] and isBridgeInScope(shape.bbox)):

            points_in_bbox = getPointsInBbox(shape.bbox)
            #writePointsToLasFile(points_in_bbox)
            #exit()

            path_points = shape.points
            path_points_with_z = []

            for i in range(len(path_points)):
                path_points_with_z.append((path_points[i][0],
                                           path_points[i][1],
                                           getLocalMaxZ(points_in_bbox, path_points[i])
                                           ))

            sirces = records[i][6]
            sirvoz = records[i][7]
            id = records[i][0]

            new_points_for_underside_of_bridge = []

            for p0, p1 in zip(path_points_with_z[0:], path_points_with_z[1:]):

                v0 = Vector3(p0)
                v1 = Vector3(p1)
                v_smer = v1-v0
                v_smer_n = Vector3(pyrr.vector3.normalize(v_smer))

                distance_of_bridge_segment = distanceBetwenTwoPoints(np.array(p0), np.array(p1))

                current_point_on_path = v0
                next_point_on_path = v0 + (.25 * v_smer_n)  # gostota točk vzdolž mostu

                while (distance_of_bridge_segment+1.75 >= distanceBetwenTwoPoints(
                        np.array((next_point_on_path.x, next_point_on_path.y, next_point_on_path.z)),
                        p0)):

                    #print("new distance: ", distanceBetwenTwoPoints(
                    #     np.array((next_point_on_path.x, next_point_on_path.y, next_point_on_path.z)),
                    #     p0))

                    rotation_matrix_90_counterclockwise = Matrix33.from_z_rotation(np.pi/2)
                    v_smer_n_90_counterclockwise = rotation_matrix_90_counterclockwise * v_smer_n
                    new_point_on_left_edge = current_point_on_path + (v_smer_n_90_counterclockwise * (sirces * 3.5)) # polovica širine mostu
                    new_point_on_left_edge[2] -= debelina_mostu # debelina mostu

                    z = new_point_on_left_edge[2]
                    while (z <= current_point_on_path.z):
                        new_point = Vector3(new_point_on_left_edge).copy()
                        new_point[2] = z + 0.1 # gostota točk v višino mostu
                        new_points_for_underside_of_bridge.append(new_point)
                        z = new_point[2]

                    rotation_matrix_90_clockwise = Matrix33.from_z_rotation(-(np.pi/2))
                    v_smer_n_90_clockwise = rotation_matrix_90_clockwise * v_smer_n
                    new_point_on_right_edge = current_point_on_path + (v_smer_n_90_clockwise * (sirces * 3.5)) # polovica širine mostu
                    new_point_on_right_edge[2] -= debelina_mostu # debelina mostu

                    z = new_point_on_right_edge[2]
                    while (z <= current_point_on_path.z):
                        new_point = Vector3(new_point_on_right_edge).copy()
                        new_point[2] = z + 0.1 # gostota točk v višino mostu
                        new_points_for_underside_of_bridge.append(new_point)
                        z = new_point[2]

                    for i in np.arange(0.0, 1.0, 0.01): # gostota točk v širino mostu
                        new_points_for_underside_of_bridge.append(pyrr.vector3.interpolate(new_point_on_left_edge, new_point_on_right_edge, i))

                    current_point_on_path = next_point_on_path
                    next_point_on_path = next_point_on_path + (.25 * v_smer_n)

            l = np.array(new_points_for_underside_of_bridge).transpose() * 100

            print()
            writePointsToLasFile(l)
            exit()



            print()
            print("to je most tipa: ", tipobj_ces)
            print("bbox: ", shape.bbox)
            print("points: ", path_points)
            print("sirces: ", sirces)
            print("sirvoz: ", sirvoz)
            print("i: ", id)




    print()
    print(shp)

print()
