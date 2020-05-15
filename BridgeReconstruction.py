#!/usr/bin/python

import sys
import os
from laspy.file import File
import laspy
import shapefile
import pyrr
from pyrr import Matrix33, Vector3
import time
import multiprocessing as mp
import numpy as np

def distanceBetwenTwoPoints(p0, p1):
    return np.linalg.norm(p0 - p1)

def writePointsToLasFile(points, name):
    outfile = laspy.file.File("{}.las".format(os.path.join(out_path, name)), mode="w", header=inFile.header)

    allx = np.array(points[0])
    ally = np.array(points[1])
    allz = np.array(points[2])

    xmin = np.floor(np.min(allx))
    ymin = np.floor(np.min(ally))
    zmin = np.floor(np.min(allz))

    outfile.header.offset = [xmin, ymin, zmin]
    outfile.header.scale = [1, 1, 1]

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
    points = [(x, y) for x, y in zip(inFile.X, inFile.Y)]
    pts = np.array(points)
    ll = np.array([bbox[0] * 100 - margin, bbox[1] * 100 - margin])  # lower-left
    ur = np.array([bbox[2] * 100 + margin, bbox[3] * 100 + margin])  # upper-right
    inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    return (inFile.X[inidx], inFile.Y[inidx], inFile.Z[inidx])


def getPointsWithinDistance(points, x, y, z, radius):
    xs = np.append(points[0], x*100)
    ys = np.append(points[1], y*100)
    zs = np.append(points[2], z)

    coords = np.vstack((xs, ys, zs)).transpose()
    first_point = coords[-1, :]
    distances = np.sqrt(np.sum((coords - first_point) ** 2, axis=1))
    keep_points = distances < radius #radius
    points_kept = (points[0][keep_points[:-1]], points[1][keep_points[:-1]], points[2][keep_points[:-1]])

    return points_kept

def getLocalMaxZ(path_point_i, x, y, points, path_point):
    z_found = max_z/100
    maxx = max(points[2])

    for z in np.arange(maxx, min_z-1, -10):
        close_points = getPointsWithinDistance(points, path_point[0], path_point[1], z, 60)

        if len(close_points[0]) > 5: # number of points inside radius
            z_found = (z-25)/100
            break

    return (path_point_i, x, y, z_found)

def getLocalMinZ(points, path_point, plus=65.0, radius=100.0):
    z_found = None

    for z in np.arange(min_z, max_z-1, 10):
        close_points = getPointsWithinDistance(points, path_point[0], path_point[1], z, radius)
        if len(close_points[0]) > 5: # number of points inside radius
            z_found = (z+plus)/radius
            break

    if (not z_found):
        return getLocalMinZ(points, path_point, plus*1.1, radius*1.1)

    return z_found

def calculateSegment(p0, p1):

    new_points = []

    v0 = Vector3(p0)
    v1 = Vector3(p1)
    v_smer = v1 - v0
    v_smer_n = Vector3(pyrr.vector3.normalize(v_smer))

    distance_of_bridge_segment = distanceBetwenTwoPoints(np.array(p0), np.array(p1))

    current_point_on_path = v0
    next_point_on_path = v0 + (.25 / D * v_smer_n)  # gostota točk vzdolž mostu

    while (distance_of_bridge_segment + 1.75 >= distanceBetwenTwoPoints(
            np.array((next_point_on_path.x, next_point_on_path.y, next_point_on_path.z)),
            p0)):

        rotation_matrix_90_counterclockwise = Matrix33.from_z_rotation(np.pi / 2)
        v_smer_n_90_counterclockwise = rotation_matrix_90_counterclockwise * v_smer_n
        new_point_on_left_edge = current_point_on_path + (
                    v_smer_n_90_counterclockwise * (sirces / 1.5))  # polovica širine mostu
        new_point_on_left_edge[2] -= sirces / 11  # debelina mostu

        z = new_point_on_left_edge[2] - (0.2 / D)
        while (z <= current_point_on_path.z):
            new_point = Vector3(new_point_on_left_edge).copy()
            new_point[2] = z + (0.2 / D)  # gostota točk v višino mostu
            new_points.append(new_point)
            z = new_point[2]

        rotation_matrix_90_clockwise = Matrix33.from_z_rotation(-(np.pi / 2))
        v_smer_n_90_clockwise = rotation_matrix_90_clockwise * v_smer_n
        new_point_on_right_edge = current_point_on_path + (
                    v_smer_n_90_clockwise * (sirces / 1.5))  # polovica širine mostu
        new_point_on_right_edge[2] -= sirces / 11  # debelina mostu

        z = new_point_on_right_edge[2] - (0.2 / D)
        while (z <= current_point_on_path.z):
            new_point = Vector3(new_point_on_right_edge).copy()
            new_point[2] = z + (0.2 / D)  # gostota točk v višino mostu
            new_points.append(new_point)
            z = new_point[2]

        for point_i in np.arange(0.0, 1.0, 0.02 / D):  # gostota točk v širino mostu
            new_points.append(
                pyrr.vector3.interpolate(new_point_on_left_edge, new_point_on_right_edge, point_i))

        terrain_point_left = new_point_on_left_edge + (v_smer_n_90_counterclockwise * (1.3))
        terrain_point_right = new_point_on_right_edge + (v_smer_n_90_clockwise * (1.3))
        terrain_point_left_z = getLocalMinZ(points_in_bbox, terrain_point_left)
        terrain_point_right_z = getLocalMinZ(points_in_bbox, terrain_point_right)

        if (abs(new_point_on_left_edge.z - terrain_point_left_z) <= 1.25 or abs(
                new_point_on_right_edge.z - terrain_point_right_z) <= 1.25):
            new_z = min(terrain_point_left_z, terrain_point_right_z)
            terrain_point_left[2] = new_z
            terrain_point_right[2] = new_z
        else:
            terrain_point_left[2] = terrain_point_left_z
            terrain_point_right[2] = terrain_point_right_z

        for point_i in np.arange(0.0, 1.0, 0.02 / D):
            new_points.append(
                pyrr.vector3.interpolate(terrain_point_left, terrain_point_right, point_i))

        current_point_on_path = next_point_on_path
        next_point_on_path = next_point_on_path + ((.25 / D) * v_smer_n)

    return new_points

def collect_result_segment(result):
    global resulting_points
    resulting_points.append(result)

def collect_result_zs(result):
    global path_points_with_z
    path_points_with_z.append(result)

out_path = sys.argv[3]
if not os.path.exists(out_path):
    exit("Out file path does not exist!")
in_file = sys.argv[2]
D = int(sys.argv[1])/100.0 # gostota točk

resulting_points = []
path_points_with_z = []

start_time = time.time()

print("Loading LAS file ...")
inFile = File(in_file, mode='r')

max_x = inFile.header.max[0] * 100
max_y = inFile.header.max[1] * 100
max_z = inFile.header.max[2] * 100
min_x = inFile.header.min[0] * 100
min_y = inFile.header.min[1] * 100
min_z = inFile.header.min[2] * 100

debelina_mostu = 1
#D = 0.25 # gostota točk

print("Loading shp file ...")
with shapefile.Reader("shapefiles/TN_CESTE_L.shp") as shp:

    shapes = shp.shapes()
    fields = shp.fields
    records = shp.records()

    for shape_i, shape in enumerate(shapes, 0):
        tipobj_ces = records[shape_i][5]

        if (tipobj_ces in [3, 4, 5, 9] and isBridgeInScope(shape.bbox)):

            sirces = records[shape_i][6]
            sirvoz = records[shape_i][7]
            id = records[shape_i][0]
            path_points = shape.points

            print()
            print("type: ", tipobj_ces)
            print("bbox: ", shape.bbox)
            print("points: ", path_points)
            print("sirces: ", sirces)
            print("sirvoz: ", sirvoz)
            print("id: ", id)

            points_in_bbox = getPointsInBbox(shape.bbox)
            writePointsToLasFile(points_in_bbox, "bbox_{}".format(id))
            #exit()

            cpu_count = mp.cpu_count()
            pool_zs = mp.Pool(cpu_count)
            for path_point_i in range(len(path_points)):
                pool_zs.apply_async(getLocalMaxZ, (path_point_i, path_points[path_point_i][0], path_points[path_point_i][1], points_in_bbox, path_points[path_point_i]), callback=collect_result_zs)
            pool_zs.close()
            pool_zs.join()

            path_points_with_z.sort(key=lambda x: x[0])
            path_points_with_z = [(x, y, z) for i, x, y, z in path_points_with_z]

            pool_segments = mp.Pool(cpu_count)
            for p0, p1 in zip(path_points_with_z[0:], path_points_with_z[1:]):
                pool_segments.apply_async(calculateSegment, (p0,p1), callback=collect_result_segment)
            pool_segments.close()
            pool_segments.join()

            t = []
            for point_list in resulting_points:
                t.extend(point_list)

            l = np.array(t).transpose() * 100

            print()
            print("writing to new_points_{}".format(id))
            writePointsToLasFile(l, "new_points_{}_{}".format(id, int(D*100)))
            print("--- it took %s seconds ---" % ((time.time() - start_time)))
            #exit()
            resulting_points = []
            path_points_with_z = []

    print(shp)
    print("--- it took %s seconds ---" % (time.time() - start_time))