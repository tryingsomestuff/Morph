import argparse
import cv2
import dlib
import numpy as np
import os
from PIL import Image
import random
from subprocess import Popen, PIPE


class NoFaceFound(Exception):
    pass


class MoreThanOneFaceFound(Exception):
    pass


def shape_helper(imgA, imgB):
    dimA = imgA.shape
    dimB = imgB.shape
    diff0 = abs(dimA[0] - dimB[0]) // 2
    diff1 = abs(dimA[1] - dimB[1]) // 2
    avg0 = (dimA[0] + dimB[0]) // 2
    avg1 = (dimA[1] + dimB[1]) // 2

    return [dimA, dimB, diff0, diff1, avg0, avg1]


def crop_helper(imgA, imgB):
    [size1, size2, diff0, diff1, avg0, avg1] = shape_helper(imgA, imgB)

    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [imgA, imgB]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        return [imgA, imgB[abs(diff0):avg0, abs(diff1):avg1]]

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        return [imgA[diff0:avg0, diff1:avg1], imgB]

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        return [imgA[diff0:avg0, :], imgB[:, abs(diff1):avg1]]

    else:
        return [imgA[:, diff1:avg1], imgB[abs(diff0):avg0, :]]


def crop(imgA, imgB):
    [dimA, dimB, diff0, diff1, avg0, avg1] = shape_helper(imgA, imgB)

    # if image of same size just return
    if(dimA[0] == dimB[0] and dimA[1] == dimB[1]):
        return [imgA, imgB]

    # if A is smaller in both dimension
    elif(dimA[0] <= dimB[0] and dimA[1] <= dimB[1]):
        # compute scale factors
        scale0 = dimA[0]/dimB[0]
        scale1 = dimA[1]/dimB[1]
        # apply the bigger one to scale B
        if(scale0 > scale1):
            scaled = cv2.resize(imgB, None, fx=scale0,
                                fy=scale0, interpolation=cv2.INTER_AREA)
        else:
            scaled = cv2.resize(imgB, None, fx=scale1,
                                fy=scale1, interpolation=cv2.INTER_AREA)
        # and crop
        return crop_helper(imgA, scaled)

    # if B is smaller in both dimension
    elif(dimA[0] >= dimB[0] and dimA[1] >= dimB[1]):
        # compute scale factors
        scale0 = dimB[0]/dimA[0]
        scale1 = dimB[1]/dimA[1]
        # apply the bigger one to scale A
        if(scale0 > scale1):
            scaled = cv2.resize(imgA, None, fx=scale0,
                                fy=scale0, interpolation=cv2.INTER_AREA)
        else:
            scaled = cv2.resize(imgA, None, fx=scale1,
                                fy=scale1, interpolation=cv2.INTER_AREA)
        # and crop
        return crop_helper(scaled, imgB)

    # A is bigger in 0 direction but smaller in 1 direction, let's crop ...
    elif(dimA[0] >= dimB[0] and dimA[1] <= dimB[1]):
        return [imgA[diff0:avg0, :], imgB[:, abs(diff1):avg1]]

    # A is smaller in 0 direction but bigger in 1 direction, let's crop ...
    else:
        return [imgA[:, diff1:avg1], imgB[abs(diff0):avg0, :]]


def face_matching(imgA, imgB):
    """
    Takes two images and returns
    dim = (n0,n1) -> size of the cropped images
    img_list[0] -> cropped A image
    img_list[1] -> corpped B image
    pointsA = [ (x,y), ... ] -> list of control point of image A (68 + 8)
    pointsB = [ (x,y), ... ] -> list of control point of image B (68 + 8)
    middles = [ (x,y), ... ] -> list of middle points between pointsA and pointsB (68 + 8)
    """
    print("Loading model")
    detector = dlib.get_frontal_face_detector()
    directory = os.path.dirname(os.path.abspath(__file__))
    predictor = dlib.shape_predictor(os.path.join(
        directory, 'data/shape_predictor_68_face_landmarks.dat'))

    print("Cropping images")
    img_list = crop(imgA, imgB)

    #cv2.imwrite('cropA.jpg', img_list[0])
    #cv2.imwrite('cropB.jpg', img_list[1])

    pointsA = []
    pointsB = []

    j = 1

    for img in img_list:
        print("Working on image {}".format(j))
        dim = (img.shape[0], img.shape[1])

        points = pointsA if j == 1 else pointsB
        j = j+1

        # Ask the detector to find the bounding boxes of each face.
        print("...Detecting BB")
		# The second parameter is the number of image pyramid layers to apply when 
		# upscaling the image prior to applying the detector (this it the equivalent 
		# of computing cv2.pyrUp N number of times on the image).
		#
		# The benefit of increasing the resolution of the input image prior to face 
		# detection is that it may allow us to detect more faces in the image.
		# T he downside is that the larger the input image, the more computaitonally 
		# expensive the detection process is.		
        faces = detector(img, 2)

        # Multiple tries with various upsampling to get a face ... #dirty ;)
        if len(faces) == 0:
            faces = detector(img, 1)
            if len(faces) == 0:
                faces = detector(img, 3)
                if len(faces) == 0:
                    raise NoFaceFound

        if len(faces) != 1:
            raise MoreThanOneFaceFound

        print("Found {} faces".format(len(faces)))

        for k, face in enumerate(faces):
            print("...Working on face {}".format(k))

            # Get the control points.
            print("...Predictor")
            landmarks = predictor(img, face)

            print("...Adding control points")
            # Gather all the control points coords
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                points.append((x, y))

            # Add 8 more control points on the image border
            # One at each corner, and one in the middle of each side
            points.append((1, 1))
            points.append((dim[1]-1, 1))
            points.append(((dim[1]-1)//2, 1))
            points.append((1, dim[0]-1))
            points.append((1, (dim[0]-1)//2))
            points.append(((dim[1]-1)//2, dim[0]-1))
            points.append((dim[1]-1, dim[0]-1))
            points.append(((dim[1]-1), (dim[0]-1)//2))

    assert(len(pointsA) == len(pointsB))

    # half-way points (will be used to build triangulation)
    middles = []
    for i in range(len(pointsA)):
        middles.append(((pointsA[i][0] + pointsB[i][0])/2,
                       (pointsA[i][1] + pointsB[i][1])/2))

    return [dim, img_list[0], img_list[1], pointsA, pointsB, middles]


def triangulate(w, h, points):
    points = [(int(x[0]), int(x[1])) for x in points]
    # map the control points to theirs number
    number_map = {x[0]: x[1] for x in zip(points, range(len(points)))}

    box = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(box)
    for p in points:
        subdiv.insert(p)
    triangles = subdiv.getTriangleList()

    connectivity = []
    for t in triangles:
        p1 = (int(t[0]), int(t[1]))
        p2 = (int(t[2]), int(t[3]))
        p3 = (int(t[4]), int(t[5]))
        connectivity.append((number_map[p1], number_map[p2], number_map[p3]))

    return connectivity


def apply_transformation(src_img, src_tri, dst_tri, dim):
    affine_trans = cv2.getAffineTransform(
        np.float32(src_tri), np.float32(dst_tri))
    dst_img = cv2.warpAffine(src_img, affine_trans, (
        dim[0], dim[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst_img


def morph_triangle(imgA, imgB, img, tA, tB, t, alpha):

    # Bounding box
    bbA = cv2.boundingRect(np.float32([tA]))
    bbB = cv2.boundingRect(np.float32([tB]))
    bb = cv2.boundingRect(np.float32([t]))

    # Translate so that top left corner of the BB is (0,0)
    tA_translated = [(tA[i][0] - bbA[0], tA[i][1] - bbA[1]) for i in range(3)]
    tB_translated = [(tB[i][0] - bbB[0], tB[i][1] - bbB[1]) for i in range(3)]
    t_translated = [(t[i][0] - bb[0], t[i][1] - bb[1]) for i in range(3)]

    # Build mask of the target triangle (will use it as a multiplier)
    mask = np.zeros((bb[3], bb[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_translated), (1.0, 1.0, 1.0), 16, 0)
    morph_triangle.counter += 1
    #if morph_triangle.counter == 1:
    #    cv2.imwrite("mask_{}.jpg".format(morph_triangle.counter), mask)

    # Extract patches from full image
    patchA = imgA[bbA[1]:bbA[1] + bbA[3], bbA[0]:bbA[0] + bbA[2]]
    patchB = imgB[bbB[1]:bbB[1] + bbB[3], bbB[0]:bbB[0] + bbB[2]]
    dim = (bb[2], bb[3])
    # Apply deformation to patches
    imgA_transformed = apply_transformation(patchA, tA_translated, t_translated, dim)
    imgB_transformed = apply_transformation(patchB, tB_translated, t_translated, dim)

    # Alpha blend the patches together
    blended = (1.0 - alpha) * imgA_transformed + alpha * imgB_transformed

    # Copy patch in output image but only where mask == 1 (inside the triangle)
    img[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]] = img[bb[1]:bb[1] + bb[3], bb[0]:bb[0]+bb[2]] * (1 - mask) + blended * mask


morph_triangle.counter = 0


def morph_sequence(duration, frame_rate,
                   imgA, imgB,
                   pointsA, pointsB,
                   connectivity,
                   dim, output_file,
                   with_triangle=False):

    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(frame_rate), '-s', str(dim[1])+'x'+str(dim[0]), '-i', '-', '-c:v',
              'libx264', '-crf', '25', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-pix_fmt', 'yuv420p', output_file], stdin=PIPE)

    imgA = np.float32(imgA)
    imgB = np.float32(imgB)

    nb_img = int(duration*frame_rate)
    for j in range(0, nb_img):

        alpha = j/(nb_img-1)

        # Get current frame points positions
        # linear interpolated between pointsA and pointsB
        points = []
        for i in range(0, len(pointsA)):
            x = (1 - alpha) * pointsA[i][0] + alpha * pointsB[i][0]
            y = (1 - alpha) * pointsA[i][1] + alpha * pointsB[i][1]
            points.append((x, y))

        frame = np.zeros(imgA.shape, dtype=imgA.dtype)
        for i in range(len(connectivity)):
            n1 = connectivity[i][0]
            n2 = connectivity[i][1]
            n3 = connectivity[i][2]

            triA = [pointsA[n1], pointsA[n2], pointsA[n3]]
            triB = [pointsB[n1], pointsB[n2], pointsB[n3]]
            tri = [points[n1], points[n2], points[n3]]

            # Morph current triangle
            morph_triangle(imgA, imgB, frame, triA, triB, tri, alpha)

            if with_triangle:
                p1 = (int(tri[0][0]), int(tri[0][1]))
                p2 = (int(tri[1][0]), int(tri[1][1]))
                p3 = (int(tri[2][0]), int(tri[2][1]))
                cv2.line(frame, p1, p2, (255, 255, 255), 1, 8, 0)
                cv2.line(frame, p2, p3, (255, 255, 255), 1, 8, 0)
                cv2.line(frame, p3, p1, (255, 255, 255), 1, 8, 0)

        frame_image = Image.fromarray(
            cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2RGB))
        # save to pipe
        frame_image.save(p.stdin, 'JPEG')

    p.stdin.close()
    p.wait()


def morph(imgA, imgB, duration, frame_rate, output_file, with_triangle=False):

	print("Generating face match")
	[dim, imgA, imgB, pointsA, pointsB, middles] = face_matching(imgA, imgB)
	print("Building triangulation (of middle points)")
	connectivity = triangulate(dim[1], dim[0], middles)
	print("Morphing...")
	morph_sequence(duration, frame_rate, imgA, imgB, pointsA,
	               pointsB, connectivity, dim, output_file, with_triangle)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	# two image mode
	parser.add_argument("--imgA", help="Image A")
	parser.add_argument("--imgB", help="Image B")
	# directory mode
	parser.add_argument("--dir", help="Directory with many images")
	parser.add_argument("--duration", type=int, default=5,
	                    help="Duration of each animation")
	parser.add_argument("--frame", type=int, default=24, help="Frame Rate")
	parser.add_argument("--output", help="Output video name")
	parser.add_argument("--outdir", default="./",
	                    help="Output directory (for multi-image processing)")
	parser.add_argument("--with_triangle", action='store_true',
	                    help="Display triangulation")
	parser.add_argument("--shuffle", action='store_true',
	                    help="Shuffle image order in directory mode")
	args = parser.parse_args()

	if(args.imgA and args.imgB):
		print("Treating " + args.imgA + " and " + args.imgB)
		imgA = cv2.imread(args.imgA)
		imgB = cv2.imread(args.imgB)
		morph(imgA, imgB, args.duration, args.frame, args.output, args.with_triangle)

	if(args.dir):
		print("Scanning " + args.dir)
		img_list = sorted(os.listdir(args.dir))
		if ( args.shuffle):
			img_list = random.shuffle(img_list)

		file_names = []
		for i in range(0, len(img_list)-1):
			print("Treating " + img_list[i] + " and " + img_list[i+1])
			imgA = cv2.imread(os.path.join(args.dir, img_list[i]))
			imgB = cv2.imread(os.path.join(args.dir, img_list[i+1]))
			out_path = args.outdir + str(i) + "_" + args.output
			morph(imgA, imgB, args.duration, args.frame, out_path, args.with_triangle)
			file_names.append("file '" + out_path + "'")

		with open('video_list.txt', 'w') as f:
			f.write('\n'.join(file_names))
			f.close()
			print("To fuse all video, use : ffmpeg -f concat -safe 0 -i video_list.txt -c copy output.mp4")

	print("done !")
