from skimage.metrics import structural_similarity
import argparse
import imutils
import cv2

# For reading arguments passed from the command line
# arg_parser = argparse.ArgumentParser()

# arg_parser.add_argument("-f", "--first", required=True, help="the first input image")
# arg_parser.add_argument("-s", "--second", required=True, help="the second input image")
# args = vars(arg_parser.parse_args())

#imageA = cv2.imread(args["first"])
#imageB = cv2.imread(args["second"])

# Read directly from the file system

imageA = cv2.imread('Images/cat_bw.png')
imageB = cv2.imread('AfterSampling2000000.png')

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned

(score, diff) = structural_similarity(imageA, imageB, full=True, multichannel=True)

diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
cv2.imshow("Difference", diff)
cv2.waitKey(0)