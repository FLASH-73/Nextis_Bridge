
import cv2
import sys

def test(arg):
    print(f"Testing cv2.VideoCapture({repr(arg)})...")
    cap = cv2.VideoCapture(arg)
    if cap.isOpened():
        print(" -> SUCCESS")
        cap.release()
    else:
        print(" -> FAILED")

if __name__ == "__main__":
    # Test int 4
    test(4)
    # Test string "/dev/video4"
    test("/dev/video4")
    # Test string "/dev/video4" with backend V4L2 (200)
    print("Testing with CAP_V4L2...")
    cap = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L2)
    if cap.isOpened():
        print(" -> SUCCESS")
        cap.release()
    else:
        print(" -> FAILED")
