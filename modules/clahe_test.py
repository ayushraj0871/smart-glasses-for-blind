# ─────────────────────────────────────────────────
# Smart Glasses for the Blind
# Module: CLAHE Low-Light Enhancement Test
# ─────────────────────────────────────────────────
# Shows your webcam feed in 3 windows side by side:
#   Left  — Original grayscale
#   Right — CLAHE enhanced
# Press Q to quit

import cv2

def run_clahe_test():

    # Create CLAHE object
    # clipLimit controls contrast enhancement strength
    # tileGridSize controls how local the enhancement is
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Open laptop webcam (0 = default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        print("Make sure your laptop webcam is not being used by another app")
        return

    print("Webcam opened successfully")
    print("Two windows will appear — Original vs CLAHE Enhanced")
    print("Try covering the camera slightly to simulate low light")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read frame")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE enhancement
        enhanced = clahe.apply(gray)

        # Add text labels to each window
        gray_labelled     = gray.copy()
        enhanced_labelled = enhanced.copy()

        cv2.putText(gray_labelled,
                    "ORIGINAL",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, 0, 2)

        cv2.putText(enhanced_labelled,
                    "CLAHE ENHANCED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, 255, 2)

        # Show both windows
        cv2.imshow("Original Grayscale", gray_labelled)
        cv2.imshow("CLAHE Enhanced", enhanced_labelled)

        # Also show colour original for reference
        cv2.imshow("Colour Feed (Reference)", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting CLAHE test")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test complete")


if __name__ == "__main__":
    run_clahe_test()