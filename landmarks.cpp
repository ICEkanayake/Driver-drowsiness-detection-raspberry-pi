#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <gpiod.h>

struct gpiod_chip *chip;
struct gpiod_line *buzzerLine;
struct gpiod_line *ledLine;

using namespace cv;
using namespace std;
using Clock = chrono::steady_clock;

// EAR settings
const double earThreshold = 0.25;
const double closedSeconds = 2.0;

Clock::time_point closedStart;
bool eyesClosed = false;
bool drowsy = false;

// Calculate Eye Aspect Ratio
double calculateEAR(const vector<Point2f>& p,
                    int a, int b, int c,
                    int d, int e, int f)
{
    double vertical1 = norm(p[b] - p[f]);
    double vertical2 = norm(p[c] - p[e]);
    double horizontal = norm(p[a] - p[d]);

    if (horizontal < 1.0) {
        return 0.0;
    }

    return (vertical1 + vertical2) / (2.0 * horizontal);
}
int main()
{
// Open Raspberry Pi GPIO chip
chip = gpiod_chip_open("/dev/gpiochip0");

if (chip == NULL) {
    perror("Cannot open GPIO chip");
    return 1;
}

// Select GPIO17 (physical pin 11)
buzzerLine = gpiod_chip_get_line(chip, 17);

if (buzzerLine == NULL) {
    perror("Cannot access GPIO17");
    gpiod_chip_close(chip);
    return 1;
}

// Set GPIO17 as output; initially OFF
if (gpiod_line_request_output(
        buzzerLine, "drowsiness-buzzer", 0) < 0) {
    perror("Cannot set GPIO17 as output");
    gpiod_chip_close(chip);
    return 1;

}



    CascadeClassifier detector;

// ===== RED LED GPIO SETUP =====
ledLine = gpiod_chip_get_line(chip, 22);

if (ledLine == NULL) {
    perror("Cannot access GPIO22");
    gpiod_line_release(buzzerLine);
    gpiod_chip_close(chip);
    return 1;
}

if (gpiod_line_request_output(
        ledLine, "drowsiness-red-led", 0) < 0) {
    perror("Cannot set GPIO22 as output");
    gpiod_line_release(buzzerLine);
    gpiod_chip_close(chip);
    return 1;
}


    if (!detector.load("haarcascade_frontalface_default.xml")) {
        cerr << "Face cascade model not found!" << endl;
        return 1;
    }

    Ptr<face::Facemark> facemark = face::FacemarkLBF::create();

    try {
        facemark->loadModel("lbfmodel.yaml");
    } catch (const cv::Exception& e) {
        cerr << "Cannot load landmark model: "
             << e.what() << endl;
        return 1;
    }

    FILE* pipe = popen(
        "rpicam-vid -t 0 --codec yuv420 "
        "--width 640 --height 480 --nopreview -o - 2>/dev/null",
        "r"
    );

    if (!pipe) {
        cerr << "Cannot start camera!" << endl;
        return 1;
    }

    const int width = 640;
    const int height = 480;
    const size_t frameSize = width * height * 3 / 2;

    vector<unsigned char> buffer(frameSize);
    Mat yuv(height * 3 / 2, width, CV_8UC1);

    cout << "Drowsiness detection running. Press q to quit."
         << endl;

    while (fread(buffer.data(), 1, frameSize, pipe) == frameSize) {

        memcpy(yuv.data, buffer.data(), frameSize);

        Mat frame, gray;

        cvtColor(yuv, frame, COLOR_YUV2BGR_I420);
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;

        detector.detectMultiScale(
            gray, faces, 1.1, 4, 0, Size(80, 80)
        );

        bool validEAR = false;

        vector<vector<Point2f>> landmarks;

        if (!faces.empty() &&
            facemark->fit(frame, faces, landmarks)) {

            for (size_t i = 0; i < landmarks.size(); i++) {

                rectangle(
                    frame, faces[i],
                    Scalar(0, 255, 0), 2
                );

                const vector<Point2f>& p = landmarks[i];

                for (const Point2f& point : p) {
                    circle(
                        frame, point, 2,
                        Scalar(0, 0, 255), FILLED
                    );
                }

                if (p.size() >= 68) {

                    double leftEAR = calculateEAR(
                        p, 36, 37, 38, 39, 40, 41
                    );

                    double rightEAR = calculateEAR(
                        p, 42, 43, 44, 45, 46, 47
                    );

                    double ear = (leftEAR + rightEAR) / 2.0;

                    validEAR = true;
  // Check whether eyes are closed
                    if (ear < earThreshold) {

                        if (!eyesClosed) {
                            eyesClosed = true;
                            closedStart = Clock::now();
                        }

                        double elapsed =
                            chrono::duration<double>(
                                Clock::now() - closedStart
                            ).count();

                        
drowsy = (elapsed >= closedSeconds);

                    } else {
                        eyesClosed = false;
                        drowsy = false;
                    }


         // ===== BUZZER AND RED LED CONTROL =====
		if (drowsy) {
  			  gpiod_line_set_value(buzzerLine, 1);  // Buzzer ON
 			   gpiod_line_set_value(ledLine, 1);     // Red LED ON
		} else {
   			 gpiod_line_set_value(buzzerLine, 0);  // Buzzer OFF
   			 gpiod_line_set_value(ledLine, 0);     // Red LED OFF
		}
		   
 			// Display EAR value
                    putText(
                        frame, format("EAR: %.3f", ear),
                        Point(20, 35),
                        FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(255, 0, 0), 2
                    );

                    // Display eye status
                    putText(
                        frame,
                        ear < earThreshold
                            ? "Eyes possibly closed"
                            : "Eyes open",
                        Point(20, 70),
                        FONT_HERSHEY_SIMPLEX, 0.7,
                        ear < earThreshold
                            ? Scalar(0, 0, 255)
                            : Scalar(0, 255, 0),
                        2
                    );

                    // Display drowsiness warning
                    putText(
                        frame,
                        drowsy
                            ? "DROWSINESS WARNING!"
                            : "Monitoring",
                        Point(20, 105),
                        FONT_HERSHEY_SIMPLEX, 0.7,
                        drowsy
                            ? Scalar(0, 0, 255)
                            : Scalar(255, 255, 0),
                        2
                    );

                    // One face is enough for this project
                    break;
                }
            }
        }

        if (!validEAR) {
            eyesClosed = false;
            drowsy = false;


           	gpiod_line_set_value(buzzerLine, 0);
		gpiod_line_set_value(ledLine, 0);


			
		 putText(
                frame, "Face/landmarks not found",
                Point(20, 35),
                FONT_HERSHEY_SIMPLEX, 0.6,
                Scalar(0, 0, 255), 2
            );
        }

        imshow("Drowsiness Detection", frame);

        if (waitKey(1) == 'q') {
            break;
        }
    }

    pclose(pipe);
    destroyAllWindows();

 gpiod_line_release(buzzerLine);
    gpiod_line_release(ledLine);
    gpiod_chip_close(chip);

    return 0;
}
