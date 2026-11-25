
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstring>

using namespace cv;
using namespace std;

int main()
{
    CascadeClassifier detector;

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

    cout << "Facial landmarks running. Press q to quit."
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

        vector<vector<Point2f>> landmarks;

        if (!faces.empty() && facemark->fit(frame, faces, landmarks)) {
            for (size_t i = 0; i < landmarks.size(); i++) {
                rectangle(frame, faces[i], Scalar(0, 255, 0), 2);

                for (const Point2f& point : landmarks[i]) {
                    circle(frame, point, 2, Scalar(0, 0, 255), FILLED);
                }
            }
        }

        imshow("Facial Landmarks", frame);

        if (waitKey(1) == 'q') {
            break;
        }
    }

    pclose(pipe);
    destroyAllWindows();
    return 0;
}
