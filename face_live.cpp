
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdio>
#include <vector>
#include <cstring>

using namespace cv;
using namespace std;

int main()
{
    CascadeClassifier faceCascade;

    if (!faceCascade.load("haarcascade_frontalface_default.xml"))
    {
        cerr << "Face model not found!" << endl;
        return 1;
    }

    FILE* pipe = popen(
        "rpicam-vid -t 0 --codec yuv420 "
        "--width 640 --height 480 --nopreview -o - 2>/dev/null",
        "r"
    );

    if (!pipe)
    {
        cerr << "Cannot start camera!" << endl;
        return 1;
    }

    const int width = 640;
    const int height = 480;
    const size_t frameSize = width * height * 3 / 2;

    vector<unsigned char> buffer(frameSize);
    Mat yuv(height * 3 / 2, width, CV_8UC1);

    while (fread(buffer.data(), 1, frameSize, pipe) == frameSize)
    {
        memcpy(yuv.data, buffer.data(), frameSize);

        Mat frame, gray;
        cvtColor(yuv, frame, COLOR_YUV2BGR_I420);
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        faceCascade.detectMultiScale(
            gray, faces, 1.1, 4, 0, Size(40, 40)
        );

        for (const Rect& face : faces)
        {
            rectangle(frame, face, Scalar(0, 255, 0), 2);
            putText(frame, "Face Detected",
                    Point(face.x, face.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.7,
                    Scalar(0, 255, 0), 2);
        }

        imshow("Live Face Detection", frame);

        if (waitKey(1) == 'q')
            break;
    }

    pclose(pipe);
    destroyAllWindows();
    return 0;
}
