// --------------------------------------------------------------------
// Kalman Filter Demonstrating, a 2-d ball demo
// --------------------------------------------------------------------
#include <iostream>

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

const int winHeight = 600;
const int winWidth = 800;

Point mousePosition = Point(winWidth >> 1, winHeight >> 1);

// mouse event callback
void mouseEvent(int event, int x, int y, int flags, void *param)
{
	if (event == CV_EVENT_MOUSEMOVE) {
		mousePosition = Point(x, y);
	}
}

void TestKF();

int main()
{
	TestKF();

    return 0;
}


void TestKF()
{
	int stateNum = 7;
	int measureNum = 4;
	KalmanFilter kf = KalmanFilter(stateNum, measureNum, 0);

	// initialization
	Mat processNoise(stateNum, 1, CV_32F);
	Mat measurement = Mat::zeros(measureNum, 1, CV_32F);

	kf.transitionMatrix = (Mat_<float>(stateNum, stateNum) <<
		1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1);

	setIdentity(kf.measurementMatrix);
	// if the process noise covariance is large (comparing to measurement covariance)
	// 	the predictor tend to believe more the measurement rather than the perdiction
	setIdentity(kf.processNoiseCov, Scalar::all(1e-1));

	// if the measurement noise covariance is very large, then the 
	// 	predictor tend to believe more the measurement rather than the prediction
	setIdentity(kf.measurementNoiseCov, Scalar::all(1e-2));

	// initialize the first state covariance matrix to all 1
	setIdentity(kf.errorCovPost, Scalar::all(1));

	// randomly generate a state point to initialize
	randn(kf.statePost, Scalar::all(0), Scalar::all(winHeight));

	namedWindow("Kalman");
	setMouseCallback("Kalman", mouseEvent);
	Mat img(winHeight, winWidth, CV_8UC3);

	while (1)
	{
		// predict
		Mat prediction = kf.predict();
		Rect predictPt = Rect(
			prediction.at<float>(0, 0), prediction.at<float>(1, 0),
			prediction.at<float>(2, 0), prediction.at<float>(3, 0));

		// generate measurement
		Point statePt = mousePosition;
		measurement.at<float>(0, 0) = statePt.x;
		measurement.at<float>(1, 0) = statePt.y;
		measurement.at<float>(2, 0) = 96;
		measurement.at<float>(3, 0) = 112;
		Rect2i measuer_rect(
			measurement.at<float>(0, 0), measurement.at<float>(1, 0),
			measurement.at<float>(2, 0), measurement.at<float>(3, 0)
		);

		// update
		const Mat statePost = kf.correct(measurement);
		Rect2i state_post(
			statePost.at<float>(0, 0), statePost.at<float>(1, 0),
			statePost.at<float>(2, 0), statePost.at<float>(3, 0));
		

		// visualization
		img.setTo(Scalar(255, 255, 255));
		rectangle(img, predictPt, CV_RGB(0, 255, 0), 8); // predicted point as green
		// measurement
		rectangle(img, measuer_rect, CV_RGB(255, 0, 0), 6); // current position as red
		// corrected
		rectangle(img, state_post, Scalar(255, 0, 0), 4);

		imshow("Kalman", img);
		char code = (char) waitKey(100);
		if (code == 27 || code == 'q' || code == 'Q')
			break;
	}
	destroyWindow("Kalman");
}
