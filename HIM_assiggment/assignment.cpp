#include <cstdio>
#include <cmath>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#define INF 99999

using namespace std;
using namespace cv;

float cos_similarity(float x[3], float y[3]) {
	float z = 0, n1 = 0, n2 = 0;
	for (int i = 0; i < 3; i++) {
		z += x[i] * y[i];
		n1 += x[i] * x[i];
		n2 += y[i] * y[i];
	}
	if (n1 == 0 || n2 == 0) return 0;
	return z / (sqrt(n1) * sqrt(n2));
}

void Convolution(const int shape_num, Mat& image, const Mat& shape, const char color) {
	////////////////// parameters //////////////////////
	/**/		   int di = 8, dj = 8;				/**/
	/**/		float threshold = 0.10f;			/**/
	////////////////////////////////////////////////////
	int height = image.rows;
	int width = image.cols;

	int shape_height = shape.rows;
	int shape_width = shape.cols;

	// shape 자기자신을 convolution하여 이상적인 convolution값 구하기
	float perf[3] = { 0, 0, 0 };
	for (int j = 0; j < shape_height; j+=dj) {
		for (int i = 0; i < shape_width; i+=di) {
			Vec4b Color = shape.at<Vec4b>(j, i);
			if (Color[0] >= 200 && Color[1] >= 200 && Color[2] >= 200) continue;
			float c0 = Color[0], c1 = Color[1], c2 = Color[2];
			perf[0] += c0 * c0;
			perf[1] += c1 * c1;
			perf[2] += c2 * c2;
		}
	}

	// pixel 단위로 convolution 값을 구하고, perfect와의 오차가 threshold 안으로 들어오면 사각형 그리기
	Mat result = Mat::zeros(image.size(), CV_32FC4);
	float standard[3] = { -1, -1, -1 };
	for (int y = 0; y <= height - shape_height; y++) {
		printf("shape %d: (%d / %d)\n", shape_num, y, height - shape_height);
		for (int x = 0; x <= width - shape_width; x++) {
			float summ[3] = { 0, 0, 0 };
			for (int j = 0; j < shape_height; j += dj) {
				for (int i = 0; i < shape_width; i += di) {
					Vec4b image_pixel = image.at<Vec4b>(y + j, x + i);
					Vec4b shape_pixel = shape.at<Vec4b>(j, i);
					if ((image_pixel[0] >= 200 && image_pixel[1] >= 200 && image_pixel[2] >= 200)
						|| (shape_pixel[0] >= 200 && shape_pixel[1] >= 200 && shape_pixel[2] >= 200)) continue;

					long long im0 = image_pixel[0], im1 = image_pixel[1], im2 = image_pixel[2];
					long long sh0 = shape_pixel[0], sh1 = shape_pixel[1], sh2 = shape_pixel[2];
					summ[0] += im0 * sh0;
					summ[1] += im1 * sh1;
					summ[2] += im2 * sh2;
				}
			}
			result.at<Vec3f>(y, x) = Vec3f{ (float)summ[0], (float)summ[1], (float)summ[2] };

			//if (cos_similarity(perf, summ) >= threshold) {
			if(abs(summ[0] - perf[0]) <= threshold * perf[0] && abs(summ[1] - perf[1]) <= threshold * perf[1] && (summ[2] - perf[2]) <= threshold * perf[2]) {
				if (color == 'R') rectangle(image, Point(x, y), Point(x + shape_width, y + shape_height), Scalar(0, 0, 255), 5);
				else if (color == 'G') rectangle(image, Point(x, y), Point(x + shape_width, y + shape_height), Scalar(0, 255, 0), 5);
				else if (color == 'B') rectangle(image, Point(x, y), Point(x + shape_width, y + shape_height), Scalar(255, 0, 0), 5);
				else return; // error
			}
		}
	}
}


int main() {
	int test_num = 1;
	String directory = "./Test" + to_string(test_num);
	Mat image = cv::imread(directory + "/Image.JPG", cv::IMREAD_UNCHANGED);
	Mat Shape1 = cv::imread(directory + "/Shape1.JPG", cv::IMREAD_UNCHANGED);
	Mat Shape2 = cv::imread(directory + "/Shape2.JPG", cv::IMREAD_UNCHANGED);
	Mat Shape3 = cv::imread(directory + "/Shape3.JPG", cv::IMREAD_UNCHANGED);
	if (image.empty() || Shape1.empty() || Shape2.empty() || Shape3.empty()) {
		printf("사진이 존재하지 않습니다\n");
		return 0;
	}
	cv::imshow("Image", image);
	cv::imshow("Shape1", Shape1);
	cv::imshow("Shape2", Shape2);
	cv::imshow("Shape3", Shape3);
	cv::waitKey(0);
	//cv::destroyAllWindows();
	
	//printf("%d, %d, %d, %d\n", height, width, shape_height, shape_width);
	Convolution(1, image, Shape1, 'R');
	Convolution(2, image, Shape2, 'G');
	Convolution(3, image, Shape3, 'B');

	cv::imshow("convoled Image", image);
	cv::waitKey(0);
	return 0;
}