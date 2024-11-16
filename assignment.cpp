#include <cstdio>
#include <cmath>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>

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


// image와 shape 4개를 Convolution하는 함수
Mat Convolution(const int shape_num, Mat image, Mat& hitMap, const Mat shape[4]) {
	////////////////// parameters //////////////////////
	/**/		   int di = 4, dj = 4;				/**/
	/**/		float threshold = 0.25f; 			/**/
	/**/		float cos_thresh = 0.90f;			/**/
	////////////////////////////////////////////////////

	Mat LUT = Mat::zeros(image.size(), CV_32SC4);
	
	int image_height = image.rows;
	int image_width = image.cols;

	// shape 자기자신을 convolution하여 이상적인 convolution값 구하기
	float perf[4][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
	for (int k = 0; k < 4; k++) {
		for (int j = 0; j < shape[k].rows; j += dj) {
			for (int i = 0; i < shape[k].cols; i += di) {
				Vec4b Color = shape[k].at<Vec4b>(j, i);
				if (Color[0] >= 200 && Color[1] >= 200 && Color[2] >= 200) continue;
				float c0 = Color[0], c1 = Color[1], c2 = Color[2];
				perf[k][0] += c0 * c0;
				perf[k][1] += c1 * c1;
				perf[k][2] += c2 * c2;
			}
		}
	}

	// 사진 4개를 각각 image와 convolution하여 결과 구하기
	for (int k = 0; k < 4; k++) {
		Mat shp = shape[k];
		for (int y = 0; y < image_height - shp.rows; y++) {
			if(y % 30 == 0) printf("도형 %d - 조각 %d: %.0f%% \n", shape_num, k + 1, (float)y / (float)(image_height - shp.rows) * 100);
			for (int x = 0; x < image_width - shp.cols; x++) {
				float summ[3] = { 0, 0, 0 };
				for (int j = 0; j < shp.rows; j+= dj) {
					for (int i = 0; i < shp.cols; i+= di) {

						Vec4b image_pixel = image.at<Vec4b>(y + shp.rows - j, x + shp.cols - i);
						Vec4b shape_pixel = shp.at<Vec4b>(j, i);

						if ((image_pixel[0] >= 200 && image_pixel[1] >= 200 && image_pixel[2] >= 200)
							|| (shape_pixel[0] >= 200 && shape_pixel[1] >= 200 && shape_pixel[2] >= 200)) continue;

						long long im0 = image_pixel[0], im1 = image_pixel[1], im2 = image_pixel[2];
						long long sh0 = shape_pixel[0], sh1 = shape_pixel[1], sh2 = shape_pixel[2];
						summ[0] += im0 * sh0;
						summ[1] += im1 * sh1;
						summ[2] += im2 * sh2;
					}
				}
				float ratio[3];
				for (int p = 0; p < 3; p++) ratio[p] = summ[p] / perf[k][p];
				if (k == 0) {
					if (y - shp.rows < 0 || x - shp.cols < 0) continue;
					hitMap.at<float>(y - shp.rows, x - shp.cols) += cos_similarity(summ, perf[k]);
				}
				else if (k == 1) {
					if (y - shp.rows < 0) continue;
					hitMap.at<float>(y - shp.rows, x) += cos_similarity(summ, perf[k]);
				}
				else if (k == 2) {
					if (x - shp.cols < 0) continue;
					hitMap.at<float>(y, x - shp.cols) += cos_similarity(summ, perf[k]);
				}
				else if (k == 3) hitMap.at<float>(y, x) += cos_similarity(summ, perf[k]);
				
				if(abs(ratio[0] -1) <= threshold && abs(ratio[1] - 1) <= threshold && abs(ratio[2] - 1) <= threshold
					&& cos_similarity(summ, perf[k]) >= cos_thresh) {
					if (k == 0) {
						if (y - shp.rows < 0 || x - shp.cols < 0) continue;
						(LUT.at<Vec4f>(y - shp.rows, x - shp.cols))[k] = 1;
					}
					else if (k == 1) {
						if (y - shp.rows < 0) continue;
						(LUT.at<Vec4f>(y - shp.rows, x))[k] = 1;
					}
					else if (k == 2) {
						if (x - shp.cols < 0) continue;
						(LUT.at<Vec4f>(y, x - shp.cols))[k] = 1;
					}
					else if (k == 3) (LUT.at<Vec4f>(y, x))[k] = 1;
				}
			}
		}
	}
	return LUT;
}

Mat Check_Same_Image(const int shape_num, Mat& image, const Mat& shape) {
	int img_height = image.rows;
	int img_width = image.cols;

	int shp_height = shape.rows;
	int shp_width = shape.cols;

	int half_height = shape.rows / 2;
	int half_width = shape.cols / 2;
	
	// 2. 네 개의 부분으로 나누기
	Mat B[4];
	B[0] = shape(cv::Rect(0, 0, half_width, half_height));           // 좌측 상단
	B[1] = shape(cv::Rect(half_width, 0, half_width, half_height));    // 우측 상단
	B[2] = shape(cv::Rect(0, half_height, half_width, half_height));   // 좌측 하단
	B[3] = shape(cv::Rect(half_width, half_height, half_width, half_height)); // 우측 하단
	
	// 각 부분에 대하여 convoltuion 진행
	// 4개의 부분이 모두 convolution 값이 높게 나온다면 같은 사진으로 판단
	// Convolution 진행
	Mat hitMap = Mat::zeros(image.size(), CV_32FC1);
	Mat LUT = Convolution(shape_num, image, hitMap, B); // Look Up Table


	for (int y = 0; y <= img_height - shp_height; y++) {
		for (int x = 0; x <= img_width - shp_width; x++) {
			if (y + shp_height >= img_height || x + shp_width >= img_width) continue;
			Vec4f lut = LUT.at<Vec4f>(y, x);

			if (lut == Vec4f(1, 1, 1, 1)) {
				if (shape_num == 1) rectangle(image, Point(x, y), Point(x + shp_width, y + shp_height), Scalar(0, 0, 255), 5);
				else if (shape_num == 2) rectangle(image, Point(x, y), Point(x + shp_width, y + shp_height), Scalar(0, 255, 0), 5);
				else if (shape_num == 3) rectangle(image, Point(x, y), Point(x + shp_width, y + shp_height), Scalar(255, 0, 0), 5);
				else continue; // eror
			}
		}
	}

	float maxi = 0, mini = 1;
	for (int y = 0; y < hitMap.rows; y++) {
		for (int x = 0; x < hitMap.cols; x++) {
			float hitmap_value = hitMap.at<float>(y, x)/ 4;
			if (hitmap_value == 0.0f) continue;
			if (hitmap_value > maxi) maxi = hitmap_value;
			if (hitmap_value < mini) mini = hitmap_value;
		}
	}

	// Contrast stretching
	for (int y = 0; y < hitMap.rows; y++) {
		for (int x = 0; x < hitMap.cols; x++) {
			float hitmap_value = hitMap.at<float>(y, x) / 4;
			if (hitmap_value == 0.0f) continue;
			float new_value = (hitmap_value - mini) / (maxi - mini) * 255;
			hitMap.at<float>(y, x) = new_value;
		}
	}
	hitMap.convertTo(hitMap, CV_8UC1);
	return hitMap;
}


int main() {
	Mat image = cv::imread("Image.JPG", cv::IMREAD_UNCHANGED);
	Mat Shape1 = cv::imread("Shape1.JPG", cv::IMREAD_UNCHANGED);
	Mat Shape2 = cv::imread("Shape2.JPG", cv::IMREAD_UNCHANGED);
	Mat Shape3 = cv::imread("Shape3.JPG", cv::IMREAD_UNCHANGED);
	if (image.empty() || Shape1.empty() || Shape2.empty() || Shape3.empty()) {
		printf("사진이 존재하지 않습니다\n");
		return 0;
	}
	flip(Shape1, Shape1, -1);
	flip(Shape2, Shape2, -1);
	flip(Shape3, Shape3, -1);
	cv::imshow("Image", image);
	cv::imshow("Shape1", Shape1);
	cv::imshow("Shape2", Shape2);
	cv::imshow("Shape3", Shape3);
	cv::waitKey(0);
	cv::destroyAllWindows();
	
	Mat hitMap1 = Check_Same_Image(1, image, Shape1);
	Mat hitMap2 = Check_Same_Image(2, image, Shape2);
	Mat hitMap3 = Check_Same_Image(3, image, Shape3);
	
	cv::imshow("convoled Image", image);
	cv::waitKey(0);
	cv::imwrite("Output.JPG", image);
	cv::imwrite("hitMap1.JPG", hitMap1);
	cv::imwrite("hitMap2.JPG", hitMap2);
	cv::imwrite("hitMap3.JPG", hitMap3);
	return 0;
}