#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

#define PI 3.14159

const int soble_x[9] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
const int soble_y[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

const std::vector<cv::Point> nebirhood_8 = { cv::Point(-1, -1), cv::Point(-1, 0), cv::Point(-1, 1),
cv::Point(0, -1), cv::Point(0, 0), cv::Point(0, 1),
cv::Point(1, -1), cv::Point(1, 0), cv::Point(1, 1) };

//get the grad of the image, and padding is 1
void getGrad(const cv::Mat &input, cv::Mat &output)
{
	cv::Mat input_mat = cv::Mat::zeros(cv::Size(input.cols + 2, input.rows + 2), input.type());
	
	
	cv::Mat input_roi = input_mat(cv::Rect(1, 1, input.cols, input.rows));
	input.copyTo(input_roi);

	cv::Mat input_gray;
	output = cv::Mat::zeros(input_mat.size(), CV_8UC1);
	cv::Mat gradDirect = cv::Mat::zeros(input_mat.size(), CV_8UC1);
	if (input_mat.channels() == 3)
		cv::cvtColor(input_mat, input_gray, CV_BGR2GRAY);
	else
		input_gray = input_mat.clone();
	
	for (int i = 1; i < output.rows - 1; i++)
	{
		for (int j = 1; j < output.cols - 1; j++)
		{
			float data_x = 0, data_y = 0;
			for (int v = 0; v < nebirhood_8.size(); v++)
			{
				data_x += input_gray.at<uchar>(i + nebirhood_8[v].x, j + nebirhood_8[v].y) * soble_x[v];
				data_y += input_gray.at<uchar>(i + nebirhood_8[v].x, j + nebirhood_8[v].y) * soble_y[v];
			}
			output.at<uchar>(i, j) = sqrt(data_x * data_x*1.0 + data_y * data_y*1.0);
			float angle = atan2f(data_y, data_x) * 180 / PI;
			gradDirect.at<uchar>(i, j) = angle  < 0? angle : 360 + angle;
		}
	}
	cv::imshow("sobel grad", output);
	//cv::waitKey();
	//Non-Maximum Suppression
	for (int i = 1; i < output.rows - 1; i++)
	{
		for (int j = 1; j < output.cols - 1; j++)
		{
			if (output.at<uchar>(i, j) > 0)
			{
				int p_tl = output.at<uchar>(i - 1, j - 1);
				int p_tm = output.at<uchar>(i - 1, j );
				int p_tr = output.at<uchar>(i - 1, j + 1);
				int p_rm = output.at<uchar>(i ,    j + 1);
				int p_bl = output.at<uchar>(i + 1, j - 1);
				int p_bm = output.at<uchar>(i + 1, j);
				int p_br = output.at<uchar>(i + 1, j + 1);
				int p_lm = output.at<uchar>(i ,    j - 1);

				float weight = 1.f;
				float angle = gradDirect.at<uchar>(i, j);
				float data1 = 0.f;
				float data2 = 0.f;
				if ((angle >= 0 && angle < 45) ||
					(angle >= 180 && angle < 225))
				{
					weight = tan(angle);
					data1 = p_tr * weight + p_rm * (1 - weight);
					data2 = p_bl * weight + p_lm * (1 - weight);
					
				}
				if ((angle >= 45 && angle < 90) ||
					(angle >= 225 && angle < 270))
				{
					weight = 1 / tan(angle);
					data1 = p_tr * weight + p_tm * (1 - weight);
					data2 = p_bl * weight + p_bm * (1 - weight);
					
				}
				if ((angle >= 90 && angle < 135) ||
					(angle >= 270 && angle < 315))
				{
					weight = tan(angle - 90);
					data1 = p_tl * weight + p_tm * (1 - weight);
					data2 = p_br * weight + p_bm * (1 - weight);

				}
				if ((angle >= 135 && angle < 180) ||
					(angle >= 315 && angle < 360))
				{
					weight = tan(180 - angle);
					data1 = p_tl * weight + p_lm * (1 - weight);
					data2 = p_br * weight + p_rm * (1 - weight);
					
				}
				if (data1 > output.at<uchar>(i, j) || data2 > output.at<uchar>(i, j))
					output.at<uchar>(i, j) = 0;
			}
		}
	}
	cv::imshow("Non Maxnum", output);
	//cv::waitKey();
}


void doubleThreshold(cv::Mat &img, const int &hight_thres, const int &low_thres)
{
	for (int i = 1; i < img.rows - 1; i++)
	{
		for (int j = 1; j < img.cols - 1; j++)
		{
			if (img.at<uchar>(i, j) > hight_thres)
				img.at<uchar>(i, j) = 255;
			if (img.at<uchar>(i, j) < hight_thres && img.at<uchar>(i, j) > low_thres)
				img.at<uchar>(i, j) = 125;
		}
	}
}

int main()
{
	cv::Mat src = cv::imread("C:/Users/123/Pictures/Saved Pictures/4.jpg");
	cv::Mat dst;
	getGrad(src, dst);
	doubleThreshold(dst, 100, 50);
	cv::imshow("test", dst);
	cv::waitKey();
	

	return 0;
}
