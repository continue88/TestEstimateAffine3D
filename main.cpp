#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
const int ImageSize = 512;
const int BoardSize = 10;

/// <summary>
/// EstimateAffine3D
/// </summary>
/// <param name="from">来源顶点，浮点数组（xyz|xyz|...)排列[0,size]， 这里为了保留提供给C#接口，统一成byte类型数组。</param>
/// <param name="to">目标顶点，浮点数组（xyz|xyz|...)排列[0,size]， 这里为了保留提供给C#接口，统一成byte类型数组。</param>
/// <param name="output">输出矩阵，浮点数组（xyz|xyz|...)排列[0,12]，这里为了保留提供给C#接口，统一成byte类型数组。</param>
/// <param name="output">有多少个顶点</param>
/// <returns></returns>
int EstimateAffine3D(unsigned char* from, unsigned char* to, unsigned char* output, int size)
{
	Mat aff_est;						// 这是计算出来的结果矩阵，为[3X4]double类型的矩阵。
	std::vector<uchar> inliers;			// 采样点？目前没用到。
	Mat fpts(1, size, CV_32FC3, from);	// 把from内存数据对应到fpts上面去，不申请新的内存。
	Mat tpts(1, size, CV_32FC3, to);	// 把to内存数据对应到tpts上面去，不申请新的内存。
	//Mat test(1, size, CV_32FC3);

	int ret = estimateAffine3D(fpts, tpts, aff_est, inliers);
	if (!aff_est.empty())
	{
		Mat out_mat(3, 4, CV_32F, output);	// 把output内存对应到out_mat，以便输出数据。
		aff_est.convertTo(out_mat, CV_32F); // 由于aff_est是64位double类型，需要转换为32位float类型，以便输出。
		//transform(fpts, test, aff_est);
	}
	return ret;
}

/// <summary>
/// ReadDataFromCsv
/// 从文本文件中读取输入数据，每行为3个浮点数，以分号【,】分割
///	在空格之后，为目标数据。
/// </summary>
/// <param name="fileName">文件名</param>
/// <param name="fromPoints">输入的顶点</param>
/// <param name="toPoints">输出的顶点</param>
/// <returns></returns>
int ReadDataFromCsv(const std::string& fileName, std::vector<Point3f>& fromPoints, std::vector<Point3f>& toPoints)
{
	std::ifstream inputFile(fileName);
	if (!inputFile.good())
	{
		std::cout << "Fail to open file:" << fileName << std::endl;
		return -1;
	}

	int mode = 0;
	char split = ','; // 逗号分隔符
	Point3f point3f;
	std::string line;
	while (!inputFile.eof())
	{
		std::getline(inputFile, line);
		if (line.empty())
		{
			mode++;
			continue;
		}

		std::istringstream iss(line);
		iss >> point3f.x >> split >> point3f.y >> split >> point3f.z;
		if (mode == 0)
			fromPoints.push_back(point3f);
		else if (mode == 1)
			toPoints.push_back(point3f);
		else // next data is not used.
			break;
	}
	return 0;
}


/// <summary>
/// 画一根线出来
/// </summary>
/// <param name="img">绘制的图像</param>
/// <param name="points">线的点</param>
/// <param name="color">颜色</param>
/// <param name="scale">点需要的缩放</param>
/// <param name="thickness">线段粗细</param>
/// <param name="base">偏移</param>
void DrawLines(Mat img, const std::vector<Point3f>& points, Scalar color, float scale, float thickness, Point3f base)
{
	Point prePoint((points[0].x - base.x) * scale, (points[0].z - base.z) * scale);
	for (size_t i = 1; i < points.size(); i++)
	{
		Point curPoint((points[i].x - base.x) * scale, (points[i].z - base.z) * scale);
		line(img,
			prePoint,
			curPoint,
			color,
			thickness,
			8);
		prePoint = curPoint;
	}
}

/// <summary>
/// BuildTheShowGraph.
///	显示出误差图形。由于高度Y的值变化较小，忽略掉吧，只显示X，Z两个轴。
/// </summary>
/// <param name="fromPoints">输入的顶点</param>
/// <param name="fromPoints">输出的顶点</param>
/// <param name="targetPoints">计算出来的顶点</param>
int BuildTheShowGraph(const std::vector<Point3f>& fromPoints, const std::vector<Point3f>& toPoints, const std::vector<Point3f>& targetPoints)
{
	// 计算出最大/最小值，用来规范边界。
	Point3f minPoint(fromPoints[0]), maxPoint(fromPoints[0]);
	for (size_t i = 0; i < fromPoints.size(); i++)
	{
		Point3f p0 = fromPoints[i], p1 = toPoints[i], p2 = targetPoints[i];
		minPoint.x = min(min(min(minPoint.x, p0.x), p1.x), p2.x);
		minPoint.y = min(min(min(minPoint.y, p0.y), p1.y), p2.y);
		minPoint.z = min(min(min(minPoint.z, p0.z), p1.z), p2.z);
		maxPoint.x = max(max(max(maxPoint.x, p0.x), p1.x), p2.x);
		maxPoint.y = max(max(max(maxPoint.y, p0.y), p1.y), p2.y);
		maxPoint.z = max(max(max(maxPoint.z, p0.z), p1.z), p2.z);
	}
	float scale = 1 / max(maxPoint.x - minPoint.x, maxPoint.z - minPoint.z);
	float imageScale = (ImageSize - BoardSize * 2) * scale;
	minPoint.x -= BoardSize / imageScale;
	minPoint.z -= BoardSize / imageScale;

	Mat imageColor = cv::Mat(ImageSize, ImageSize, CV_8UC4);

	// draw the lines.
	DrawLines(imageColor, fromPoints, Scalar(0, 0, 255), imageScale, 2, minPoint);
	DrawLines(imageColor, toPoints, Scalar(0, 255, 255), imageScale, 2, minPoint);
	DrawLines(imageColor, targetPoints, Scalar(0, 255, 0), imageScale, 1, minPoint);

	imshow("colorMap", imageColor);
	waitKey(0);
	return 0;
}

/// <summary>
/// MainEntry.
/// </summary>
int main()
{
	// read the data from text file.
	std::vector<Point3f> fromPoints, toPoints, targetPoints;
	ReadDataFromCsv("test.csv", fromPoints, toPoints);

	// build the data.
	float matrix[3 * 4];
	int ret = EstimateAffine3D(
		(unsigned char*)&fromPoints[0],
		(unsigned char*)&toPoints[0],
		(unsigned char*)&matrix[0],
		(int)fromPoints.size());

	// calculate the target points;
	Mat mat(3, 4, CV_32F, matrix);
	transform(fromPoints, targetPoints, mat);

	// show the result.
	BuildTheShowGraph(fromPoints, toPoints, targetPoints);

	return 0;
}