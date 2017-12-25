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
/// <param name="from">��Դ���㣬�������飨xyz|xyz|...)����[0,size]�� ����Ϊ�˱����ṩ��C#�ӿڣ�ͳһ��byte�������顣</param>
/// <param name="to">Ŀ�궥�㣬�������飨xyz|xyz|...)����[0,size]�� ����Ϊ�˱����ṩ��C#�ӿڣ�ͳһ��byte�������顣</param>
/// <param name="output">������󣬸������飨xyz|xyz|...)����[0,12]������Ϊ�˱����ṩ��C#�ӿڣ�ͳһ��byte�������顣</param>
/// <param name="output">�ж��ٸ�����</param>
/// <returns></returns>
int EstimateAffine3D(unsigned char* from, unsigned char* to, unsigned char* output, int size)
{
	Mat aff_est;						// ���Ǽ�������Ľ������Ϊ[3X4]double���͵ľ���
	std::vector<uchar> inliers;			// �����㣿Ŀǰû�õ���
	Mat fpts(1, size, CV_32FC3, from);	// ��from�ڴ����ݶ�Ӧ��fpts����ȥ���������µ��ڴ档
	Mat tpts(1, size, CV_32FC3, to);	// ��to�ڴ����ݶ�Ӧ��tpts����ȥ���������µ��ڴ档
	//Mat test(1, size, CV_32FC3);

	int ret = estimateAffine3D(fpts, tpts, aff_est, inliers);
	if (!aff_est.empty())
	{
		Mat out_mat(3, 4, CV_32F, output);	// ��output�ڴ��Ӧ��out_mat���Ա�������ݡ�
		aff_est.convertTo(out_mat, CV_32F); // ����aff_est��64λdouble���ͣ���Ҫת��Ϊ32λfloat���ͣ��Ա������
		//transform(fpts, test, aff_est);
	}
	return ret;
}

/// <summary>
/// ReadDataFromCsv
/// ���ı��ļ��ж�ȡ�������ݣ�ÿ��Ϊ3�����������Էֺš�,���ָ�
///	�ڿո�֮��ΪĿ�����ݡ�
/// </summary>
/// <param name="fileName">�ļ���</param>
/// <param name="fromPoints">����Ķ���</param>
/// <param name="toPoints">����Ķ���</param>
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
	char split = ','; // ���ŷָ���
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
/// ��һ���߳���
/// </summary>
/// <param name="img">���Ƶ�ͼ��</param>
/// <param name="points">�ߵĵ�</param>
/// <param name="color">��ɫ</param>
/// <param name="scale">����Ҫ������</param>
/// <param name="thickness">�߶δ�ϸ</param>
/// <param name="base">ƫ��</param>
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
///	��ʾ�����ͼ�Ρ����ڸ߶�Y��ֵ�仯��С�����Ե��ɣ�ֻ��ʾX��Z�����ᡣ
/// </summary>
/// <param name="fromPoints">����Ķ���</param>
/// <param name="fromPoints">����Ķ���</param>
/// <param name="targetPoints">��������Ķ���</param>
int BuildTheShowGraph(const std::vector<Point3f>& fromPoints, const std::vector<Point3f>& toPoints, const std::vector<Point3f>& targetPoints)
{
	// ��������/��Сֵ�������淶�߽硣
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