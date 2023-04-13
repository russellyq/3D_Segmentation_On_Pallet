#include <ros/ros.h>
#include <time.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud.h>
#include <laser_geometry/laser_geometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <tf/transform_listener.h>
#include <pcl/common/common.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose2D.h>
#include <pcl/common/transforms.h>
#include <serialrotor/serialrotorposservice.h>
#include <serialrotor/serialrotorstatus.h>
#include <vector>
#include <algorithm>
#include <pcl/filters/passthrough.h>
#include <palletunloader/Capture3Dservice.h>
#include <palletunloader/pallet_report_srv.h>
#include <palletunloader/which_pallet_srv.h>
#include <palletunloader/pallet_report_msg.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <string>     // std::string, std::to_string
#include <sensor_msgs/point_cloud2_iterator.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <ios>      // std::ofstream
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
float StartSwipAngle = 90; // -100 degrees
float EndSwipAngle = 105; // +100 degrees
float MinCaptureAngle = 92; // from -91 degrees
float MaxCaptureAngle = +100; // to -91 degrees
float HomeAngle = +90; //
int HomeSpeed = 45;
int RotationSpeed = 254;
int RotationStepSize = 7;
float MinSquareLineLenght = 0.001; // Square of the min acceptable line length
float MaxSquareLineLenght = 0.250; // Square of the max acceptable line length
unsigned int MininPointsCount = 0; // Minimum Number of Points in a point cloud segment
float AllowedPointDistance = 0.01; // in meters, For extracting a line out of a point cloud segment. This determines the maximum allowed distance of points from the line we extracted from this point cloud segment
float ToleranceFromLongLine = 0.05; // in meters, For extracting a line out of a point cloud segment. This determines the maximum allowed distance of points from the line we extracted from this point cloud segment
float MinPointDistanceTolerance = 0.05; // in meters; Max distance between consequent points in a segment
float PalletpillarThicknessMax = 0.12; // In meters, max thickness of pillar
float ForkWindowWidth = 0.62; // In meters, Approx pillar to pillar distance
float ForkWindowWidthTolerance = 0.10;
float MidPilarPointFromSidePilarLineDistanceMax = 0.05;
float SensorElevation = 0.30;
float PalletWidthFor3DAnalysis = 1.2; //meters
float PalletWidthToleranceFor3DAnalysis = 0.1; //meters
float PalletAngleToleranceFor3DAnalysis = 10; //Degrees
float PalletCenterToleranceFor3DAnalysis = 0.10; //meters
float GroupingMaxDistance = 0.3;// meters
bool Publish3D = true;
bool PublishRvizPalletLine = true;
ros::Subscriber Camerasub;
ros::Subscriber LASERsub;
ros::Subscriber RotorStatussb;
ros::ServiceClient client;
ros::Publisher pub3D;
ros::Publisher vis_pub;
ros::Publisher visArray_Pub;
ros::Publisher Projectedlaserscan;
ros::Publisher ReportPublisher;
vector<float> PosVector ;
vector<sensor_msgs::LaserScan> Scans ;
bool NewPosAvailable = false;
float NewPos = 0;
string ScansFileName = "";
string PosFileName = "";
bool ShouldReadScansFromFile = false;
bool IsDebugging = false;
sensor_msgs::CompressedImage ImageToReport;

enum PalletDetectionState
{
  PDS_ServiceRequested = 0,
  PDS_GoingToHomePosition = 1,
  PDS_GoingStartSweepPosition = 2,
  PDS_Scanning = 3,
  PDS_ScanComplete = 4,
  PDS_Processing = 5,
  PDS_ProcessingComplete = 6,
  PDS_Responded = 7,
};

PalletDetectionState CurrentPalletDetectionState = PDS_Responded;

struct PointCloudSegmentStruct
{
    unsigned int StartIndex;
    unsigned int EndIndex;
};

struct LineSegmentStruct
{
    PointCloudSegmentStruct PointCloudSegment;
    float StartX;
    float StartY;
    float StartZ;
    float EndX;
    float EndY;
    float EndZ;
};

struct ConfirmLineStruct
{
    bool IsConfirmed;
    LineSegmentStruct ConfirmedLine;
};

template <typename ForwardIt> ForwardIt Next(ForwardIt it, typename std::iterator_traits<ForwardIt>::difference_type n = 1)
{
    std::advance(it, n);
    return it;
}

template <typename T> std::string ToString(T val)
{
    std::stringstream stream;
    stream << val;
    return stream.str();
}

class DetectionClass
{
    public:
    DetectionClass();
    void LASERCallback (const sensor_msgs::LaserScan::ConstPtr&);
    void RotorStatusCallback(const serialrotor::serialrotorstatus::ConstPtr&);
    std::list<PointCloudSegmentStruct> GetPointCloudSegments(const sensor_msgs::LaserScan& , float, unsigned int );
    geometry_msgs::Point ConvertToCartesian(int , float , float , float);
    geometry_msgs::Point ProjectPointOnLine(geometry_msgs::Point ,float ,float , geometry_msgs::Point );
    std::list <LineSegmentStruct> GetLineSegments(const sensor_msgs::LaserScan& , float , float ,unsigned int );
    float LineLenght(LineSegmentStruct );
    float DistanceFromPointToLine (geometry_msgs::Point , geometry_msgs::Point , geometry_msgs::Point );
    float PointToPointDistance(geometry_msgs::Point , geometry_msgs::Point );
    std::vector <LineSegmentStruct> DetectPalletFromThreePillarFrontFaceLines(std::list<LineSegmentStruct> );
    sensor_msgs::PointCloud2 ShowOutputCloud (vector<sensor_msgs::LaserScan> );
    void DrawLine(ros::Publisher* , LineSegmentStruct );
    void DrawLines(ros::Publisher* , std::vector <LineSegmentStruct> , float);
    ConfirmLineStruct Further_3Dto2D_Analyze(LineSegmentStruct, pcl::PointCloud<pcl::PointXYZ> );
    sensor_msgs::LaserScan PointCloud2LASERScan(sensor_msgs::PointCloud2);
    vector <LineSegmentStruct> OutputResults (vector <LineSegmentStruct>);
    vector <LineSegmentStruct> DetectPallets();


    geometry_msgs::Pose2D DetectPallet();
    void SweepAndScan();

 };

DetectionClass::DetectionClass()
{
}

void* RotorServiceCaller (void* Dummy)
{
  float PrevPos = 0;
  PosVector.clear();
  Scans.clear();


  serialrotor::serialrotorposservice srv;
    srv.request.SetpointPosDegree = HomeAngle;
    srv.request.Speed = HomeSpeed;
    std::cout << "-------------------------------------------"<<std::endl;
    std::cout << "Going To Home Position ..." << std::endl;
    ROS_INFO(" Calling Rotor service, Pos = [%6.3f] degree, Speed = [%d]." , srv.request.SetpointPosDegree , srv.request.Speed);
    CurrentPalletDetectionState = PDS_GoingToHomePosition;
    if (client.call(srv))
    {
      ROS_INFO("Rotor Service Completed Successfully.");
      std::cout << "In Home Position." << std::endl;
    }
    else
    {
      ROS_ERROR("Failed to call rotor service.");
    }

  srv.request.SetpointPosDegree = StartSwipAngle;
  srv.request.Speed = HomeSpeed;
  std::cout << "-------------------------------------------"<<std::endl;
  std::cout << "Going to Start Sweep Position..." << std::endl;
  ROS_INFO(" Calling Rotor service, Pos = [%6.3f] degree, Speed = [%d]." , srv.request.SetpointPosDegree , srv.request.Speed);
  CurrentPalletDetectionState = PDS_GoingStartSweepPosition;
  if (client.call(srv))
  {
    ROS_INFO("Rotor Service Completed Successfully.");
    std::cout << "Ready to Scan." << std::endl;
  }
  else
  {
    ROS_ERROR("Failed to call rotor service.");
  }

  srv.request.SetpointPosDegree = MaxCaptureAngle;
  srv.request.Speed = RotationSpeed;
  std::cout << "-------------------------------------------"<<std::endl;
  std::cout << "Scanning..." << std::endl;
  ROS_INFO(" Calling Rotor service, Pos = [%6.3f] degree, Speed = [%d]." , srv.request.SetpointPosDegree , srv.request.Speed);
  CurrentPalletDetectionState = PDS_Scanning;
  if (client.call(srv))
  {
    ROS_INFO("Rotor Service Completed Successfully.");
  }
  else
  {
    ROS_ERROR("Failed to call rotor service.");
  }

  srv.request.SetpointPosDegree = HomeAngle;
  srv.request.Speed = HomeSpeed;
  std::cout << "-------------------------------------------"<<std::endl;
  std::cout << "Returning To Home Position ..." << std::endl;
  ROS_INFO(" Calling Rotor service, Pos = [%6.3f] degree, Speed = [%d]." , srv.request.SetpointPosDegree , srv.request.Speed);
  CurrentPalletDetectionState = PDS_GoingToHomePosition;
  if (client.call(srv))
  {
    ROS_INFO("Rotor Service Completed Successfully.");
    std::cout << "In Home Position." << std::endl;
  }
  else
  {
    ROS_ERROR("Failed to call rotor service.");
  }
  ROS_INFO("Current average resolution is: [%6.3f] degrees.", (*max_element(PosVector.begin(), PosVector.end())- *min_element(PosVector.begin(), PosVector.end()))/PosVector.size());
  CurrentPalletDetectionState = PDS_ScanComplete;

  return Dummy;
}

void DetectionClass::SweepAndScan()
{
  pthread_t RotorServiceThread;
  pthread_create (&RotorServiceThread, NULL, &RotorServiceCaller, (void *)NULL);
  while (ros::ok() && CurrentPalletDetectionState != PDS_ScanComplete)
  {
    usleep (1);
    ros::spinOnce();
  }
  usleep (100); // wait for pthread function to return
}

std::list<PointCloudSegmentStruct> DetectionClass::GetPointCloudSegments(const sensor_msgs::LaserScan& scan, float MinPtDistPoints, unsigned int MinNumberofPoints)
{
  unsigned int StartIndex = 0;
  unsigned int EndIndex = 1;
  unsigned int KeepIndex = 1;
  PointCloudSegmentStruct Segment;
  std::list <PointCloudSegmentStruct> Segments;
  Segment.StartIndex = StartIndex;
  for (size_t Counter = 1; Counter < scan.ranges.size(); Counter++)
  {
    if (scan.ranges[Counter] == 0)
      continue;
    if (fabs(scan.ranges[Counter] - scan.ranges[Counter - 1]) < MinPtDistPoints) //(Tolerance * fmax(scan.ranges[Counter], scan.ranges[Counter - 1])))
      EndIndex = Counter;
    else
    {
      if (KeepIndex - StartIndex > MinNumberofPoints)
      {
        Segment.EndIndex = KeepIndex;
        Segments.push_back(Segment);
      }
      StartIndex = Counter;
      Segment.StartIndex = StartIndex;
    }
    KeepIndex = EndIndex;
  }
  if (KeepIndex - StartIndex > MinNumberofPoints)
  {
    Segment.EndIndex = KeepIndex;
    Segments.push_back(Segment);
  }
  return Segments;
}

geometry_msgs::Point DetectionClass::ConvertToCartesian(int Index, float R, float angle_increment, float angle_min)
{
  geometry_msgs::Point retval;
  double Angle =  Index* angle_increment+angle_min;
  float X =  (R * cos(Angle));
  float Y =  (R * sin(Angle));
  retval.x = X;
  retval.y = Y;
  return retval;
}

geometry_msgs::Point DetectionClass::ProjectPointOnLine(geometry_msgs::Point OutPoint,float m,float d, geometry_msgs::Point APointOnLine)
{
  geometry_msgs::Point retval;
  float X = (OutPoint.x + m * OutPoint.y - m * d) / (m * m + 1);
  float Y = (m * OutPoint.x + m * m * OutPoint.y + d) / (m * m + 1);
  if (isinf(m))
  {
    X = APointOnLine.x;
    Y = OutPoint.y;
  }
  retval.x = X;
  retval.y = Y;
  return retval;
}

std::list <LineSegmentStruct> DetectionClass::GetLineSegments(const sensor_msgs::LaserScan& scan, float MinSqLineLenght, float MaxSqLineLenght,unsigned int MinNumberofPoints)
{
  std::list<PointCloudSegmentStruct> Segments ((std::list<PointCloudSegmentStruct>)GetPointCloudSegments(scan, MinPointDistanceTolerance,MininPointsCount));
  std::list <LineSegmentStruct> Lines;
  std::list<PointCloudSegmentStruct>::iterator SegmentIterator;
  float A = 0, B = 0, C = 0, X = 0, Y = 0;
  float StartX = 0, StartY = 0, EndX = 0, EndY = 0;
  float MaxDistance = 0;
  float Distance = 0;
  int MaxIndex = 0;
  for (size_t SegmentCounter = 0; SegmentCounter < Segments.size(); SegmentCounter++)
  {
    SegmentIterator = Segments.begin();
    for (size_t Counter=0;Counter<SegmentCounter;Counter++)
      SegmentIterator++;
    geometry_msgs::Point StartPoint = ConvertToCartesian(((PointCloudSegmentStruct)(*SegmentIterator)).StartIndex,scan.ranges[((PointCloudSegmentStruct)(*SegmentIterator)).StartIndex],scan.angle_increment, scan.angle_min);
    geometry_msgs::Point EndPoint = ConvertToCartesian(((PointCloudSegmentStruct)(*SegmentIterator)).EndIndex, scan.ranges[((PointCloudSegmentStruct)(*SegmentIterator)).EndIndex],scan.angle_increment, scan.angle_min);
    StartX = StartPoint.x;
    StartY = StartPoint.y;
    EndX = EndPoint.x;
    EndY = EndPoint.y;
    A = EndX - StartX;
    B = StartY - EndY;
    C = (StartX - EndX) * StartY + (EndY - StartY) * StartX;
    MaxDistance = 0;
    MaxIndex = 0;
    for (size_t Counter = ((PointCloudSegmentStruct)(*SegmentIterator)).StartIndex; Counter < ((PointCloudSegmentStruct)(*SegmentIterator)).EndIndex; Counter++)
    {
      if (scan.ranges[Counter] > 0)
      {
        geometry_msgs::Point PointToDet = ConvertToCartesian(Counter, scan.ranges[Counter],scan.angle_increment, scan.angle_min);
        X = PointToDet.x;
        Y = PointToDet.y;
        Distance = fabs(((A * Y + B * X + C) / sqrt(A * A + B * B)));
        if (Distance > MaxDistance)
        {
          MaxDistance = Distance;
          MaxIndex = Counter;
        }
      }
    }

    if (MaxDistance > AllowedPointDistance)
    {
      PointCloudSegmentStruct Segment;
      Segment.StartIndex = ((PointCloudSegmentStruct)(*SegmentIterator)).StartIndex;
      Segment.EndIndex = MaxIndex;
      if (Segment.EndIndex - Segment.StartIndex > MinNumberofPoints)
      {
        SegmentIterator++;
        Segments.insert(SegmentIterator, Segment);
        SegmentIterator--;
        SegmentIterator--;
        Segment.StartIndex = MaxIndex;
        Segment.EndIndex = ((PointCloudSegmentStruct)(*SegmentIterator)).EndIndex;
        if (Segment.EndIndex - Segment.StartIndex > MinNumberofPoints)
        {
          SegmentIterator++;
          SegmentIterator++;
          Segments.insert(SegmentIterator, Segment);
          SegmentIterator--;
          SegmentIterator--;
          SegmentIterator--;
        }
      }
      else
      {
        Segment.StartIndex = MaxIndex;
        Segment.EndIndex = ((PointCloudSegmentStruct)(*SegmentIterator)).EndIndex;
        if (Segment.EndIndex - Segment.StartIndex > MinNumberofPoints)
        {
          SegmentIterator++;
          Segments.insert(SegmentIterator, Segment);
          SegmentIterator--;
          SegmentIterator--;
        }
      }
      Segments.erase(SegmentIterator);
      SegmentCounter--;
    }
    else
    {
      if ((StartX - EndX) * (StartX - EndX) + (StartY - EndY) * (StartY - EndY) > MinSqLineLenght
          && (StartX - EndX) * (StartX - EndX) + (StartY - EndY) * (StartY - EndY) < MaxSqLineLenght)
      {
        LineSegmentStruct line;
        PointCloudSegmentStruct Segment =(PointCloudSegmentStruct)(*SegmentIterator);
        double Xm = 0;
        double Ym = 0;
        int PointsCount = 0;
        for (size_t Counter = Segment.StartIndex; Counter < Segment.EndIndex; Counter++)
        {
          if (scan.ranges[Counter] > 0)
          {
            geometry_msgs::Point PointToDet = ConvertToCartesian(Counter, scan.ranges[Counter],scan.angle_increment, scan.angle_min);
            Xm += PointToDet.x;
            Ym += PointToDet.y;
            PointsCount++;
          }
        }
        Xm /= PointsCount;
        Ym /= PointsCount;
        double Num = 0;
        double Denum = 0;
        for (size_t Counter = Segment.StartIndex; Counter < Segment.EndIndex; Counter++)
        {
          if (scan.ranges[Counter] > 0)
          {
            geometry_msgs::Point PointToDet = ConvertToCartesian(Counter, scan.ranges[Counter],scan.angle_increment, scan.angle_min);
            Num += ((Xm - PointToDet.x) * (Ym - PointToDet.y));
            Denum += ((Ym - PointToDet.y) * (Ym - PointToDet.y) - (Xm - PointToDet.x) * (Xm - PointToDet.x));
          }
        }
        double LineLenght1 = 0;
        LineSegmentStruct SuspLine1;
        double Phi = M_PI / 4;
        if (Denum != 0)
          Phi = atan(-2 * Num / Denum) / 2;
        double Slope = std::numeric_limits<double>::quiet_NaN();
        if (Phi != 0 && !((Phi <M_PI/2+0.001 && Phi >M_PI/2-0.001) || (Phi >-M_PI/2-0.001 && Phi <-M_PI/2+0.001)))
          Slope = -1 / tan(Phi);
        double R = Xm * cos(Phi) + Ym * sin(Phi);
        double XL = R * cos(Phi);
        double YL = R * sin(Phi);
        if ((Phi <M_PI/2+0.001 && Phi >M_PI/2-0.001) || (Phi >-M_PI/2-0.001 && Phi <-M_PI/2+0.001))
        {
          line.StartX = (float)XL;
          line.StartY = (float)StartY;
          line.EndX = (float)XL;
          line.EndY = (float)EndY;
        }
        else
        {
          geometry_msgs::Point XYPoint;
          XYPoint.x = XL;
          XYPoint.y = YL;
          geometry_msgs::Point StartOutPoint = ConvertToCartesian(Segment.StartIndex, scan.ranges[Segment.StartIndex],scan.angle_increment, scan.angle_min);
          geometry_msgs::Point EndOutPoint = ConvertToCartesian(Segment.EndIndex, scan.ranges[Segment.EndIndex],scan.angle_increment, scan.angle_min);
          geometry_msgs::Point ProjectedStartPoint = ProjectPointOnLine(StartOutPoint, (float)Slope, (float)(YL - Slope * XL),XYPoint);
          geometry_msgs::Point ProjectedEndPoint = ProjectPointOnLine(EndOutPoint, (float)Slope, (float)(YL - Slope * XL), XYPoint);
          line.StartX = ProjectedStartPoint.x;
          line.StartY = ProjectedStartPoint.y;
          line.EndX = ProjectedEndPoint.x;
          line.EndY = ProjectedEndPoint.y;
        }
        LineLenght1 = sqrt((line.StartX - line.EndX) * (line.StartX - line.EndX) + (line.StartY - line.EndY) * (line.StartY - line.EndY));
        SuspLine1 = line;
        double LineLenght2 = 0;
        //                        if (Math.Sqrt((line.StartX - line.EndX) * (line.StartX - line.EndX) + (line.StartY - line.EndY) * (line.StartY - line.EndY)) < 100)
        {
          Phi -= M_PI / 2;
          Slope = std::numeric_limits<double>::quiet_NaN();
          if (Phi != 0 && !((Phi <M_PI/2+0.001 && Phi >M_PI/2-0.001) || (Phi >-M_PI/2-0.001 && Phi <-M_PI/2+0.001)))
            Slope = -1 / tan(Phi);
          R = Xm * cos(Phi) + Ym * sin(Phi);
          XL = R * cos(Phi);
          YL = R * sin(Phi);
          if ((Phi <M_PI/2+0.001 && Phi >M_PI/2-0.001) || (Phi >-M_PI/2-0.001 && Phi <-M_PI/2+0.001))
          {
            line.StartX = (float)XL;
            line.StartY = (float)StartY;
            line.EndX = (float)XL;
            line.EndY = (float)EndY;
          }
          else
          {
            geometry_msgs::Point XYPoint;
            XYPoint.x = XL;
            XYPoint.y = YL;
            geometry_msgs::Point StartOutPoint = ConvertToCartesian(Segment.StartIndex, scan.ranges[Segment.StartIndex],scan.angle_increment, scan.angle_min);
            geometry_msgs::Point EndOutPoint = ConvertToCartesian(Segment.EndIndex, scan.ranges[Segment.EndIndex],scan.angle_increment, scan.angle_min);
            geometry_msgs::Point ProjectedStartPoint = ProjectPointOnLine(StartOutPoint, (float)Slope, (float)(YL - Slope * XL), XYPoint);
            geometry_msgs::Point ProjectedEndPoint = ProjectPointOnLine(EndOutPoint, (float)Slope, (float)(YL - Slope * XL), XYPoint);
            line.StartX = ProjectedStartPoint.x;
            line.StartY = ProjectedStartPoint.y;
            line.EndX = ProjectedEndPoint.x;
            line.EndY = ProjectedEndPoint.y;
          }
          LineLenght2 = sqrt((line.StartX - line.EndX) * (line.StartX - line.EndX) + (line.StartY - line.EndY) * (line.StartY - line.EndY));
        }
        if (LineLenght1 > LineLenght2)
          line = SuspLine1;
        if (line.StartX == line.StartX && line.StartY == line.StartY && line.EndX == line.EndX && line.EndY == line.EndY) // checking nan and inf values
        {
          line.StartZ=0;
          line.EndZ = 0;
          Lines.push_back(line);
        }
      }
    }
  }
  return Lines;
}

float DetectionClass::LineLenght(LineSegmentStruct inputline)
{
  return sqrt((inputline.StartX-inputline.EndX)*(inputline.StartX-inputline.EndX) + (inputline.StartY-inputline.EndY)*(inputline.StartY-inputline.EndY));
}

float DetectionClass::DistanceFromPointToLine (geometry_msgs::Point ThePoint, geometry_msgs::Point LineP1, geometry_msgs::Point LineP2)
{
  float x0=ThePoint.x, y0=ThePoint.y, x1=LineP1.x, y1=LineP1.y, x2=LineP2.x, y2 = LineP2.y;
  return fabs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)/sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1));
}

float DetectionClass::PointToPointDistance(geometry_msgs::Point Point1, geometry_msgs::Point Point2)
{
  return sqrt((Point1.x - Point2.x) * (Point1.x - Point2.x) + (Point1.y - Point2.y) * (Point1.y - Point2.y));
}

std::vector <LineSegmentStruct> DetectionClass::DetectPalletFromThreePillarFrontFaceLines(std::list<LineSegmentStruct> Lines)
{
  LineSegmentStruct Pallet;
  std::vector <LineSegmentStruct> Pallets;

  //std::vector<LineSegmentStruct> v{ std::list::begin(Lines), std::list::end(Lines) };

  std::vector<LineSegmentStruct> v;
  v.reserve(Lines.size());
  v.insert(v.end(),Lines.begin(), Lines.end());

  std::vector<LineSegmentStruct> TempLines;


  Pallet.StartX = 0;
  Pallet.StartY = 0;
  Pallet.EndX = 0;
  Pallet.EndY = 0;

  if (Lines.size()<3)
  {
    return Pallets;
  }

  if (IsDebugging)
  {
    DrawLines(&visArray_Pub,v,0.02);
    ros::spinOnce();
  }

  LineSegmentStruct Line1;
  LineSegmentStruct Line2;
  LineSegmentStruct Line3;
  LineSegmentStruct TempLine;

  geometry_msgs::Point Line1MidPoint;
  geometry_msgs::Point Line2MidPoint;
  geometry_msgs::Point Line3MidPoint;
  geometry_msgs::Point TempLineP1;
  geometry_msgs::Point TempLineP2;

  std::list<LineSegmentStruct> Batch1;
  std::list<LineSegmentStruct> Batch2;
  std::list<LineSegmentStruct> AllIntermediateLines;

  bool DefetiveLinesegmentFound = false;

  int counter =1;
  if (IsDebugging)
  {

    for ( std::list <LineSegmentStruct>::iterator it1 = Lines.begin(); it1!=Lines.end(); ++it1)
    {
      std::cout << counter++ << "- ["<<((LineSegmentStruct)(*it1)).StartX << " ," <<((LineSegmentStruct)(*it1)).StartY << "] to [" <<((LineSegmentStruct)(*it1)).EndX << " ,"<<((LineSegmentStruct)(*it1)).EndY << "]." << " Len = [" << LineLenght((LineSegmentStruct)(*it1)) << "]."<< std::endl;
    }
  }
  for ( std::list <LineSegmentStruct>::iterator it1 = Lines.begin(); it1!=Lines.end(); ++it1)
  {
    bool PointSetFound = false;
    Line1 = ((LineSegmentStruct)(*it1));
    TempLines.clear();
    TempLines.push_back(Line1);
    if (IsDebugging)
    {
      DrawLines(&visArray_Pub,TempLines,0.02);
      ros::spinOnce();
    }
    if (LineLenght(Line1) > PalletpillarThicknessMax)
      continue;
    Line1MidPoint.x = (Line1.StartX + Line1.EndX)/2.0;
    Line1MidPoint.y = (Line1.StartY + Line1.EndY)/2.0;
    AllIntermediateLines.clear();
    Batch1.clear();
    for ( std::list <LineSegmentStruct>::iterator it2 = Next(it1,1); it2!=Lines.end(); ++it2)
    {
      Line2 = ((LineSegmentStruct)(*it2));
      TempLines.clear();
      TempLines.push_back(Line1);
      TempLines.push_back(Line2);
      if (IsDebugging)
      {
        DrawLines(&visArray_Pub,TempLines,0.02);
        ros::spinOnce();
      }
      if (LineLenght(Line2) > PalletpillarThicknessMax)
      {
        if (IsDebugging)
        {
          ROS_INFO("Rejeted Since [%3.3f] is larger than PalletpillarThicknessMax [%3.3f]", LineLenght(Line2), PalletpillarThicknessMax);
        }
        Batch1.push_back((LineSegmentStruct)(*it2));
        continue;
      }
      Line2MidPoint.x = (Line2.StartX + Line2.EndX)/2.0;
      Line2MidPoint.y = (Line2.StartY + Line2.EndY)/2.0;
      if (fabs(PointToPointDistance(Line2MidPoint, Line1MidPoint) - ForkWindowWidth) > ForkWindowWidthTolerance)
      {
        if (IsDebugging)
        {
          ROS_INFO("Distance = [%3.3f], Rejeted Since [%3.3f] is larger than ForkWindowWidthTolerance, Allowed Window: [%3.3f], Allowed Tol: [%3.3f]", PointToPointDistance(Line2MidPoint, Line1MidPoint), PointToPointDistance(Line2MidPoint, Line1MidPoint) - ForkWindowWidth, ForkWindowWidth, ForkWindowWidthTolerance);
        }
        Batch1.push_back((LineSegmentStruct)(*it2));
        continue;
      }
      Batch2.clear();
      for ( std::list <LineSegmentStruct>::iterator it3 = Next(it2,1); it3!=Lines.end(); ++it3)
      {
        Line3 = ((LineSegmentStruct)(*it3));
        TempLines.clear();
        TempLines.push_back(Line1);
        TempLines.push_back(Line2);
        TempLines.push_back(Line3);
        if (IsDebugging)
        {
          DrawLines(&visArray_Pub,TempLines,0.02);
          ros::spinOnce();
        }
        if (LineLenght(Line3) > PalletpillarThicknessMax)
        {
          if (IsDebugging)
          {
            ROS_INFO("Rejeted Since [%3.3f] is larger than PalletpillarThicknessMax [%3.3f]", LineLenght(Line3), PalletpillarThicknessMax);
          }
          Batch2.push_back((LineSegmentStruct)(*it3));
          continue;
        }
        Line3MidPoint.x = (Line3.StartX + Line3.EndX)/2.0;
        Line3MidPoint.y = (Line3.StartY + Line3.EndY)/2.0;
        if (fabs(PointToPointDistance(Line1MidPoint, Line3MidPoint) - 2*ForkWindowWidth) > ForkWindowWidthTolerance)
        {
          if (IsDebugging)
          {
            ROS_INFO("Rejeted Since [%3.3f] is larger than ForkWindowWidthTolerance [%3.3f]",
                     fabs(PointToPointDistance(Line1MidPoint, Line3MidPoint) - (2*ForkWindowWidth))
                     , ForkWindowWidthTolerance);
          }
          Batch2.push_back((LineSegmentStruct)(*it3));
          continue;
        }
        if (fabs(PointToPointDistance(Line2MidPoint, Line3MidPoint) - ForkWindowWidth) > ForkWindowWidthTolerance)
        {
          if (IsDebugging)
          {
            ROS_INFO("Rejeted Since [%3.3f] is larger than ForkWindowWidthTolerance [%3.3f]",
                     fabs(PointToPointDistance(Line2MidPoint, Line3MidPoint) - ForkWindowWidth)
                     , ForkWindowWidthTolerance);
          }
          Batch2.push_back((LineSegmentStruct)(*it3));
          continue;
        }
        if (DistanceFromPointToLine(Line2MidPoint, Line1MidPoint, Line3MidPoint) > MidPilarPointFromSidePilarLineDistanceMax)
        {
          if (IsDebugging)
          {
            ROS_INFO("Rejeted Since [%3.3f] is larger than MidPilarPointFromSidePilarLineDistanceMax [%3.3f]",
                     DistanceFromPointToLine(Line2MidPoint, Line1MidPoint, Line3MidPoint)
                     , MidPilarPointFromSidePilarLineDistanceMax);
          }
          Batch2.push_back((LineSegmentStruct)(*it3));
          continue;
        }

        AllIntermediateLines.insert( AllIntermediateLines.end(), Batch1.begin(), Batch1.end() );
        AllIntermediateLines.insert( AllIntermediateLines.end(), Batch2.begin(), Batch2.end() );

        DefetiveLinesegmentFound = false;
        for ( std::list <LineSegmentStruct>::iterator it4 = AllIntermediateLines.begin(); it4!=AllIntermediateLines.end(); ++it4)
        {
          TempLine = ((LineSegmentStruct)(*it4));
          TempLines.clear();
          TempLines.push_back(Line1);
          TempLines.push_back(Line2);
          TempLines.push_back(Line3);
          TempLines.push_back(TempLine);
          if (IsDebugging)
          {
            DrawLines(&visArray_Pub,TempLines,0.02);
            ros::spinOnce();
          }
          TempLineP1.x = TempLine.StartX;
          TempLineP1.y = TempLine.StartY;
          TempLineP2.x = TempLine.EndX;
          TempLineP2.y = TempLine.EndY;
 /*         if (DistanceFromPointToLine(TempLineP1,Line1MidPoint, Line3MidPoint) < ToleranceFromLongLine && DistanceFromPointToLine(TempLineP2,Line1MidPoint, Line3MidPoint) < ToleranceFromLongLine)
          {
            DefetiveLinesegmentFound = true;
            break;
          }*/
          if (TempLineP1.x < std::min(std::min(Line1MidPoint.x, Line2MidPoint.x), Line3MidPoint.x) && TempLineP2.x < std::min(std::min(Line1MidPoint.x, Line2MidPoint.x), Line3MidPoint.x))
          {
            DefetiveLinesegmentFound = true;
            break;
          }
        }
        if (DefetiveLinesegmentFound)
        {
          AllIntermediateLines.clear();
          Batch2.push_back((LineSegmentStruct)(*it3));
          Pallet.StartX = 0;
          Pallet.StartY = 0;
          Pallet.EndX = 0;
          Pallet.EndY = 0;
        }
        else
        {
          Pallet.StartX = Line1MidPoint.x;
          Pallet.StartY = Line1MidPoint.y;
          Pallet.EndX = Line3MidPoint.x;
          Pallet.EndY = Line3MidPoint.y;
          Pallets.push_back(Pallet);
          Pallet.StartX = 0;
          Pallet.StartY = 0;
          Pallet.EndX = 0;
          Pallet.EndY = 0;
          AllIntermediateLines.clear();
          Batch1.clear();
          Batch2.clear();
          PointSetFound = true;
//          it1 = it3;
          break;
        }
      }
      if(PointSetFound)
        break;
      Batch1.push_back((LineSegmentStruct)(*it2));
    }
  }
  return Pallets;
}

sensor_msgs::PointCloud2 DetectionClass::ShowOutputCloud (vector<sensor_msgs::LaserScan> inputscans)
{
  pcl::PointCloud<pcl::PointXYZ> BigCloud;
  laser_geometry::LaserProjection projector_;
  sensor_msgs::PointCloud2 BigCloudMsg;
  tf::TransformListener listener_;

  for (uint Counter = 0; Counter < PosVector.size(); Counter++)
  {
        sensor_msgs::LaserScan scan_in = inputscans[Counter];
    // Convert laserscan message to RosPointCloud2 Message
//		listener_.waitForTransform(scan_in.header.frame_id, "/vertical_laser_link", scan_in.header.stamp + ros::Duration().fromSec(scan_in.ranges.size()*scan_in.time_increment),ros::Duration(1));
    sensor_msgs::PointCloud2 roscloud;
    projector_.transformLaserScanToPointCloud("/vertical_laser_link", scan_in, roscloud,listener_);
    //Convert RosPointCloud2 to PCLPointCloud2
    pcl::PCLPointCloud2 pcl_pointcloud2;
    pcl_conversions::toPCL(roscloud,pcl_pointcloud2);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::fromPCLPointCloud2(pcl_pointcloud2, *cloudXYZ);

    //Transform PCLPointCloud<T>
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate (Eigen::AngleAxisf (PosVector[Counter] * M_PI / 180.0 - M_PI/2.0, Eigen::Vector3f::UnitY()));
    pcl::transformPointCloud (*cloudXYZ, *transformed_cloud, transform);
//    transform = Eigen::Affine3f::Identity();
//    transform.translation() << 0.0, 0.0, SensorElevation;
//    pcl::transformPointCloud (*transformed_cloud, *transformed_cloud, transform);
//		ROS_INFO ("Appending New Scan ... [%d] @ [%6.3f] Degrees. ", Counter+1, PosVector[Counter]);

    //Append to Big PointCloud
    BigCloud+= (*transformed_cloud);
  }
  BigCloud.header.frame_id = "/vertical_laser_link";
  pcl::toROSMsg(BigCloud, BigCloudMsg);
  pub3D.publish(BigCloudMsg);
  ros::spinOnce();
  return BigCloudMsg;
}

void DetectionClass::DrawLine(ros::Publisher * vispub, LineSegmentStruct TheLine)
{
  geometry_msgs::Point pt;

  visualization_msgs::Marker marker;
  marker.header.frame_id = "/vertical_laser_link";
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.id =0;
  marker.pose.position.x = 0;
  marker.pose.position.y = 0;
  marker.pose.position.z = 0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.05;
  marker.scale.y = 0.0;
  marker.scale.z = 0.05;
  marker.color.a = 1.0;
  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
//	marker.lifetime = ros::Duration(4);

  visualization_msgs::Marker marker2;
  marker2.header.frame_id = "/vertical_laser_link";
  marker2.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker2.action = visualization_msgs::Marker::ADD;
  marker2.id =1;
  marker2.pose.orientation.x = 0.0;
  marker2.pose.orientation.y = 0.0;
  marker2.pose.orientation.z = 0.0;
  marker2.pose.orientation.w = 1.0;
  marker2.scale.x = 0.1;
  marker2.scale.y = 0.0;
  marker2.scale.z = 0.1;
  marker2.color.a = 1.0;
  marker2.color.r = 1.0;
  marker2.color.g = 0.0;
  marker2.color.b = 0.0;
//	marker2.lifetime = ros::Duration(4);

  float angle = (atan2(TheLine.EndY-TheLine.StartY,TheLine.EndX-TheLine.StartX)- M_PI/2.0) *180.0/M_PI;
  std::string s="";
  pt.x=TheLine.StartX ;
  pt.y=TheLine.StartY ;
  pt.z= TheLine.StartZ ;
  marker.points.push_back(pt);
  pt.x=TheLine.EndX ;
  pt.y=TheLine.EndY ;
  pt.z= TheLine.EndZ ;
  marker.points.push_back(pt);
  marker2.pose.position.x = (TheLine.StartX + TheLine.EndX)/2.0;
  marker2.pose.position.y = (TheLine.StartY + TheLine.EndY)/2.0;
  marker2.pose.position.z = (TheLine.StartZ + TheLine.EndZ)/2.0;
  s.append("[");
  s.append(ToString((TheLine.StartY + TheLine.EndY)/-2.0));
  s.append(" , ");
  s.append(ToString((TheLine.StartX + TheLine.EndX)/2.0));
  s.append("] , Angle = [");
  s.append(ToString(angle));
  s.append("] degrees.");
  s.append("Pallet width = [");
  s.append(ToString(LineLenght(TheLine)));
  s.append("] meters.");
  marker2.text = s;

  if (!marker.points.empty())
  {
    vispub->publish( marker );
    ros::spinOnce();
    vispub->publish( marker2 );
    ros::spinOnce();
  }
}

void DetectionClass::DrawLines(ros::Publisher * visArrayPub, std::vector <LineSegmentStruct> Lines, float Thickness)
{
  if (Lines.size()  == 0)
    return;
  geometry_msgs::Point pt;

  visualization_msgs::Marker marker;
  visualization_msgs::Marker marker2;
  visualization_msgs::MarkerArray marArray;

//*************This portion clears the marker array history *** vvvvvvvv
/*  marker.header.frame_id = "/vertical_laser_link";
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::DELETEALL;
  marker2.header.frame_id = "/vertical_laser_link";
  marker2.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker2.action = visualization_msgs::Marker::DELETEALL;

  marArray.markers.push_back(marker);
  marArray.markers.push_back(marker2);
  visArrayPub->publish( marArray );
  ros::spinOnce();*/
  marArray.markers.clear();
//***************************** ^^^^^^^  **************

  marker.header.frame_id = "/vertical_laser_link";
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.id =0;
  marker.pose.position.x = 0;
  marker.pose.position.y = 0;
  marker.pose.position.z = 0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = Thickness;
  marker.scale.y = 0.0;
  marker.scale.z = Thickness;
  marker.color.a = 1.0;
  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
//  marker.text = "***********************";
//	marker.lifetime = ros::Duration(4);

  marker2.header.frame_id = "/vertical_laser_link";
  marker2.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker2.action = visualization_msgs::Marker::ADD;
  marker2.pose.orientation.x = 0.0;
  marker2.pose.orientation.y = 0.0;
  marker2.pose.orientation.z = 0.0;
  marker2.pose.orientation.w = 1.0;
  marker2.scale.x = 0.3;
  marker2.scale.y = 0.0;
  marker2.scale.z = 0.3;
  marker2.color.a = 1.0;
  marker2.color.r = 1.0;
  marker2.color.g = 1.0;
  marker2.color.b = 0.0;
//	marker2.lifetime = ros::Duration(4);

  int PalletCounter=0;
  for ( std::vector <LineSegmentStruct>::iterator it = Lines.begin(); it!=Lines.end(); ++it)
  {
    if (((LineSegmentStruct)(*it)).StartX != 0 || ((LineSegmentStruct)(*it)).StartY != 0)
    {
      float angle = (atan2(((LineSegmentStruct)(*it)).EndY-((LineSegmentStruct)(*it)).StartY,((LineSegmentStruct)(*it)).EndX-((LineSegmentStruct)(*it)).StartX)- M_PI/2.0) *180.0/M_PI;
//      if (fabs(angle) <45)
      {
        std::string s="";
        pt.x=((LineSegmentStruct)(*it)).StartX ;
        pt.y=((LineSegmentStruct)(*it)).StartY ;
        pt.z=((LineSegmentStruct)(*it)).StartZ ;
        marker.points.push_back(pt);
        pt.x=((LineSegmentStruct)(*it)).EndX ;
        pt.y=((LineSegmentStruct)(*it)).EndY ;
        pt.z=((LineSegmentStruct)(*it)).EndZ ;
        marker.points.push_back(pt);
        marker2.id =PalletCounter+1;
        marker2.pose.position.x = (((LineSegmentStruct)(*it)).StartX + ((LineSegmentStruct)(*it)).EndX)/2.0 - 1;
        marker2.pose.position.y = (((LineSegmentStruct)(*it)).StartY + ((LineSegmentStruct)(*it)).EndY)/2.0;
        marker2.pose.position.z = (((LineSegmentStruct)(*it)).StartZ + ((LineSegmentStruct)(*it)).EndZ)/2.0 + 0.1;
        s.append("[");
        s. append(ToString(PalletCounter+1));
//        s. append(ToString((((LineSegmentStruct)(*it)).StartY + ((LineSegmentStruct)(*it)).EndY)/-2.0));
//        s.append(" , ");
//        s. append(ToString((((LineSegmentStruct)(*it)).StartX + ((LineSegmentStruct)(*it)).EndX)/2.0));
//        s.append("] , Angle = [");
//        s.append(ToString(angle));
        s.append("]");
        marker2.text = s;
        marArray.markers.push_back(marker2);
        PalletCounter++;
      }
    }
  }
  marArray.markers.push_back(marker);

  if (!marArray.markers.empty())
  {
    visArrayPub->publish( marArray );
    ros::spinOnce();
  }
/*
  if (!marker.points.empty())
  {
    vispub->publish( marker );
    ros::spinOnce();
//    vispub->publish( marker2 );
//    ros::spinOnce();
  }*/
}

LineSegmentStruct TransformLine (LineSegmentStruct BetterLine,LineSegmentStruct Origline)
{
  LineSegmentStruct TransformedLine;
  float Tx = (Origline.StartX + Origline.EndX)/2.0;
  float Ty = (Origline.StartY + Origline.EndY)/2.0;
  float Theta = atan2(Origline.EndY-Origline.StartY,Origline.EndX-Origline.StartX)- M_PI/2.0;

  std::cout << "Origin StartX:" << Origline.StartX << std::endl;
  std::cout << "Origin StartY:" << Origline.StartY << std::endl;
  std::cout << "Origin EndX:" << Origline.EndX << std::endl;
  std::cout << "Origin EndY:" << Origline.EndY << std::endl;
  //std::cout << "cos(TAng):" << cos(TAng) << std::endl;
//  std::cout << "sin(TAng):" << sin(TAng) << std::endl;
/*  std::cout << "Origline.StartX * cos(TAng):" << Origline.StartX * cos(TAng) << std::endl;
  std::cout << "Origline.StartY * sin(TAng):" << Origline.StartY * sin(TAng) << std::endl;
  std::cout << "Origline.EndX * cos(TAng):" << Origline.EndX * cos(TAng) << std::endl;
  std::cout << "Origline.EndY * cos(TAng):" << Origline.EndY * cos(TAng) << std::endl;*/
/*  TransformedLine.StartX = Origline.StartX * cos(TAng) - Origline.StartY * sin(TAng) + Tx;
  TransformedLine.StartY = Origline.StartX * sin(TAng) + Origline.StartY * cos(TAng) + Ty;
  TransformedLine.EndX = Origline.EndX * cos(TAng) - Origline.EndY * sin(TAng) + Tx;
  TransformedLine.EndY = Origline.EndX * sin(TAng) + Origline.EndY * cos(TAng) + Ty;*/
    TransformedLine.StartX = BetterLine.StartX * cos(Theta) - BetterLine.StartY * sin(Theta) + Tx;
    TransformedLine.StartY = BetterLine.StartX * sin(Theta) + BetterLine.StartY * cos(Theta) + Ty;
    TransformedLine.EndX = BetterLine.EndX * cos(Theta) - BetterLine.EndY * sin(Theta) + Tx;
    TransformedLine.EndY = BetterLine.EndX * sin(Theta) + BetterLine.EndY * cos(Theta) + Ty;
  return TransformedLine;
}

std::vector <LineSegmentStruct> CalculateZ(std::vector <LineSegmentStruct> InputLines,float Angle_in_Degrees)
{
  std::vector <LineSegmentStruct> LinesToReturn;
  if (!InputLines.empty())
  {
    for ( std::vector <LineSegmentStruct>::iterator it = InputLines.begin(); it!=InputLines.end(); ++it)
    {
      LineSegmentStruct LinetoUpdate = (LineSegmentStruct)(*it);
      float StartXYDistance = sqrt(LinetoUpdate.StartX*LinetoUpdate.StartX+LinetoUpdate.StartY*LinetoUpdate.StartY);
      LinetoUpdate.StartZ = StartXYDistance * sin((90-Angle_in_Degrees)*M_PI/180.0);
      float EndXYDistance = sqrt(LinetoUpdate.EndX*LinetoUpdate.EndX+LinetoUpdate.EndY*LinetoUpdate.EndY);
      LinetoUpdate.EndZ = EndXYDistance * sin((90-Angle_in_Degrees)*M_PI/180.0);
      LinesToReturn.push_back(LinetoUpdate);
    }
  }
  return LinesToReturn;
}

std::vector <LineSegmentStruct> RemoveObviousFalseCase(std::vector <LineSegmentStruct> InputLines, float MaxAngle,float MinDistFromCenter,float MinDepth,float MaxHeight)
{
  std::vector <LineSegmentStruct> LinesToReturn;
  if (!InputLines.empty())
  {
    for ( std::vector <LineSegmentStruct>::iterator it = InputLines.begin(); it!=InputLines.end(); ++it)
    {
      LineSegmentStruct LinetoUpdate = (LineSegmentStruct)(*it);

      float angle = (atan2(LinetoUpdate.EndY-LinetoUpdate.StartY,LinetoUpdate.EndX-LinetoUpdate.StartX)- M_PI/2.0) *180.0/M_PI;
      if (fabs(angle) > MaxAngle)
      {
        ROS_INFO("MaxAngle Filter");
        continue;
      }

      float CenterX  = (LinetoUpdate.StartX+LinetoUpdate.EndX)/2.0;
      if (fabs(CenterX) < MinDepth)
      {
        ROS_INFO("Depth Filter");
        continue;
      }

      float CenterY  = (LinetoUpdate.StartY+LinetoUpdate.EndY)/2.0;
      if (fabs(CenterY) < MinDistFromCenter)
      {
        ROS_INFO("Center Filter");
        continue;
      }

      float CenterZ  = (LinetoUpdate.StartZ+LinetoUpdate.EndZ)/2.0;
      if (CenterZ < -1 || CenterZ > MaxHeight)
      {
        ROS_INFO("Height Filter");
        continue;
      }

      LinesToReturn.push_back(LinetoUpdate);
    }
  }
  return LinesToReturn;
}

std::vector <LineSegmentStruct> DetectionClass::OutputResults (vector <LineSegmentStruct> DetectedPalletLines)
{
  std::vector <LineSegmentStruct> RepGroup;

  std::cout << "-------------------------------------------"<<std::endl;
  ROS_INFO ("Here is the sorted list of possible pallet lines:");
  int counter=1;
  for ( std::vector <LineSegmentStruct>::iterator it = DetectedPalletLines.begin(); it!=DetectedPalletLines.end(); ++it)
  {
    LineSegmentStruct candidateline = (LineSegmentStruct)(*it);
    float cen1 = (candidateline.StartY+candidateline.EndY)/2.0;
    std::cout << counter++ << "- ["<<((LineSegmentStruct)(*it)).StartX << " ," <<((LineSegmentStruct)(*it)).StartY <<
                 "] to [" <<((LineSegmentStruct)(*it)).EndX << " ,"<<((LineSegmentStruct)(*it)).EndY << "]." <<
                 " Len = [" << LineLenght((LineSegmentStruct)(*it)) << "];"<< "Centered at: [" << cen1 << "]." <<std::endl;

  }
  std::vector <std::vector <LineSegmentStruct> > PalletGroups;
  if (DetectedPalletLines.size()>0)
  {
    for ( std::vector <LineSegmentStruct>::iterator it = DetectedPalletLines.begin(); it!=DetectedPalletLines.end(); ++it)
    {
      LineSegmentStruct candidateline = (LineSegmentStruct)(*it);
      if (PalletGroups.empty())
      {
        std::vector <LineSegmentStruct> NewGroup;
        NewGroup.push_back(candidateline);
        PalletGroups.push_back(NewGroup);
        continue;
      }
      float CenterofCandidateLineX = (candidateline.StartX+candidateline.EndX)/2.0;
      float CenterofCandidateLineY = (candidateline.StartY+candidateline.EndY)/2.0;
      float CenterofCandidateLineZ = (candidateline.StartZ+candidateline.EndZ)/2.0;
      bool FoundTheMatchingGroup = false;
      for ( std::vector <std::vector <LineSegmentStruct> >::iterator pgit = PalletGroups.begin(); pgit!=PalletGroups.end(); ++pgit)
      {
        LineSegmentStruct TheGroupCenterLine;
        TheGroupCenterLine.StartX = 0;
        TheGroupCenterLine.StartY = 0;
        TheGroupCenterLine.StartZ = 0;
        TheGroupCenterLine.EndX = 0;
        TheGroupCenterLine.EndY = 0;
        TheGroupCenterLine.EndZ = 0;
        //Calculate the center line the group
        for ( std::vector <LineSegmentStruct>::iterator tgit = pgit->begin(); tgit!=pgit->end(); ++tgit)
        {
          LineSegmentStruct TheLine = (LineSegmentStruct)(*tgit);
          TheGroupCenterLine.StartX += TheLine.StartX;
          TheGroupCenterLine.StartY += TheLine.StartY;
          TheGroupCenterLine.StartZ += TheLine.StartZ;
          TheGroupCenterLine.EndX += TheLine.EndX;
          TheGroupCenterLine.EndY += TheLine.EndY;
          TheGroupCenterLine.EndZ += TheLine.EndZ;
        }
        TheGroupCenterLine.StartX /= ((std::vector <LineSegmentStruct>)(*pgit)).size();
        TheGroupCenterLine.StartY /= ((std::vector <LineSegmentStruct>)(*pgit)).size();
        TheGroupCenterLine.StartZ /= ((std::vector <LineSegmentStruct>)(*pgit)).size();
        TheGroupCenterLine.EndX /= ((std::vector <LineSegmentStruct>)(*pgit)).size();
        TheGroupCenterLine.EndY /= ((std::vector <LineSegmentStruct>)(*pgit)).size();
        TheGroupCenterLine.EndZ /= ((std::vector <LineSegmentStruct>)(*pgit)).size();
        float CenterofGroupX = (TheGroupCenterLine.StartX+TheGroupCenterLine.EndX)/2.0;
        float CenterofGroupY = (TheGroupCenterLine.StartY+TheGroupCenterLine.EndY)/2.0;
        float CenterofGroupZ = (TheGroupCenterLine.StartZ+TheGroupCenterLine.EndZ)/2.0;
        float DistanceFromCandidateLineToGroupLine = sqrt((CenterofGroupX-CenterofCandidateLineX)*(CenterofGroupX-CenterofCandidateLineX)
                                                          + (CenterofGroupY-CenterofCandidateLineY)*(CenterofGroupY-CenterofCandidateLineY)
                                                          + (CenterofGroupZ-CenterofCandidateLineZ)*(CenterofGroupZ-CenterofCandidateLineZ)
                                                          );
        if (DistanceFromCandidateLineToGroupLine < GroupingMaxDistance)
        {
          std::vector <LineSegmentStruct> TheGroup = std::vector <LineSegmentStruct> (*pgit);
          TheGroup.push_back(candidateline);
          pgit = PalletGroups.erase(pgit);
          PalletGroups.push_back(TheGroup);
          FoundTheMatchingGroup = true;
          break;
        }
      }
      if(!FoundTheMatchingGroup)
      {
        std::vector <LineSegmentStruct> NewGroup;
        NewGroup.push_back(candidateline);
        PalletGroups.push_back(NewGroup);
      }
    }
    int DummyCounter = 0;

    std::cout << "\033[21;30;47m\n ****  Pallets are classified into the below groups: ****"<< std::endl;
    for ( std::vector <std::vector <LineSegmentStruct> >::iterator pgit = PalletGroups.begin(); pgit!=PalletGroups.end(); ++pgit)
    {
      DummyCounter++;
      std::cout << "\n\033[21;30;43m----  Group " << DummyCounter << ": ----\033[21;30;47m" << std::endl;

      int DummyCounter2 = 0;
      LineSegmentStruct TheGroupCenterLine;
      TheGroupCenterLine.StartX = 0;
      TheGroupCenterLine.StartY = 0;
      TheGroupCenterLine.StartZ = 0;
      TheGroupCenterLine.EndX = 0;
      TheGroupCenterLine.EndY = 0;
      TheGroupCenterLine.EndZ = 0;
      for ( std::vector <LineSegmentStruct>::iterator tgit = pgit->begin(); tgit!=pgit->end(); ++tgit)
      {
        DummyCounter2++;
        LineSegmentStruct TheLine = (LineSegmentStruct)(*tgit);
        std::cout << DummyCounter2 << "- The Line From ["
                  << TheLine.StartX <<", " << TheLine.StartY << ", " << TheLine.StartZ
                  << "] To ["
                  << TheLine.EndX  <<", "<< TheLine.EndY <<", " << TheLine.EndZ
                  << "]" << std::endl;
        TheGroupCenterLine.StartX += TheLine.StartX;
        TheGroupCenterLine.StartY += TheLine.StartY;
        TheGroupCenterLine.StartZ += TheLine.StartZ;
        TheGroupCenterLine.EndX += TheLine.EndX;
        TheGroupCenterLine.EndY += TheLine.EndY;
        TheGroupCenterLine.EndZ += TheLine.EndZ;
      }
      TheGroupCenterLine.StartX /= ((std::vector <LineSegmentStruct>)(*pgit)).size();
      TheGroupCenterLine.StartY /= ((std::vector <LineSegmentStruct>)(*pgit)).size();
      TheGroupCenterLine.StartZ /= ((std::vector <LineSegmentStruct>)(*pgit)).size();
      TheGroupCenterLine.EndX /= ((std::vector <LineSegmentStruct>)(*pgit)).size();
      TheGroupCenterLine.EndY /= ((std::vector <LineSegmentStruct>)(*pgit)).size();
      TheGroupCenterLine.EndZ /= ((std::vector <LineSegmentStruct>)(*pgit)).size();
      RepGroup.push_back(TheGroupCenterLine);
      float CenterofGroupX = (TheGroupCenterLine.StartX+TheGroupCenterLine.EndX)/2.0;
      float CenterofGroupY = (TheGroupCenterLine.StartY+TheGroupCenterLine.EndY)/2.0;
      float CenterofGroupZ = (TheGroupCenterLine.StartZ+TheGroupCenterLine.EndZ)/2.0;
      float angle = (atan2(TheGroupCenterLine.EndY - TheGroupCenterLine.StartY, TheGroupCenterLine.EndX - TheGroupCenterLine.StartX) -  M_PI / 2.0)*180.0/M_PI;
      std::cout << "This group is represented by the line centered at: \n\t\t \033[4m["
                << CenterofGroupX <<", " << CenterofGroupY << ", " << CenterofGroupZ
                << "], Angle = ["<< angle<< "] Degrees\033[24m" << std::endl;
    }
    std::cout << "\033[0m\n";
  }
  DrawLines(&visArray_Pub,RepGroup,0.05);
  return RepGroup;
}

vector <LineSegmentStruct> DetectionClass::DetectPallets()
{
    vector <LineSegmentStruct> PalletLines;
    vector <LineSegmentStruct> DetectedPallets;

    PalletLines.clear();
    vector<sensor_msgs::LaserScan> TrimmedScans (Scans);
    int StartTrimZone = 0.4 * Scans[0].ranges.size();
    int EndTrimZone = 0.6 * Scans[0].ranges.size();
    std::cout << "-------------------------------------------"<<std::endl;
    std::cout << "------------    2D Analysis  --------------"<<std::endl;
    ROS_INFO ("Searching For Pallets, FOV = [%3.3f] Degrees", 0.2 * (Scans[0].angle_max- Scans[0].angle_min) * 180/M_PI);
    for (uint ScanCounter = 0; ScanCounter< Scans.size(); ScanCounter++)
    {
      for (uint RayCounter = 0; RayCounter< Scans[ScanCounter].ranges.size(); RayCounter++)
      {
        if (RayCounter < StartTrimZone || RayCounter > EndTrimZone)
          TrimmedScans[ScanCounter].ranges[RayCounter] = 0;
      }
    }
    for (uint Counter = 0; Counter < PosVector.size(); Counter++)
    {
      if (IsDebugging)
      {
        ROS_INFO ("Processing scan [%d] @ angle [%6.3f] degrees ...", Counter+1, PosVector[Counter]);
      }
      sensor_msgs::LaserScan LASERscan = TrimmedScans[Counter];
      std::list<LineSegmentStruct> LineSegments = GetLineSegments(LASERscan, MinSquareLineLenght, MaxSquareLineLenght, MininPointsCount);
      std::vector <LineSegmentStruct> candidatelines = DetectPalletFromThreePillarFrontFaceLines(LineSegments);
      candidatelines = CalculateZ(candidatelines,PosVector[Counter]);
//      candidatelines = RemoveObviousFalseCase(candidatelines,100,1.0,1.0,1.0);

      if (!candidatelines.empty())
      {
        for ( std::vector <LineSegmentStruct>::iterator it = candidatelines.begin(); it!=candidatelines.end(); ++it)
        {
          LineSegmentStruct candidateline = (LineSegmentStruct)(*it);
          float angle = (atan2(candidateline.EndY-candidateline.StartY,candidateline.EndX-candidateline.StartX)- M_PI/2.0) *180.0/M_PI;
          if (fabs(angle) < 45)
          {
            ROS_INFO ("Pallet candidate from scan [%d] @ angle [%6.3f] degrees: ( X = [%6.3f] , Y = [%6.3f] Theta = [%6.3f] )",
                      Counter+1, PosVector[Counter],
                      (candidateline.StartY + candidateline.EndY)/2.0,
                      (candidateline.StartX + candidateline.EndX)/2.0, angle);
            PalletLines.push_back(candidateline);
          }
        }
      }
    }
    if (Publish3D)
      sensor_msgs::PointCloud2 Cloud3d = ShowOutputCloud(TrimmedScans);
    if (!PalletLines.empty())
    {
      if (PalletLines.size()>1)
        for ( std::vector <LineSegmentStruct>::iterator it = PalletLines.begin(); it!=PalletLines.end(); ++it)
          for ( std::vector <LineSegmentStruct>::iterator it2 = it+1; it2!=PalletLines.end(); ++it2)
          {
            LineSegmentStruct candidateline = (LineSegmentStruct)(*it);
            LineSegmentStruct candidateline2 = (LineSegmentStruct)(*it2);
            float cen1 = (candidateline.StartZ+candidateline.EndZ)/2.0;
            float cen2 = (candidateline2.StartZ+candidateline2.EndZ)/2.0;
            if (fabs(cen1)<fabs(cen2))
              std::iter_swap (it,it2);
          }
      DetectedPallets = OutputResults(PalletLines);
    }

    if (PalletLines.empty())
    {
        ROS_INFO ("No Pallets Found At All !!!");
    }

    std::cout << "-------------------------------------------"<<std::endl;
    std::cout << "Waiting for the next request ..." << std::endl;
    ros::spinOnce();
    sleep(1);
    ros::spinOnce();
    return DetectedPallets;
}

sensor_msgs::LaserScan DetectionClass::PointCloud2LASERScan(sensor_msgs::PointCloud2 cloud_msg)
{
  // build laserscan output
  sensor_msgs::LaserScan output;
  output.header.frame_id = "/vertical_laser_link";

  output.angle_min = -M_PI / 2.0;
  output.angle_max = M_PI / 2.0;
  output.angle_increment = 1 * M_PI / 180.0;
  output.time_increment = 0.0;
  output.scan_time = 1 / 40.0;
  output.range_min = 0.50;
  output.range_max = 10;

  float max_height = 0.1;
  float min_height = -0.1;

  // determine amount of rays to create
  uint32_t ranges_size =
      std::ceil((output.angle_max - output.angle_min) / output.angle_increment);

  // initialize with zero
  output.ranges.assign(ranges_size, 0);

  sensor_msgs::PointCloud2ConstPtr cloud_out(
      new sensor_msgs::PointCloud2(cloud_msg));

  // Iterate through pointcloud
  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_out, "x"),
       iter_y(*cloud_out, "y"), iter_z(*cloud_out, "z");
       iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
  {


/*    if (*iter_z > max_height || *iter_z < min_height)
    {
      ROS_INFO("rejected for height %f not in range (%f, %f)\n", *iter_z,
               min_height, max_height);
      continue;
    }*/

    double range = 100;
    if (std::isnan(*iter_x) || std::isnan(*iter_y) || std::isnan(*iter_z))
    {
       range = 100;
    }
    else
    {
      range = hypot(*iter_x, *iter_y);
    }

    if (range < output.range_min)
    {
/*      ROS_INFO(
          "rejected for range %f below minimum value %f. Point: (%f, %f, %f)",
          range, output.range_min, *iter_x, *iter_y, *iter_z);*/
      continue;
    }

    double angle = atan2(*iter_y, *iter_x);
    if (angle < output.angle_min || angle > output.angle_max)
    {
/*      ROS_INFO("rejected for angle %f not in range (%f, %f)\n", angle,
               output.angle_min, output.angle_max);*/
      continue;
    }

    // overwrite range at laserscan ray if new range is smaller
    int index = (angle - output.angle_min) / output.angle_increment;
    if (output.ranges[index] <0.1)
    {
      output.ranges[index] = range;
    }
    if (range < output.ranges[index])
    {
      output.ranges[index] = range;
    }
  }

  for (int index = 1; index < output.ranges.size()-1 ; index++)
  {
    if (output.ranges[index]  <0.1 && output.ranges[index+1]>0.1 && output.ranges[index-1]>0.1)
      output.ranges[index] = (output.ranges[index+1]+output.ranges[index-1])/2.0;
  }

  Projectedlaserscan.publish(output);
  ros::spinOnce();

  sensor_msgs::LaserScan highresolutionoutput;
  highresolutionoutput.header.frame_id = "/vertical_laser_link";

  highresolutionoutput.angle_min = -M_PI / 2.0;
  highresolutionoutput.angle_max = M_PI / 2.0;
  highresolutionoutput.angle_increment = 0.25 * M_PI / 180.0;
  highresolutionoutput.time_increment = 0.0;
  highresolutionoutput.scan_time = 1 / 40.0;
  highresolutionoutput.range_min = 0.50;
  highresolutionoutput.range_max = 10;
  highresolutionoutput.header.frame_id = "/vertical_laser_link";

  // determine amount of rays to create
  ranges_size = std::ceil((highresolutionoutput.angle_max - highresolutionoutput.angle_min) / highresolutionoutput.angle_increment);

  // initialize with zero
  highresolutionoutput.ranges.assign(ranges_size, 0);

  for (int index = 0; index < output.ranges.size()-1 ; index++)
  {
    highresolutionoutput.ranges [4*index] = output.ranges[index];
    highresolutionoutput.ranges [4*index+1] = (3*output.ranges[index] + output.ranges[index+1])/4.0;
    highresolutionoutput.ranges [4*index+2] = (2*output.ranges[index] + 2*output.ranges[index+1])/4.0;
    highresolutionoutput.ranges [4*index+3] = (output.ranges[index] + 3*output.ranges[index+1])/4.0;
  }
  highresolutionoutput.ranges [highresolutionoutput.ranges.size()-1 ] = output.ranges [output.ranges.size()-1 ];
  Projectedlaserscan.publish(highresolutionoutput);
  ros::spinOnce();
  return (highresolutionoutput);
}

void ReadParams(ros::NodeHandle nh)
{
  std::string param_name(ros::this_node::getName());
  param_name.append("/ShouldReportParams");
  bool ShouldReportParams = true;
  if (nh.hasParam(param_name))
  {
    nh.getParam(param_name, ShouldReportParams);
  }
  else
  {
    ROS_WARN("ShouldReportParams was not set as param. All parameter readings will be reported ...");
  }
  if (ShouldReportParams)
  {
    param_name = ros::this_node::getName();
    param_name.append("/StartSwipAngle");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, StartSwipAngle);
      ROS_INFO("StartSwipAngle is set as external parameter [%f]", StartSwipAngle);
    }
    else
    {
      ROS_WARN("StartSwipAngle is (!!!)NOT(!!!) set as external parameter. Default value  [%f] used.", StartSwipAngle);
    }

    param_name = ros::this_node::getName();
    param_name.append("/EndSwipAngle");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, EndSwipAngle);
      ROS_INFO("EndSwipAngle is set as external parameter [%f]", EndSwipAngle);
    }
    else
    {
      ROS_WARN("EndSwipAngle is (!!!)NOT(!!!) set as external parameter. Default value  [%f] used.", EndSwipAngle);
    }

    param_name = ros::this_node::getName();
    param_name.append("/MinCaptureAngle");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, MinCaptureAngle);
      ROS_INFO("MinCaptureAngle is set as external parameter [%f]", MinCaptureAngle);
    }
    else
    {
      ROS_WARN("MinCaptureAngle is (!!!)NOT(!!!) set as external parameter. Default value [%f] used.", MinCaptureAngle);
    }

    param_name = ros::this_node::getName();
    param_name.append("/MaxCaptureAngle");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, MaxCaptureAngle);
      ROS_INFO("MaxCaptureAngle is set as external parameter [%f]", MaxCaptureAngle);
    }
    else
    {
      ROS_WARN("MaxCaptureAngle is (!!!)NOT(!!!) set as external parameter. Default value [%f] used.", MaxCaptureAngle);
    }

    param_name = ros::this_node::getName();
    param_name.append("/HomeAngle");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, HomeAngle);
      ROS_INFO("HomeAngle is set as external parameter [%f]", HomeAngle);
    }
    else
    {
      ROS_WARN("HomeAngle is (!!!)NOT(!!!) set as external parameter. Default value [%f] used.", HomeAngle);
    }

    param_name = ros::this_node::getName();
    param_name.append("/HomeSpeed");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, HomeSpeed);
      ROS_INFO("HomeSpeed is set as external parameter [%d]", HomeSpeed);
    }
    else
    {
      ROS_WARN("HomeSpeed is (!!!)NOT(!!!) set as external parameter. Default value [%d] used.", HomeSpeed);
    }

    param_name = ros::this_node::getName();
    param_name.append("/RotationSpeed");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, RotationSpeed);
      ROS_INFO("RotationSpeed is set as external parameter [%d]", RotationSpeed);
    }
    else
    {
      ROS_WARN("RotationSpeed is (!!!)NOT(!!!) set as external parameter. Default value [%d] used.", RotationSpeed);
    }

    param_name = ros::this_node::getName();
    param_name.append("/RotationStepSize");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, RotationStepSize);
      ROS_INFO("RotationStepSize is set as external parameter [%d]", RotationStepSize);
    }
    else
    {
      ROS_WARN("RotationStepSize is (!!!)NOT(!!!) set as external parameter. Default value [%d] used.", RotationStepSize);
    }

    param_name = ros::this_node::getName();
    param_name.append("/MinSquareLineLenght");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, MinSquareLineLenght);
      ROS_INFO("MinSquareLineLenght is set as external parameter [%f]", MinSquareLineLenght);
    }
    else
    {
      ROS_WARN("MinSquareLineLenght is (!!!)NOT(!!!) set as external parameter. Default value [%f] used.", MinSquareLineLenght);
    }

    param_name = ros::this_node::getName();
    param_name.append("/MaxSquareLineLenght");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, MaxSquareLineLenght);
      ROS_INFO("MaxSquareLineLenght is set as external parameter [%f]", MaxSquareLineLenght);
    }
    else
    {
      ROS_WARN("MaxSquareLineLenght is (!!!)NOT(!!!) set as external parameter. Default value [%f] used.", MaxSquareLineLenght);
    }

    param_name = ros::this_node::getName();
    param_name.append("/MininPointsCount");
    if (nh.hasParam(param_name))
    {
      int dummy;
      nh.getParam(param_name, dummy);
      MininPointsCount = dummy;
      ROS_INFO("MininPointsCount is set as external parameter [%d]", dummy);
    }
    else
    {
      ROS_WARN("MininPointsCount is (!!!)NOT(!!!) set as external parameter. Default value [%d] used.", MininPointsCount);
    }

    param_name = ros::this_node::getName();
    param_name.append("/AllowedPointDistance");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, AllowedPointDistance);
      ROS_INFO("AllowedPointDistance is set as external parameter [%f]", AllowedPointDistance);
    }
    else
    {
      ROS_WARN("AllowedPointDistance is (!!!)NOT(!!!) set as external parameter. Default value [%f] used.", AllowedPointDistance);
    }

    param_name = ros::this_node::getName();
    param_name.append("/MinPointDistanceTolerance");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, MinPointDistanceTolerance);
      ROS_INFO("MinPointDistanceTolerance is set as external parameter [%f]", MinPointDistanceTolerance);
    }
    else
    {
      ROS_WARN("MinPointDistanceTolerance is (!!!)NOT(!!!) set as external parameter. Default value [%f] used.", MinPointDistanceTolerance);
    }

    param_name = ros::this_node::getName();
    param_name.append("/PalletpillarThicknessMax");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, PalletpillarThicknessMax);
      ROS_INFO("PalletpillarThicknessMax is set as external parameter [%f]", PalletpillarThicknessMax);
    }
    else
    {
      ROS_WARN("PalletpillarThicknessMax is (!!!)NOT(!!!) set as external parameter. Default value [%f] used.", PalletpillarThicknessMax);
    }

    param_name = ros::this_node::getName();
    param_name.append("/ForkWindowWidth");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, ForkWindowWidth);
      ROS_INFO("ForkWindowWidth is set as external parameter [%f]", ForkWindowWidth);
    }
    else
    {
      ROS_WARN("ForkWindowWidth is (!!!)NOT(!!!) set as external parameter. Default value [%f] used.", ForkWindowWidth);
    }

    param_name = ros::this_node::getName();
    param_name.append("/ForkWindowWidthTolerance");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, ForkWindowWidthTolerance);
      ROS_INFO("ForkWindowWidthTolerance is set as external parameter [%f]", ForkWindowWidthTolerance);
    }
    else
    {
      ROS_WARN("ForkWindowWidthTolerance is (!!!)NOT(!!!) set as external parameter. Default value [%f] used.", ForkWindowWidthTolerance);
    }

    param_name = ros::this_node::getName();
    param_name.append("/MidPilarPointFromSidePilarLineDistanceMax");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, MidPilarPointFromSidePilarLineDistanceMax);
      ROS_INFO("MidPilarPointFromSidePilarLineDistanceMax is set as external parameter [%f]", MidPilarPointFromSidePilarLineDistanceMax);
    }
    else
    {
      ROS_WARN("MidPilarPointFromSidePilarLineDistanceMax is (!!!)NOT(!!!) set as external parameter [%f]. Default value used.", MidPilarPointFromSidePilarLineDistanceMax);
    }


    param_name = ros::this_node::getName();
    param_name.append("/SensorElevation");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, SensorElevation);
      ROS_INFO("SensorElevation is set as external parameter [%f]", SensorElevation);
    }
    else
    {
      ROS_WARN("SensorElevation is (!!!)NOT(!!!) set as external parameter [%f]. Default value used.", SensorElevation);
    }

    param_name = ros::this_node::getName();
    param_name.append("/Publish3D");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, Publish3D);
      ROS_INFO("Publish3D is set as external parameter [%s]", Publish3D ? "true" : "false");
    }
    else
    {
      ROS_WARN("Publish3D is (!!!)NOT(!!!) set as external parameter. Default value [%s] used.", Publish3D ? "true" : "false");
    }

    param_name = ros::this_node::getName();
    param_name.append("/PublishRvizPalletLine");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, PublishRvizPalletLine);
      ROS_INFO("PublishRvizPalletLine is set as external parameter [%s]", PublishRvizPalletLine ? "true" : "false");
    }
    else
    {
      ROS_WARN("PublishRvizPalletLine is (!!!)NOT(!!!) set as external parameter. Default value [%s] used.", PublishRvizPalletLine ? "true" : "false");
    }

    param_name = ros::this_node::getName();
    param_name.append("/PalletWidthFor3DAnalysis");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, PalletWidthFor3DAnalysis);
      ROS_INFO("PalletWidthFor3DAnalysis is set as external parameter [%f]", PalletWidthFor3DAnalysis);
    }
    else
    {
      ROS_WARN("PalletWidthFor3DAnalysis is (!!!)NOT(!!!) set as external parameter [%f]. Default value used.", PalletWidthFor3DAnalysis);
    }

    param_name = ros::this_node::getName();
    param_name.append("/PalletWidthToleranceFor3DAnalysis");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, PalletWidthToleranceFor3DAnalysis);
      ROS_INFO("PalletWidthToleranceFor3DAnalysis is set as external parameter [%f]", PalletWidthToleranceFor3DAnalysis);
    }
    else
    {
      ROS_WARN("PalletWidthToleranceFor3DAnalysis is (!!!)NOT(!!!) set as external parameter [%f]. Default value used.", PalletWidthToleranceFor3DAnalysis);
    }

    param_name = ros::this_node::getName();
    param_name.append("/PalletAngleToleranceFor3DAnalysis");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, PalletAngleToleranceFor3DAnalysis);
      ROS_INFO("PalletAngleToleranceFor3DAnalysis is set as external parameter [%f]", PalletAngleToleranceFor3DAnalysis);
    }
    else
    {
      ROS_WARN("PalletAngleToleranceFor3DAnalysis is (!!!)NOT(!!!) set as external parameter [%f]. Default value used.", PalletAngleToleranceFor3DAnalysis);
    }

    param_name = ros::this_node::getName();
    param_name.append("/PalletCenterToleranceFor3DAnalysis");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, PalletCenterToleranceFor3DAnalysis);
      ROS_INFO("PalletCenterToleranceFor3DAnalysis is set as external parameter [%f]", PalletCenterToleranceFor3DAnalysis);
    }
    else
    {
      ROS_WARN("PalletCenterToleranceFor3DAnalysis is (!!!)NOT(!!!) set as external parameter [%f]. Default value used.", PalletCenterToleranceFor3DAnalysis);
    }

    param_name = ros::this_node::getName();
    param_name.append("/ShouldReadScansFromFile");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, ShouldReadScansFromFile);
      ROS_INFO("ShouldReadScansFromFile is set as external parameter [%s]", ShouldReadScansFromFile ? "true" : "false");
      if(ShouldReadScansFromFile)
      {
        param_name = ros::this_node::getName();
        param_name.append("/PosFileName");
        if (nh.hasParam(param_name))
        {
          nh.getParam(param_name, PosFileName);
          ROS_INFO("PosFileName is set as external parameter [%s]", PosFileName.c_str());
        }
        param_name = ros::this_node::getName();
        param_name.append("/ScansFileName");
        if (nh.hasParam(param_name))
        {
          nh.getParam(param_name, ScansFileName);
          ROS_INFO("ScansFileName is set as external parameter [%s]", ScansFileName.c_str());
        }
      }
    }

    param_name = ros::this_node::getName();
    param_name.append("/GroupingMaxDistance");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, GroupingMaxDistance);
      ROS_INFO("GroupingMaxDistance is set as external parameter [%3.3f]", GroupingMaxDistance);
    }
    else
    {
      ROS_WARN("GroupingMaxDistance is (!!!)NOT(!!!) set as external parameter. Default value [%3.3f] used.", GroupingMaxDistance);
    }

    param_name = ros::this_node::getName();
    param_name.append("/IsDebugging");
    if (nh.hasParam(param_name))
    {
      nh.getParam(param_name, IsDebugging);
      ROS_INFO("IsDebugging is set as external parameter [%s]", IsDebugging ? "true" : "false");
    }
    else
    {
      ROS_WARN("IsDebugging is (!!!)NOT(!!!) set as external parameter. Default value [%s] used.", IsDebugging ? "true" : "false");
    }

  }
}

void LoadScansFromFile()
{
    std::ifstream ifs(PosFileName.c_str(), std::ios::in|std::ios::binary);
    ifs.seekg (0, std::ios::end);
    std::streampos end = ifs.tellg();
    ifs.seekg (0, std::ios::beg);
    std::streampos begin = ifs.tellg();

    uint32_t file_size = end-begin;
    boost::shared_array<uint8_t> ibuffer(new uint8_t[file_size]);
    ifs.read((char*) ibuffer.get(), file_size);
    ros::serialization::IStream istream(ibuffer.get(), file_size);
    ros::serialization::deserialize(istream, PosVector);
    ifs.close();

    std::ifstream ifs2(ScansFileName.c_str(), std::ios::in|std::ios::binary);
    ifs2.seekg (0, std::ios::end);
    std::streampos end2 = ifs2.tellg();
    ifs2.seekg (0, std::ios::beg);
    std::streampos begin2 = ifs2.tellg();

    uint32_t file_size2 = end2-begin2;
    boost::shared_array<uint8_t> ibuffer2(new uint8_t[file_size2]);
    ifs2.read((char*) ibuffer2.get(), file_size2);
    ros::serialization::IStream istream2(ibuffer2.get(), file_size2);
    ros::serialization::deserialize(istream2, Scans);
    ifs2.close();
}

bool ReportPalletServiceHandler(palletunloader::pallet_report_srv::Request  &req, palletunloader::pallet_report_srv::Response &res)
{

  NewPosAvailable = false;
  NewPos = 0;
  PosVector.clear();
  Scans.clear();
  DetectionClass detectionobject;
  /***********To Study The ode through file loading***********/
  if (ShouldReadScansFromFile)
  {
    LoadScansFromFile();
  }
  /*************************/
  else
  {
    CurrentPalletDetectionState = PDS_ServiceRequested;
    detectionobject.SweepAndScan();
  }
  std::cout << "*** Scan Complete ***" << std::endl;
  vector <LineSegmentStruct> DetectedPallets = detectionobject.DetectPallets();
  geometry_msgs::Pose dummypos;
  dummypos.position.x = 0;
  dummypos.position.y = 0;
  dummypos.position.z = 0;
  res.Pallet1 = dummypos;
  res.Pallet2 = dummypos;
  res.Pallet3 = dummypos;
  res.Pallet4 = dummypos;
  res.Camera_Image = ImageToReport;
  res.LiDAR_PointCloud = detectionobject.ShowOutputCloud(Scans);
  res.header.stamp = ros::Time::now();
  res.header.frame_id = "/vertical_laser_link";
  if (DetectedPallets.size()>0)
  {
    dummypos.position.x = (DetectedPallets[0].StartX + DetectedPallets[0].EndX)/2.0;
    dummypos.position.y = (DetectedPallets[0].StartY + DetectedPallets[0].EndY)/2.0;
    dummypos.position.z = (DetectedPallets[0].StartZ + DetectedPallets[0].EndZ)/2.0;
    res.Pallet1 = dummypos;

    if (DetectedPallets.size()>1)
    {
      dummypos.position.x = (DetectedPallets[1].StartX + DetectedPallets[1].EndX)/2.0;
      dummypos.position.y = (DetectedPallets[1].StartY + DetectedPallets[1].EndY)/2.0;
      dummypos.position.z = (DetectedPallets[1].StartZ + DetectedPallets[1].EndZ)/2.0;
      res.Pallet2 = dummypos;

      if (DetectedPallets.size()>2)
      {
        dummypos.position.x = (DetectedPallets[2].StartX + DetectedPallets[2].EndX)/2.0;
        dummypos.position.y = (DetectedPallets[2].StartY + DetectedPallets[2].EndY)/2.0;
        dummypos.position.z = (DetectedPallets[2].StartZ + DetectedPallets[2].EndZ)/2.0;
        res.Pallet3 = dummypos;

        if (DetectedPallets.size()>3)
        {
          dummypos.position.x = (DetectedPallets[3].StartX + DetectedPallets[3].EndX)/2.0;
          dummypos.position.y = (DetectedPallets[3].StartY + DetectedPallets[3].EndY)/2.0;
          dummypos.position.z = (DetectedPallets[3].StartZ + DetectedPallets[3].EndZ)/2.0;
          res.Pallet4 = dummypos;
        }
      }
    }
  }
  palletunloader::pallet_report_msg RptMsg;
  RptMsg.Camera_Image = res.Camera_Image;
  RptMsg.LiDAR_PointCloud = res.LiDAR_PointCloud;
  RptMsg.Pallet1 = res.Pallet1;
  RptMsg.Pallet2 = res.Pallet2;
  RptMsg.Pallet3 = res.Pallet3;
  RptMsg.Pallet4 = res.Pallet4;
  RptMsg.header = res.header;
  ReportPublisher.publish(RptMsg);

  ros::spinOnce();
  return true;
}

bool WhichPalletServiceHandler(palletunloader::which_pallet_srv::Request  &req, palletunloader::which_pallet_srv::Response &res)
{

  NewPosAvailable = false;
  NewPos = 0;
  PosVector.clear();
  Scans.clear();
  DetectionClass detectionobject;
  /***********To Study The code through file loading***********/
  if (ShouldReadScansFromFile)
  {
    LoadScansFromFile();
  }
  /*************************/
  else
  {
    CurrentPalletDetectionState = PDS_ServiceRequested;
    detectionobject.SweepAndScan();
  }
  std::cout << "*** Scan Complete ***" << std::endl;
  vector <LineSegmentStruct> DetectedPallets = detectionobject.DetectPallets();
  geometry_msgs::Pose dummypos;
  dummypos.position.x = 0;
  dummypos.position.y = 0;
  dummypos.position.z = 0;
  res.Pallet = dummypos;
  res.Yaw_2D_Radian = 0;
  if (DetectedPallets.size()>0)
  {
    dummypos.position.x = (DetectedPallets[0].StartX + DetectedPallets[0].EndX)/2.0;
    dummypos.position.y = (DetectedPallets[0].StartY + DetectedPallets[0].EndY)/2.0;
    dummypos.position.z = (DetectedPallets[0].StartZ + DetectedPallets[0].EndZ)/2.0;
    res.Pallet = dummypos;
    res.Yaw_2D_Radian = atan2( DetectedPallets[0].EndY- DetectedPallets[0].StartY, DetectedPallets[0].EndX- DetectedPallets[0].StartX)- M_PI/2.0;
  }
  res.header.stamp = ros::Time::now();
  res.header.frame_id = "/vertical_laser_link";
  return true;
}

void LASERCallback (const sensor_msgs::LaserScan::ConstPtr& scan_in)
{
  if (CurrentPalletDetectionState == PDS_Scanning)
  {
    if (NewPosAvailable)
    {
      float LocalNewPos = NewPos;
      NewPosAvailable =false;
      if (LocalNewPos>MinCaptureAngle && LocalNewPos < MaxCaptureAngle)
      {
        if (std::find(PosVector.begin(), PosVector.end(), LocalNewPos) == PosVector.end())
        {
          PosVector.push_back(LocalNewPos);
          Scans.push_back(*scan_in);
          ROS_INFO ("Appending Scan #[%d] @ [%6.3f] Degrees. ", (int) PosVector.size(), LocalNewPos);
        }
      }
    }
  }
}

void RotorStatusCallback(const serialrotor::serialrotorstatus::ConstPtr& msg)
{
    NewPos = msg->EncoderPosDegree;
    NewPosAvailable =true;
}

void imageCallback(const sensor_msgs::CompressedImageConstPtr& msg)
{
  ImageToReport = *msg;
}

int main (int argc, char** argv)
{
  ros::init (argc, argv, "palletunloader");

  ros::NodeHandle nh;
  ReadParams(nh);

  ros::ServiceServer report_pallet_service = nh.advertiseService("/PalletUnloaderService", ReportPalletServiceHandler);
  ros::ServiceServer which_pallet_service = nh.advertiseService("/which_pallet_service", WhichPalletServiceHandler);
  client = nh.serviceClient<serialrotor::serialrotorposservice>("/rotor_service");

  LASERsub = nh.subscribe <sensor_msgs::LaserScan> ("/vertical_scan", 1, &LASERCallback);
  RotorStatussb = nh.subscribe <serialrotor::serialrotorstatus>("/rotor_status", 1, &RotorStatusCallback);
  Camerasub = nh.subscribe("/left/image_raw/compressed", 1, imageCallback);
  pub3D = nh.advertise <sensor_msgs::PointCloud2> ("output3D", 1);
  vis_pub = nh.advertise<visualization_msgs::Marker>( "/visualization_marker", 1 );
  visArray_Pub = nh.advertise<visualization_msgs::MarkerArray>( "/visualization_marker_array", 1 );
  Projectedlaserscan = nh.advertise<sensor_msgs::LaserScan>("projectedlaserscan", 1);
  ReportPublisher = nh.advertise<palletunloader::pallet_report_msg>("/PalletReportMsg", 1);

//  sleep(1);

  ros::spin();
  return (0);
}
