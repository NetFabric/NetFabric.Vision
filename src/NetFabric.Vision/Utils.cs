using System;
using System.Collections.Generic;
using OpenCV.Net;

namespace NetFabric.Vision
{
    static class Utils
    {
        const byte HorizontalValue = 0;
        const byte VerticalValue = 255;
        const byte EdgeValue = 0;

        static readonly Mat PrewittX = Mat.FromArray(new float[] { -1, 0, 1, -1, 0, 1, -1, 0, 1 });
        static readonly Mat PrewittY = Mat.FromArray(new float[] { -1, -1, -1, 0, 0, 0, 1, 1, 1 });
        static readonly Mat ScharrX = Mat.FromArray(new float[] { -3, 0, 3, -10, 0, 10, -3, 0, 3 });
        static readonly Mat ScharrY = Mat.FromArray(new float[] { -3, -10, -3, 0, 0, 0, 3, 10, 3 });

        public static void ComputeGradient(Mat source, GradientOperator gradientOperator, double gradientThreshold,
            Mat gradientX, Mat gradientY, Mat absGradientX, Mat absGradientY,
            Mat gradientMap, Mat directionMap)
        {
            // calculate gradients
            switch(gradientOperator)
            {
                case GradientOperator.Prewitt:
                    CV.Filter2D(source, gradientX, PrewittX);
                    CV.Filter2D(source, gradientY, PrewittY);
                    if(gradientThreshold < 0.0)
                        gradientThreshold = 6.0;
                    break;
                case GradientOperator.Sobel:
                    CV.Sobel(source, gradientX, 1, 0);
                    CV.Sobel(source, gradientY, 0, 1);
                    break;
                case GradientOperator.Scharr:
                    CV.Filter2D(source, gradientX, PrewittX);
                    CV.Filter2D(source, gradientY, PrewittY);
                    break;
                default:
                    throw new Exception($"Unknown gradient operator: {gradientOperator}");
            }

            // calculate absolute values for gradients
            CV.ConvertScaleAbs(gradientX, absGradientX);
            CV.ConvertScaleAbs(gradientY, absGradientY);

            // merge gradients  
            // d = 0.5 * abs(dx) + 0.5 * abs(dy)
            CV.AddWeighted(absGradientX, 0.5, absGradientY, 0.5, 0.0, gradientMap);

            // eliminate gradient weak pixels
            CV.Threshold(gradientMap, gradientMap, gradientThreshold, 255, ThresholdTypes.ToZero);

            // edge direction 
            // abs(dx) >= abs(dy) => VERTICAL
            CV.Cmp(absGradientX, absGradientY, directionMap, ComparisonOperation.GreaterOrEqual);
        }

        public static List<Point> ExtractAnchors(Mat gradientMap, Mat directionMap, int scanInterval, int threshold)
        {
            var anchors = new List<Point>();

            // iterate through the Rows
            for(int row = 1, rowEnd = gradientMap.Rows - 1; row < rowEnd; row += scanInterval)
            {
                // iterate through the columns
                for(int col = 1, colEnd = gradientMap.Cols - 1; col < colEnd; col += scanInterval)
                {
                    var g = (byte)gradientMap.GetReal(row, col);

                    if((byte)directionMap.GetReal(row, col) == HorizontalValue)
                    {
                        // compare to horizontal neighbors
                        if(Math.Abs(g - (byte)gradientMap.GetReal(row - 1, col)) > threshold &&
                            Math.Abs(g - (byte)gradientMap.GetReal(row + 1, col)) > threshold)
                        {
                            anchors.Add(new Point(col, row));
                        }
                    }
                    else
                    {
                        // compare to vertical neighbors
                        if(Math.Abs(g - (byte)gradientMap.GetReal(row, col - 1)) > threshold &&
                            Math.Abs(g - (byte)gradientMap.GetReal(row, col + 1)) > threshold)
                        {
                            anchors.Add(new Point(col, row));
                        }
                    }
                }
            }

            return anchors;
        }

        public static unsafe List<Point>[] ExtractAnchorsParameterFree(Mat gradientMap, Mat directionMap, int scanInterval)
        {
            var anchors = new List<Point>[256];

            // iterate through the Rows
            for(int row = 1, rowEnd = gradientMap.Rows - 1; row < rowEnd; row += scanInterval)
            {
                // iterate through the columns
                for(int col = 1, colEnd = gradientMap.Cols - 1; col < colEnd; col += scanInterval)
                {
                    var g = (byte)gradientMap.GetReal(row, col);

                    if((byte)directionMap.GetReal(row, col) == HorizontalValue)
                    {
                        // compare to horizontal neighbors
                        if(g > (byte)gradientMap.GetReal(row - 1, col) && g > (byte)gradientMap.GetReal(row + 1, col))
                        {
                            anchors[g].Add(new Point(col, row));
                        }
                    }
                    else
                    {
                        // compare to vertical neighbors
                        if(g > (byte)gradientMap.GetReal(row, col - 1) && g > (byte)gradientMap.GetReal(row, col + 1))
                        {
                            anchors[g].Add(new Point(col, row));
                        }
                    }
                }
            }

            return anchors;
        }
    }
}
