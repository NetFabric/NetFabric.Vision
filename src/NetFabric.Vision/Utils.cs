using System;
using OpenCV.Net;

namespace NetFabric.Vision
{
    static class Utils
    {
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
    }
}
