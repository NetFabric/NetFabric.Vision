using System;
using OpenCV.Net;

namespace NetFabric.Vision
{
    public partial class EdgeDrawing
    {
        public void DrawEdges(Arr source, Arr destination, GradientOperator gradientOperator, int gradientThreshold, int anchorScanInterval, int anchorThreshold)
        {
            // compute the gradient and direction maps
            _gradients.ComputeGradient(source, gradientOperator, gradientThreshold);

            // compute the anchors
            var anchors = _gradients.ExtractAnchors(anchorScanInterval, anchorThreshold);

            // connect the anchors by smart routing
            foreach(var anchor in anchors)
            {
                DrawEdge(anchor.Y, anchor.X, _gradients.GradientMap, _gradients.DirectionMap, destination, _rows, _columns);
            }
        }

        internal static void DrawEdge(int row, int column, Arr gradientMap, Arr directionMap, Arr edgeMap, int rows, int columns)
        {
            var edgel = edgeMap.GetReal(row, column);
            if(edgel != 0)
                return;

            var direction = directionMap.GetReal(row, column);
            if(direction == Gradients.HorizontalValue)
            { // go horizontal
                GoLeft(row, column, gradientMap, directionMap, edgeMap, rows, columns);
                GoRight(row, column, gradientMap, directionMap, edgeMap, rows, columns);
            }
            else
            { // go vertical
                GoUp(row, column, gradientMap, directionMap, edgeMap, rows, columns);
                GoDown(row, column, gradientMap, directionMap, edgeMap, rows, columns);
            }
        }

        static void GoLeft(int row, int column, Arr gradientMap, Arr directionMap, Arr edgeMap, int rows, int columns)
        {
            double upGradient, straightGradient, downGradient, direction, edgel;
            do
            {
                edgeMap.SetReal(row, column, 255);

                --column; // left
                if(column == 0)
                    return;

                upGradient = gradientMap.GetReal(row - 1, column);
                straightGradient = gradientMap.GetReal(row, column);
                downGradient = gradientMap.GetReal(row + 1, column);

                if(upGradient > straightGradient && upGradient > downGradient)
                {
                    --row; // up
                    if(row == 0)
                        return;
                }
                else if(downGradient > straightGradient && downGradient > upGradient)
                {
                    ++row; // down
                    if(row == rows - 1)
                        return;
                }
                else if(straightGradient == 0)
                {
                    return;
                }

                edgel = edgeMap.GetReal(row, column);
                if(edgel != 0)
                    return;

                direction = directionMap.GetReal(row, column);
            }
            while(direction == Gradients.HorizontalValue);

            // go vertical
            GoUp(row, column, gradientMap, directionMap, edgeMap, rows, columns);
            GoDown(row, column, gradientMap, directionMap, edgeMap, rows, columns);
        }

        static void GoRight(int row, int column, Arr gradientMap, Arr directionMap, Arr edgeMap, int rows, int columns)
        {
            double upGradient, straightGradient, downGradient, direction, edgel;
            do
            {
                edgeMap.SetReal(row, column, 255);

                ++column; // right
                if(column == columns - 1)
                    return;

                upGradient = gradientMap.GetReal(row - 1, column);
                straightGradient = gradientMap.GetReal(row, column);
                downGradient = gradientMap.GetReal(row + 1, column);

                if(upGradient > straightGradient && upGradient > downGradient)
                {
                    --row; // up
                    if(row == 0)
                        return;
                }
                else if(downGradient > straightGradient && downGradient > upGradient)
                {
                    ++row; // down
                    if(row == rows - 1)
                        return;
                }
                else if(straightGradient == 0)
                {
                    return;
                }

                edgel = edgeMap.GetReal(row, column);
                if(edgel != 0)
                    return;

                direction = directionMap.GetReal(row, column);
            }
            while(direction == Gradients.HorizontalValue);

            // go vertical
            GoUp(row, column, gradientMap, directionMap, edgeMap, rows, columns);
            GoDown(row, column, gradientMap, directionMap, edgeMap, rows, columns);
        }

        static void GoUp(int row, int column, Arr gradientMap, Arr directionMap, Arr edgeMap, int rows, int columns)
        {
            double leftGradient, straightGradient, rightGradient, direction, edgel;
            do
            {
                edgeMap.SetReal(row, column, 255);

                --row; // up
                if(row == 0)
                    return;

                leftGradient = gradientMap.GetReal(row, column - 1);
                straightGradient = gradientMap.GetReal(row, column);
                rightGradient = gradientMap.GetReal(row, column + 1);

                if(leftGradient > straightGradient && leftGradient > rightGradient)
                {
                    --column; // left
                    if(column == 0)
                        return;
                }
                else if(rightGradient > straightGradient && rightGradient > leftGradient)
                {
                    ++column; // right
                    if(column == columns - 1)
                        return;
                }
                else if(straightGradient == 0)
                {
                    return;
                }

                edgel = edgeMap.GetReal(row, column);
                if(edgel != 0)
                    return;

                direction = directionMap.GetReal(row, column);
            }
            while(direction == Gradients.VerticalValue);

            // go horizontal
            GoLeft(row, column, gradientMap, directionMap, edgeMap, rows, columns);
            GoRight(row, column, gradientMap, directionMap, edgeMap, rows, columns);
        }

        static void GoDown(int row, int column, Arr gradientMap, Arr directionMap, Arr edgeMap, int rows, int columns)
        {
            double leftGradient, straightGradient, rightGradient, direction, edgel;
            do
            {
                edgeMap.SetReal(row, column, 255);

                ++row; // down
                if(row == rows - 1)
                    return;

                leftGradient = gradientMap.GetReal(row, column - 1);
                straightGradient = gradientMap.GetReal(row, column);
                rightGradient = gradientMap.GetReal(row, column + 1);

                if(leftGradient > straightGradient && leftGradient > rightGradient)
                {
                    --column; // left
                    if(column == 0)
                        return;
                }
                else if(rightGradient > straightGradient && rightGradient > leftGradient)
                {
                    ++column; // right
                    if(column == columns - 1)
                        return;
                }
                else if(straightGradient == 0)
                {
                    return;
                }

                edgel = edgeMap.GetReal(row, column);
                if(edgel != 0)
                    return;

                direction = directionMap.GetReal(row, column);
            }
            while(direction == Gradients.VerticalValue);

            // go horizontal
            GoLeft(row, column, gradientMap, directionMap, edgeMap, rows, columns);
            GoRight(row, column, gradientMap, directionMap, edgeMap, rows, columns);
        }
    }
}
