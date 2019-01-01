using System;
using System.Collections.Generic;
using OpenCV.Net;

namespace NetFabric.Vision
{
    public partial class EdgeDrawing
    {
        const byte EdgeValue = 0;

        readonly int _rows;
        readonly int _columns;
        readonly Gradients _gradients;
        Mat _edgesMap;

        public EdgeDrawing(int rows, int columns)
        {
            _rows = rows;
            _columns = columns;
            _gradients = new Gradients(rows, columns);
        }


    }
}
