using System;
using System.Linq;
using System.Reactive.Linq;
using Bonsai;
using Bonsai.Vision;
using OpenCV.Net;

namespace NetFabric.Vision.Bonsai
{
    public class EdgeDrawing : Transform<IplImage, Contours>
    {
        public override IObservable<Contours> Process(IObservable<IplImage> source)
        {
            return source.Select<IplImage, Contours>(input =>
            {
                // TODO: process the input object and return the result.
                throw new NotImplementedException();
                return default;
            });
        }
    }
}

