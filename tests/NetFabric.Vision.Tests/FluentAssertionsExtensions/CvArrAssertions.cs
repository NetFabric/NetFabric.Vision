using System;
using FluentAssertions;
using FluentAssertions.Execution;
using FluentAssertions.Primitives;
using OpenCV.Net;

namespace NetFabric.Vision.Tests
{
    /// <summary>
    /// Contains a number of methods to assert that an <see cref="Arr"/> is in the expected state.
    /// </summary>
    public class CvArrAssertions<TAssertions> : ReferenceTypeAssertions<Arr, TAssertions>
        where TAssertions : CvArrAssertions<TAssertions>
    {
        protected override string Identifier => "arr";

        /// <summary>
        /// Expects the current <see cref="Arr"/> to have the same size, same channels and contain all the same values in the same positions as the <see cref="Arr"/> identified by
        /// <paramref name="elements" />. 
        /// </summary>
        /// <param name="elements">A params array with the expected elements.</param>
        public AndConstraint<TAssertions> Equal(params object[] elements) => Equal(elements, String.Empty);

        /// <summary>
        /// Expects the current <see cref="Arr"/> to have the same size, same channels and contain all the same values in the same positions as the <see cref="Arr"/> identified by
        /// <paramref name="expected" />. 
        /// </summary>
        /// <param name="expected">An <see cref="Arr"/> with the expected elements.</param>
        /// <param name="because">
        /// A formatted phrase as is supported by <see cref="String.Format(String,Object[])" /> explaining why the assertion
        /// is needed. If the phrase does not start with the word <i>because</i>, it is prepended automatically.
        /// </param>
        /// <param name="becauseArgs">
        /// Zero or more objects to format using the placeholders in <see cref="because" />.
        /// </param>
        public AndConstraint<TAssertions> Equal(Arr expected, string because = "", params object[] becauseArgs)
        {
            AssertSubjectEquality(expected, 0);

            return new AndConstraint<TAssertions>((TAssertions)this);
        }

        protected void AssertSubjectEquality(Arr expectation, int precision,
            string because = "", params object[] becauseArgs)
        {
            var subjectIsNull = Subject is null;
            var expectationIsNull = expectation is null;
            if(subjectIsNull && expectationIsNull)
            {
                return;
            }

            if(expectation == null)
            {
                throw new ArgumentNullException(nameof(expectation), "Cannot compare arr with <null>.");
            }

            var assertion = Execute.Assertion.BecauseOf(because, becauseArgs);
            if(subjectIsNull)
            {
                Execute.Assertion
                    .BecauseOf(because, becauseArgs)
                    .FailWith("Expected arr to be equal{reason}, but found <null>.");
            }

            assertion
                .WithExpectation("Expected {context:arr} to be equal {reason}, ")
                .Given(() => Subject)
                .AssertEitherArrHaveSameSizeAndElementType(expectation.Size, expectation.ElementType)
                .Then
                .AssertArrsHaveSameValues(expectation);
        }


    }
}
