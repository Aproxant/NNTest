using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackpropagationNN
{
    public static class ActivationFunctions
    {
        public static double[] Sigmoid(double[] x)
        {
            double[] outputActi = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                outputActi[i] = 1f / (1f + Math.Exp(-x[i]));
            }
            return outputActi;
        }

        public static double[] SigmoidDerivative(double[] x)
        {
            double[] res = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                res[i] = x[i] * (1 - x[i]);
            }
            return res;
        }
        public static double[] SoftMax(double[] x)
        {
            double[] resNorm = new double[x.Length];
            double sum = x.Sum(z => Math.Exp(z));
            for (int i = 0; i < x.Length; i++)
            {
                resNorm[i] = Math.Exp(x[i]) / sum;
            }
            return resNorm;
        }

    }
}
