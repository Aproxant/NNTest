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
        public static double[] SoftMax(double[] normalizeOuputs)
        {
            double[] resNorm = new double[normalizeOuputs.Length];
            double sum = normalizeOuputs.Sum(x => Math.Exp(x));
            for (int i = 0; i < normalizeOuputs.Length; i++)
            {
                resNorm[i] = Math.Exp(normalizeOuputs[i]) / sum;
            }
            return resNorm;
        }
    }
}
