using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackpropagationNN
{
    public static class ErrorFunctions
    {
        public static double MSE_Error(double[] actual, double[] predicted)
        {
            double sum_square_error = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                sum_square_error += (actual[i] - predicted[i]) * (actual[i] - predicted[i]);
            }
            return sum_square_error / actual.Length;
        }

        public static double Cross_Entropy(double[] actual, double[] predicted)
        {
            double error_sum = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                error_sum += ((actual[i] * Math.Log(predicted[i])) + ((1 - actual[i]) * Math.Log(1 - predicted[i]))) * (-1.0);
            }
            return error_sum;
        }

    }
}
