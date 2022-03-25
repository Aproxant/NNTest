using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackpropagationNN
{
    public static class ErrorFunctions
    {
        /*
        public static double Cross_Entropy(double[] actual, double[] predicted)
        {
            double error_sum = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                error_sum += ((actual[i] * Math.Log(predicted[i])) + ((1 - actual[i]) * Math.Log(1 - predicted[i]))) * (-1.0);
            }
            return error_sum;
        }
        */
        public static double Cross_Entropy(double[] actual, double[] predicted)
        {
            double error_sum = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                error_sum += (-1)*((actual[i] * Math.Log(predicted[i])));
                if(Double.IsNaN(error_sum))
                    Console.WriteLine(error_sum);
            }
            return error_sum;
        }

    }
}
