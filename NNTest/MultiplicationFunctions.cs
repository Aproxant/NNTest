using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackpropagationNN
{
    public static class MultiplicationFunctions
    {
        public static double[,] arrayToMatrix(double[] arr)
        {
            double[,] result = new double[arr.Length, 1];
            for (int i = 0; i < result.Length; i++)
                result[i, 0] = arr[i];
            return result;
        }
        public static double[] matrixToArray(double[,] arr)
        {
            double[] result = new double[arr.Length];
            for (int i = 0; i < result.Length; i++)
                result[i] = arr[i, 0];
            return result;
        }
        public static double[,] multiplyVectors(double[] one, double[] two)
        {
            double[,] result = new double[one.Length, two.Length];
            for (int i = 0; i < one.Length; i++)
            {
                for (int j = 0; j < two.Length; j++)
                {
                    result[i, j] = one[i] * two[j];
                }
            }
            return result;
        }
        public static double[,] matrixTranspose(double[,] mat)
        {
            int w = mat.GetLength(0);
            int h = mat.GetLength(1);

            double[,] result = new double[h, w];

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    result[j, i] = mat[i, j];
                }
            }
            return result;
        }
        public static double[,] multiplyMatrix(double[,] one, double[,] two)
        {
            int rA = one.GetLength(0);
            int cA = one.GetLength(1);
            int rB = two.GetLength(0);
            int cB = two.GetLength(1);
            double temp = 0;
            double[,] result = new double[rA, cB];
            for (int i = 0; i < rA; i++)
            {
                for (int j = 0; j < cB; j++)
                {
                    temp = 0;
                    for (int k = 0; k < cA; k++)
                    {
                        temp += one[i, k] * two[k, j];
                    }
                    result[i, j] = temp;
                }
            }
            return result;
        }

    }
}
