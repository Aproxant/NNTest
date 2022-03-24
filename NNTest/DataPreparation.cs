using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Microsoft.Toolkit.Extensions;

namespace NNTest
{
    public static class DataPreparation
    {
        public static Random rnd=new Random();
        public static double[][] Read_Data(string path)
        {
            string[] values = File.ReadAllLines(path)
                                           .Skip(1).ToArray();
            double[][] data = new double[values.Length][];
            for(int i=0;i<values.Length;i++)
            {
                values[i]=values[i].Replace('.', ',');
                data[i] = Array.ConvertAll(values[i].Split(';'), Double.Parse);
            }
            return data;
        }
        public static Tuple<double[][], double[][]> SchufleData(Tuple<double[][], double[][]> data, int nrOfShuffles)
        {
            for(int i=0;i<nrOfShuffles;i++)
            {
                int one = rnd.Next(0, data.Item2.Length);
                int two = rnd.Next(0, data.Item2.Length);
                double[] tmp1 = data.Item1[one];
                data.Item1[one] = data.Item1[two];
                data.Item1[two] = tmp1;

                double[] tmp2 = data.Item2[one];
                data.Item2[one] = data.Item2[two];
                data.Item2[two] = tmp2;
            }
            return data;
        }
        
        
        public static Tuple<double[][], double[][]> Split_data(double[][] data)
        {
            double[][] train = new double[data.GetLength(0)][];
            double[] test = new double[data.GetLength(0)];
            for(int i=0;i<data.GetLength(0);i++)
            {
                for(int j=0;j<data[i].Length-1; j++)
                {
                    train[i] = new double[data[i].Length - 1];
                    Array.Copy(data[i], 0, train[i], 0, data[i].Length - 1);
                }
                test[i] = data[i][data[i].Length - 1];

            }
            double[] dis = test.Distinct().ToArray();


            //map target to output
            Dictionary<double, int> mapped_Values = new Dictionary<double, int>();//change to categorical also
            for(int i=0;i<dis.Length;i++)
            {
                mapped_Values.Add(dis[i], i);
            }
            double[][] testFinal = new double[test.Length][];
            for(int i=0;i<test.Length;i++)
            {
                testFinal[i] = new double[dis.Length];
                for (int j=0;j<dis.Length;j++)
                {
                    testFinal[i][j] = 0;
                }
                
                testFinal[i][mapped_Values[test[i]]] = 1;
            }
            return new Tuple<double[][], double[][]>(train, testFinal);
        }



    }
}
