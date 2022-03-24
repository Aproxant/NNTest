using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using BackpropagationNN;

namespace NNTest
{
    class Program
    {
        static void Main(string[] args)
        {

            double[][] data = DataPreparation.Read_Data("C:\\Users\\TP\\Downloads\\winequality-red.csv");

            var trainTest = DataPreparation.Split_data(data);

            trainTest = DataPreparation.SchufleData(trainTest,30);

            var final = DataPreparation.train_and_test(trainTest,0.7);

            int[] networkStruc = { trainTest.Item1[0].Length,2,trainTest.Item2[0].Length };

            var neuron=new NeuralNetwork(networkStruc);
            neuron.Train(trainTest.Item1, trainTest.Item2, 5, 0.1,0.9);

            return;
        }
    }
}
